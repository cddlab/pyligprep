import os
from dataclasses import asdict
from unittest.mock import patch

import pytest

from pyligprep import cli
from pyligprep.artifacts import Stage
from pyligprep.chemistry import load_molecule
from pyligprep.errors import InputError
from pyligprep.optimization import OptimizationOptions
from pyligprep.quantum import QuantumOptions, build_molecule


@pytest.fixture(autouse=True)
def isolated_slurm(monkeypatch):
  for name in tuple(os.environ):
    if name.startswith("SLURM_"):
      monkeypatch.delenv(name)
  monkeypatch.setattr(os, "sched_getaffinity", lambda pid: set(range(16)), raising=False)


@pytest.mark.parametrize(
  "environment,expected",
  [
    ({"SLURM_CPUS_ON_NODE": "8", "SLURM_CPUS_PER_TASK": "1", "SLURM_NTASKS": "1"}, 1),
    ({"SLURM_CPUS_ON_NODE": "8", "SLURM_CPUS_PER_TASK": "8", "SLURM_NTASKS": "1"}, 8),
    ({"SLURM_CPUS_ON_NODE": "32", "SLURM_CPUS_PER_TASK": "4", "SLURM_NTASKS": "1"}, 4),
    ({"SLURM_CPUS_ON_NODE": "16", "SLURM_CPUS_PER_TASK": "8", "SLURM_NTASKS": "1"}, 8),
    ({"SLURM_CPUS_ON_NODE": "8", "SLURM_NTASKS": "1"}, 1),
    ({"SLURM_CPUS_PER_TASK": "4"}, 4),
    ({"SLURM_STEP_ID": "0", "SLURM_STEP_NUM_TASKS": "1", "SLURM_CPUS_PER_TASK": "2", "SLURM_NTASKS": "8"}, 2),
    ({"SLURM_STEPID": "1", "SLURM_NTASKS": "1", "SLURM_CPUS_ON_NODE": "8"}, 1),
    ({}, 1),
  ],
)
def test_auto_threads_follow_local_slurm_allocation(monkeypatch, environment, expected):
  from pyligprep.resources import resolve_threads

  monkeypatch.setenv("SLURM_JOB_ID", "123")
  for name, value in environment.items():
    monkeypatch.setenv(name, value)
  assert resolve_threads() == expected


@pytest.mark.parametrize("cpus", [set(range(8)), set(range(8, 16)), {1, 3, 5, 7, 9, 11, 13, 15}])
def test_affinity_uses_cpu_count_without_rebinding(monkeypatch, cpus):
  from pyligprep.resources import resolve_threads

  monkeypatch.setenv("SLURM_JOB_ID", "123")
  monkeypatch.setenv("SLURM_CPUS_ON_NODE", "16")
  monkeypatch.setenv("SLURM_CPUS_PER_TASK", "16")
  monkeypatch.setattr(os, "sched_getaffinity", lambda pid: cpus)
  with patch.object(os, "sched_setaffinity", create=True) as bind:
    assert resolve_threads() == 8
  bind.assert_not_called()


@pytest.mark.parametrize("requested", [0, -1, True, 1.5])
def test_invalid_thread_counts_are_rejected(requested):
  from pyligprep.resources import resolve_threads

  with pytest.raises(InputError, match="positive integer"):
    resolve_threads(requested)


def test_explicit_count_cannot_exceed_slurm_or_affinity(monkeypatch):
  from pyligprep.resources import resolve_threads

  monkeypatch.setenv("SLURM_JOB_ID", "123")
  monkeypatch.setenv("SLURM_CPUS_ON_NODE", "8")
  monkeypatch.setenv("SLURM_CPUS_PER_TASK", "8")
  assert resolve_threads(4) == 4
  with pytest.raises(InputError, match="exceeds.*8"):
    resolve_threads(16)
  monkeypatch.setattr(os, "sched_getaffinity", lambda pid: {8, 10})
  with pytest.raises(InputError, match="exceeds.*2"):
    resolve_threads(4)


def test_non_slurm_defaults_and_explicit_override(monkeypatch):
  from pyligprep.resources import resolve_threads

  monkeypatch.setenv("SLURM_CPUS_PER_TASK", "2")  # No job ID: not a running Slurm job.
  assert resolve_threads() == 8
  assert resolve_threads(3) == 3


@pytest.mark.parametrize("unavailable", ["missing", "error"])
def test_affinity_unavailable_uses_slurm_count(monkeypatch, unavailable):
  from pyligprep.resources import resolve_threads

  monkeypatch.setenv("SLURM_JOBID", "123")
  monkeypatch.setenv("SLURM_CPUS_PER_TASK", "8")
  if unavailable == "missing":
    monkeypatch.delattr(os, "sched_getaffinity")
  else:

    def unavailable_affinity(pid):
      raise OSError("unavailable")

    monkeypatch.setattr(os, "sched_getaffinity", unavailable_affinity)
  assert resolve_threads() == 8


@pytest.mark.parametrize("value", ["", "0", "-1", "eight", "8(x2)"])
def test_invalid_slurm_count_is_explained(monkeypatch, value):
  from pyligprep.resources import resolve_threads

  monkeypatch.setenv("SLURM_JOB_ID", "123")
  monkeypatch.setenv("SLURM_CPUS_ON_NODE", value)
  with pytest.raises(InputError, match="SLURM_CPUS_ON_NODE"):
    resolve_threads()


@pytest.mark.parametrize(
  "environment",
  [
    {"SLURM_NTASKS": "8", "SLURM_CPUS_PER_TASK": "1", "SLURM_CPUS_ON_NODE": "8"},
    {"SLURM_NTASKS": "32", "SLURM_JOB_NUM_NODES": "4"},
    {"SLURM_STEP_ID": "0", "SLURM_STEP_NUM_TASKS": "8", "SLURM_NTASKS": "1", "SLURM_CPUS_PER_TASK": "8"},
    {"SLURM_STEP_ID": "batch", "SLURM_STEP_NUM_TASKS": "1", "SLURM_NTASKS": "8"},
  ],
)
def test_multiple_tasks_are_rejected(monkeypatch, environment):
  from pyligprep.resources import resolve_threads

  monkeypatch.setenv("SLURM_JOB_ID", "123")
  for name, value in environment.items():
    monkeypatch.setenv(name, value)
  with pytest.raises(InputError, match="single Slurm task.*--ntasks=1 --cpus-per-task"):
    resolve_threads()


@pytest.mark.parametrize(
  "command,target,quantum_index",
  [
    ("opt", "pyligprep.optimization.optimize", 3),
    ("resp", "pyligprep.charges.calculate_charges", 3),
    ("run", "pyligprep.pipeline.run_pipeline", 4),
  ],
)
@pytest.mark.parametrize("resume,explicit,expected", [(False, None, 4), (True, None, 4), (True, 2, 2)])
def test_cli_uses_current_allocation_even_on_resume(
  monkeypatch, command, target, quantum_index, resume, explicit, expected, ethanol_pdb, tmp_path, capsys
):
  monkeypatch.setenv("SLURM_JOB_ID", "123")
  monkeypatch.setenv("SLURM_CPUS_ON_NODE", "4")
  monkeypatch.setenv("SLURM_CPUS_PER_TASK", "4")
  output = tmp_path / command
  if resume:
    molecule = load_molecule(ethanol_pdb, 0, 1)
    settings = {"optimization": OptimizationOptions().to_dict(), "quantum": asdict(QuantumOptions(threads=16))}
    with Stage(output, command, molecule.to_dict(), settings) as stage:
      stage.complete({"molecule": molecule.to_dict()})
    args = [command, "--resume", str(output)]
  else:
    args = [command, "-i", str(ethanol_pdb), "-o", str(output), "--charge", "0", "--multiplicity", "1"]
  if explicit is not None:
    args.extend(["--threads", str(explicit)])
  with patch(target) as call:
    assert cli.main(args) == 0
  assert call.call_args.args[quantum_index].threads == expected
  assert f"CPU threads: {expected}" in capsys.readouterr().out


def test_oversubscription_stops_before_output_or_gpu(monkeypatch, ethanol_pdb, tmp_path, capsys):
  monkeypatch.setenv("SLURM_JOB_ID", "123")
  monkeypatch.setenv("SLURM_CPUS_ON_NODE", "4")
  monkeypatch.setenv("SLURM_CPUS_PER_TASK", "4")
  output = tmp_path / "opt"
  with patch("pyligprep.quantum.gpu_probe", side_effect=AssertionError("GPU must not start")):
    assert (
      cli.main(
        [
          "opt",
          "-i",
          str(ethanol_pdb),
          "-o",
          str(output),
          "--charge",
          "0",
          "--multiplicity",
          "1",
          "--threads",
          "8",
        ]
      )
      == 2
    )
  assert not output.exists()
  assert "exceeds" in capsys.readouterr().err


def test_pyscf_receives_detected_threads(monkeypatch, ethanol_pdb, tmp_path):
  from pyligprep.resources import resolve_threads

  monkeypatch.setenv("SLURM_JOB_ID", "123")
  monkeypatch.setenv("SLURM_CPUS_ON_NODE", "8")
  monkeypatch.setenv("SLURM_CPUS_PER_TASK", "8")
  monkeypatch.setattr(os, "sched_getaffinity", lambda pid: set(range(8, 16)))
  molecule = load_molecule(ethanol_pdb, 0, 1)
  with patch("pyscf.lib.num_threads") as set_threads, patch("pyscf.gto.M"):
    build_molecule(molecule, "sto-3g", QuantumOptions(threads=resolve_threads()), tmp_path / "qm.log")
  set_threads.assert_called_once_with(8)
