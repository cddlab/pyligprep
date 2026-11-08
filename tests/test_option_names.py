from dataclasses import asdict
from unittest.mock import patch

import pytest

from pyligprep import cli
from pyligprep.artifacts import Stage
from pyligprep.chemistry import load_molecule
from pyligprep.optimization import OptimizationOptions
from pyligprep.quantum import QuantumOptions


@pytest.mark.parametrize(
  "command,target,quantum_index",
  [("opt", "pyligprep.optimization.optimize", 3), ("run", "pyligprep.pipeline.run_pipeline", 4)],
)
def test_gaussian_option_names_reach_calculation(command, target, quantum_index, ethanol_pdb, tmp_path):
  with patch(target) as call:
    assert (
      cli.main(
        [
          command,
          "-i",
          str(ethanol_pdb),
          "-o",
          str(tmp_path / command),
          "--charge",
          "0",
          "--multiplicity",
          "1",
          "--opt-maxcycles",
          "407",
          "--opt-maxstep",
          "7",
          "--scf-maxcycle",
          "321",
        ]
      )
      == 0
    )
  assert call.call_args.args[2].opt_max_cycles == 407
  assert call.call_args.args[2].opt_max_step == 7
  assert call.call_args.args[quantum_index].scf_max_cycle == 321


def test_scf_maxcycle_reaches_resp(ethanol_pdb, tmp_path):
  with patch("pyligprep.charges.calculate_charges") as call:
    assert (
      cli.main(
        [
          "resp",
          "-i",
          str(ethanol_pdb),
          "-o",
          str(tmp_path / "resp"),
          "--charge",
          "0",
          "--multiplicity",
          "1",
          "--scf-maxcycle",
          "321",
        ]
      )
      == 0
    )
  assert call.call_args.args[3].scf_max_cycle == 321


@pytest.mark.parametrize("command", ["opt", "run"])
@pytest.mark.parametrize("flag", ["--max-steps", "--opt-trust", "--max-scf-cycles"])
def test_old_option_names_are_rejected(command, flag, capsys):
  with pytest.raises(SystemExit) as error:
    cli.make_parser().parse_args([command, "--resume", "unused", flag, "10"])
  assert error.value.code == 2
  assert f"unrecognized arguments: {flag}" in capsys.readouterr().err


def test_renamed_defaults_and_saved_keys():
  optimization = OptimizationOptions().to_dict()
  quantum = asdict(QuantumOptions())
  assert optimization["opt_max_cycles"] == 200
  assert OptimizationOptions().opt_max_step is None
  assert quantum["scf_max_cycle"] == 512
  assert OptimizationOptions(opt_max_step=10).to_dict()["opt_max_step"] == 10
  assert not {"max_steps", "opt_trust"} & optimization.keys()
  assert "max_scf_cycles" not in quantum


@pytest.mark.parametrize(
  "command,target,quantum_index",
  [("opt", "pyligprep.optimization.optimize", 3), ("run", "pyligprep.pipeline.run_pipeline", 4)],
)
@pytest.mark.parametrize("override", [False, True])
def test_resume_uses_renamed_settings(command, target, quantum_index, override, ethanol_pdb, tmp_path):
  molecule = load_molecule(ethanol_pdb, 0, 1)
  settings = {
    "optimization": OptimizationOptions(opt_max_cycles=213, opt_max_step=3).to_dict(),
    "quantum": asdict(QuantumOptions(scf_max_cycle=256)),
  }
  output = tmp_path / command
  with Stage(output, command, molecule.to_dict(), settings) as stage:
    stage.complete({"molecule": molecule.to_dict()})
  args = [command, "--resume", str(output)]
  if override:
    args.extend(["--opt-maxcycles", "413", "--opt-maxstep", "10", "--scf-maxcycle", "1024"])
  with patch(target) as call:
    assert cli.main(args) == 0
  assert call.call_args.args[2].opt_max_cycles == (413 if override else 213)
  assert call.call_args.args[2].opt_max_step == (10 if override else 3)
  assert call.call_args.args[quantum_index].scf_max_cycle == (1024 if override else 256)


@pytest.mark.parametrize(
  "group,key,value",
  [("optimization", "max_steps", 200), ("optimization", "opt_trust", 10), ("quantum", "max_scf_cycles", 512)],
)
def test_old_saved_names_have_an_actionable_error(group, key, value, ethanol_pdb, tmp_path, capsys):
  molecule = load_molecule(ethanol_pdb, 0, 1)
  output = tmp_path / "opt"
  with Stage(output, "opt", molecule.to_dict(), {group: {key: value}}) as stage:
    stage.complete({"molecule": molecule.to_dict()})
  state_path = output / "tmp" / "state.json"
  before = state_path.read_bytes()
  with patch("pyligprep.optimization.optimize") as call:
    assert cli.main(["opt", "--resume", str(output)]) == 2
  call.assert_not_called()
  error = capsys.readouterr().err
  assert f"Unsupported saved {group} settings: {key}" in error
  assert "new output directory" in error
  assert state_path.read_bytes() == before
