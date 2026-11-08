from dataclasses import replace
from decimal import Decimal

import numpy as np
import pytest

from pyligprep.amber import export_parameters
from pyligprep.artifacts import Stage, completed_result, load_state
from pyligprep.charges import RespOptions, calculate_charges
from pyligprep.chemistry import load_molecule
from pyligprep.errors import CalculationError, InputError
from pyligprep.files import digest, file_hash, read_json
from pyligprep.optimization import OptimizationOptions, optimize
from pyligprep.quantum import QuantumOptions


def test_stage_resume_layout_and_tamper_detection(tmp_path):
  root = tmp_path / "stage"
  with pytest.raises(RuntimeError):
    with Stage(root, "opt", {"input": 1}, {"settings": 1}):
      raise RuntimeError("interrupted")
  assert load_state(root)["status"] == "failed"
  with Stage(root, "opt", {"input": 1}, {"settings": 1}, resume=True) as stage:
    assert stage.resumed and stage.prior_settings == {"settings": 1}
    out, chk = stage.directory / "out.txt", stage.directory / "scf.chk"
    out.write_text("deliverable\n")
    chk.write_text("checkpoint\n")
    stage.mark("scf", "fp", ["scf.chk"])
    stage.complete({"value": 4, "files": {"out.txt": file_hash(out)}}, deliverables=("out.txt",), discard=("scf.chk",))
  assert not chk.exists() and "scf" not in load_state(root)["steps"]
  assert sorted(p.name for p in root.iterdir()) == ["out.txt", "tmp", "versions.json"]
  assert read_json(root / "versions.json")["pyligprep"]
  assert completed_result(root, "opt")["value"] == 4
  with Stage(root, "opt", {"input": 1}, {"settings": 1}, resume=True) as stage:
    assert stage.cached["value"] == 4
  with pytest.raises(InputError, match="differs"):
    with Stage(root, "opt", {"input": 2}, {"settings": 1}, resume=True):
      pass
  (root / "tmp" / "out.txt").write_text("tampered\n")
  with pytest.raises(InputError, match="modified"):
    completed_result(root, "opt")


def test_old_format_and_missing_state_are_explained(tmp_path):
  old = tmp_path / "old"
  old.mkdir()
  (old / "manifest.json").write_text("{}")
  with pytest.raises(InputError, match="old output format"):
    load_state(old)
  with pytest.raises(InputError, match="old output format"):
    with Stage(old, "opt", {}, {}, resume=True):
      pass
  with pytest.raises(InputError, match="old output format"):
    completed_result(old, "opt")
  with pytest.raises(InputError, match="No stage state"):
    load_state(tmp_path / "missing")


def test_substep_cache_and_molecule_change_reset(tmp_path):
  root = tmp_path / "resp"
  with pytest.raises(RuntimeError):
    with Stage(root, "resp", {"input": 1}, {}) as stage:
      (stage.directory / "hf.chk").write_text("hf\n")
      stage.mark("hf", "key", ["hf.chk"])
      raise RuntimeError("ESP failed")
  with Stage(root, "resp", {"input": 1}, {}, resume=True) as stage:
    assert stage.reusable("hf", "key")
    assert not stage.reusable("hf", "other")
    (stage.directory / "hf.chk").write_text("edited\n")
    assert not stage.reusable("hf", "key")
    stage.forget("hf")
    assert "hf" not in load_state(root)["steps"]
  with Stage(root, "resp", {"input": 2}, {}, resume=True) as stage:
    assert not stage.resumed
    assert not (stage.directory / "hf.chk").exists()


def test_resp_resume_reuses_esp_after_fit_failure(ethanol_pdb, tmp_path, monkeypatch):
  from pyligprep import charges

  molecule = load_molecule(ethanol_pdb, 0, 1)
  quantum = QuantumOptions(device="cpu", threads=1)
  output = tmp_path / "resp"
  with pytest.raises(CalculationError, match="converge"):
    calculate_charges(molecule, output, RespOptions(max_iterations=1), quantum)

  def must_not_run(*args, **kwargs):
    raise AssertionError("HF must not run when ESP has been completed")

  monkeypatch.setattr(charges, "build_molecule", must_not_run)
  result = calculate_charges(molecule, output, RespOptions(), quantum, resume=True)
  assert abs(sum(result["charges"])) < 1e-8
  assert sorted(p.name for p in output.iterdir()) == ["charges.json", "tmp", "versions.json"]
  assert read_json(output / "charges.json")["charges"] == result["charges"]
  assert not (output / "tmp" / "hf.chk").exists()


def test_hf_is_reused_after_esp_failure(ethanol_pdb, tmp_path, monkeypatch):
  from pyligprep import charges

  molecule = load_molecule(ethanol_pdb, 0, 1)
  quantum = QuantumOptions(device="cpu", threads=1)
  output = tmp_path / "resp"
  evaluate = charges.evaluate_esp

  def interrupted(*args, **kwargs):
    raise CalculationError("ESP interrupted")

  monkeypatch.setattr(charges, "evaluate_esp", interrupted)
  with pytest.raises(CalculationError, match="ESP interrupted"):
    calculate_charges(molecule, output, quantum=quantum)
  assert (output / "tmp" / "hf.chk").exists()

  def forbidden_scf(*args, **kwargs):
    raise RuntimeError("SCF was requested")

  monkeypatch.setattr(charges, "evaluate_esp", evaluate)
  monkeypatch.setattr(charges, "run_scf", forbidden_scf)
  result = calculate_charges(molecule, output, quantum=quantum, resume=True)
  assert abs(sum(result["charges"])) < 1e-8


def test_changed_fit_reuses_esp_but_completed_stage_drops_hf_checkpoint(ethanol_pdb, tmp_path, monkeypatch):
  from pyligprep import charges

  molecule = load_molecule(ethanol_pdb, 0, 1)
  quantum = QuantumOptions(device="cpu", threads=1)
  output = tmp_path / "resp"
  calculate_charges(molecule, output, quantum=quantum)

  def forbidden_scf(*args, **kwargs):
    raise RuntimeError("SCF was requested")

  monkeypatch.setattr(charges, "run_scf", forbidden_scf)
  result = calculate_charges(molecule, output, RespOptions(qwt2=0.002), quantum, resume=True)
  assert abs(sum(result["charges"])) < 1e-8
  # hf.chk is deleted once a stage completes, so a new ESP grid needs a fresh HF.
  with pytest.raises(RuntimeError, match="SCF was requested"):
    calculate_charges(molecule, output, RespOptions(density=1.2), quantum, resume=True)
  with pytest.raises(RuntimeError, match="SCF was requested"):
    calculate_charges(
      molecule.with_coords(molecule.coords + 0.01),
      output,
      RespOptions(density=1.2),
      quantum,
      resume=True,
    )


def test_stage_lock_rejects_concurrent_writer(tmp_path):
  output = tmp_path / "stage"
  with Stage(output, "opt", {}, {}) as outer:
    with pytest.raises(InputError, match="lock"):
      with Stage(output, "opt", {}, {}, resume=True):
        pass
    outer.complete({"done": True})


@pytest.mark.amber
def test_amber_exports_preserve_identity_and_charges(ethanol_pdb, tmp_path, amber_bin):
  molecule = load_molecule(ethanol_pdb, 0, 1)
  charges = [-0.1, 0.1, -0.6, 0.05, 0.05, 0.05, 0.1, 0.1, 0.25]
  assert abs(sum(charges)) < 1e-9
  result = export_parameters(
    {"molecule": molecule.to_dict(), "charges": charges},
    tmp_path / "parameters",
    amber_bin,
  )
  assert result["charges"] == charges
  assert all((tmp_path / "parameters" / name).is_file() for name in ("ligand.mol2", "ligand.prep", "ligand.frcmod"))


@pytest.mark.amber
def test_export_rounds_charges_conserves_total_and_preserves_equivalence(ethanol_pdb, tmp_path, amber_bin):
  from pyligprep.amber import read_mol2
  from pyligprep.charge_rounding import final_equivalence_groups
  from pyligprep.resp import stage_constraints

  molecule = load_molecule(ethanol_pdb, 0, 1)
  raw = [
    -0.100004,
    0.100004,
    -0.600002,
    0.050004,
    0.050004,
    0.050004,
    0.100002,
    0.100002,
    0.249986,
  ]
  constraints = stage_constraints(molecule)
  source = {
    "molecule": molecule.to_dict(),
    "charges": raw,
    "constraints": constraints,
  }
  output = tmp_path / "parameters"
  result = export_parameters(source, output, amber_bin)
  assert result["resp_charges"] == source["charges"] == raw
  atoms, _ = read_mol2(output / "ligand.mol2")
  texts = [a["charge_text"] for a in atoms]
  assert all(len(q.split(".")[1]) == 6 and q.endswith("0") for q in texts)
  assert sum(Decimal(q) for q in texts) == 0
  for group in final_equivalence_groups(molecule, constraints):
    assert len({texts[i] for i in group}) == 1
  assert (output / "ligand.prep").is_file()
  assert not (output / "ligand.prepi").exists()
  prep = (output / "ligand.prep").read_text().splitlines()
  prep_charges = {w[1]: w[10] for line in prep if len(w := line.split()) >= 11 and w[0].isdigit() and w[2] != "DU"}
  assert [prep_charges[a["name"]] for a in atoms] == texts
  report = read_json(output / "tmp" / "charge-rounding.json")
  assert report["residual_before_correction_units"] != 0
  assert sum(report["corrected_units"]) == 0
  versions = read_json(output / "versions.json")
  assert set(versions["binaries"]) >= {"ambertools", "antechamber", "parmchk2", "tleap"}
  assert digest(export_parameters(source, output, amber_bin, resume=True)) == digest(result)


@pytest.mark.amber
def test_old_export_cache_is_rebuilt_with_prep_extension(ethanol_pdb, tmp_path, amber_bin):
  molecule = load_molecule(ethanol_pdb, 0, 1)
  source = {"molecule": molecule.to_dict(), "charges": [0.0] * len(molecule.atoms)}
  output = tmp_path / "parameters"
  with Stage(output, "export", molecule.to_dict(), {"force_field": "gaff2"}) as stage:
    legacy = stage.directory / "ligand.prepi"
    legacy.write_text("old artifact retained\n")
    stage.complete({**source, "files": {"ligand.prepi": file_hash(legacy)}})
  export_parameters(source, output, amber_bin, resume=True)
  assert legacy.read_text() == "old artifact retained\n"
  assert (output / "ligand.prep").exists()
  assert load_state(output)["status"] == "complete"


@pytest.mark.gpu
@pytest.mark.amber
def test_gpu_pipeline_and_optimizer_restart(ethanol_pdb, tmp_path, amber_bin):
  molecule = load_molecule(ethanol_pdb, 0, 1)
  quantum = QuantumOptions(device="gpu", threads=4, memory_mb=12000)
  output = tmp_path / "opt"
  options = OptimizationOptions(opt_max_cycles=1)
  with pytest.raises(CalculationError, match="Geometry did not converge"):
    optimize(molecule, output, options, quantum)
  berny = optimize(molecule, output, replace(options, opt_max_cycles=100), quantum, resume=True)
  assert berny["converged"]
  from pyligprep.model import Molecule

  final = Molecule.from_dict(berny["molecule"])
  geometric = optimize(
    final,
    tmp_path / "geometric",
    OptimizationOptions(optimizer="geometric", opt_max_cycles=10),
    quantum,
  )
  assert geometric["converged"]
  result = calculate_charges(final, tmp_path / "resp", quantum=quantum)
  assert abs(sum(result["charges"])) < 1e-8
  export_parameters(result, tmp_path / "parameters", amber_bin)
  cached = calculate_charges(final, tmp_path / "resp", quantum=quantum, resume=True)
  np.testing.assert_array_equal(cached["charges"], result["charges"])


@pytest.mark.gpu
@pytest.mark.amber
def test_gpu_cli_run_and_cached_resume(ethanol_pdb, tmp_path, amber_bin):
  from pyligprep.cli import main

  output = tmp_path / "run"
  assert (
    main(
      [
        "run",
        "-i",
        str(ethanol_pdb),
        "-o",
        str(output),
        "--charge",
        "0",
        "--multiplicity",
        "1",
        "--device",
        "gpu",
        "--threads",
        "4",
        "--amber-bin",
        str(amber_bin),
        "--keep-tmp",
      ]
    )
    == 0
  )
  opt_state = read_json(output / "tmp" / "opt" / "state.json")
  assert not list((output / "tmp").rglob("*.chk"))
  for name in ("ligand.mol2", "ligand.prep", "ligand.frcmod"):
    assert (output / name).read_bytes() == (output / "tmp" / "export" / name).read_bytes()
  assert main(["run", "--resume", str(output)]) == 0
  assert opt_state["status"] == "complete"
  assert sorted(p.name for p in output.iterdir()) == ["ligand.frcmod", "ligand.mol2", "ligand.prep", "versions.json"]
