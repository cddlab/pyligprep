from dataclasses import asdict
from unittest.mock import patch

import pytest

from pyligprep import cli
from pyligprep.artifacts import Stage, completed_result
from pyligprep.chemistry import load_molecule
from pyligprep.optimization import OptimizationOptions
from pyligprep.quantum import QuantumOptions


def test_required_charge_and_multiplicity(ethanol_pdb, tmp_path, capsys):
  assert cli.main(["opt", "-i", str(ethanol_pdb), "-o", str(tmp_path / "opt")]) == 2
  assert "mandatory" in capsys.readouterr().err
  assert not (tmp_path / "opt").exists()


def test_check_without_gpu(ethanol_pdb, capsys):
  assert cli.main(["check", "-i", str(ethanol_pdb), "--charge", "0", "--multiplicity", "1"]) == 0
  assert "Valid: 9 atoms" in capsys.readouterr().out


def test_default_opt_dispatch_is_berny_and_b3lyp(ethanol_pdb, tmp_path):
  with patch("pyligprep.optimization.optimize") as call:
    assert (
      cli.main(
        [
          "opt",
          "-i",
          str(ethanol_pdb),
          "-o",
          str(tmp_path / "opt"),
          "--charge",
          "0",
          "--multiplicity",
          "1",
          "--device",
          "cpu",
        ]
      )
      == 0
    )
  options, quantum = call.call_args.args[2:4]
  assert options == OptimizationOptions()
  assert quantum.device == "cpu"


def test_resume_keeps_saved_options(ethanol_pdb, tmp_path):
  mol = load_molecule(ethanol_pdb, 0, 1)
  options = OptimizationOptions(method="PBE0-D3BJ", optimizer="geometric")
  root = tmp_path / "opt"
  with Stage(
    root,
    "opt",
    mol.to_dict(),
    {"optimization": asdict(options), "quantum": asdict(QuantumOptions())},
  ) as stage:
    stage.complete({"molecule": mol.to_dict()})
  with patch("pyligprep.optimization.optimize") as call:
    assert cli.main(["opt", "--resume", str(root), "--max-steps", "400"]) == 0
  actual = call.call_args.args[2]
  assert actual.method == "PBE0-D3BJ"
  assert actual.optimizer == "geometric"
  assert actual.max_steps == 400
  assert cli.main(["resp", "--resume", str(root)]) == 2
  assert cli.main(["opt", "--resume", str(root), "--charge", "1"]) == 2


def test_source_requires_completed_opt(ethanol_pdb, tmp_path):
  root = tmp_path / "opt"
  mol = load_molecule(ethanol_pdb, 0, 1)
  with pytest.raises(RuntimeError):
    with Stage(root, "opt", mol.to_dict(), {}):
      raise RuntimeError("failed")
  assert cli.main(["resp", "--from", str(root), "-o", str(tmp_path / "resp")]) == 2


def test_doctor_reports_missing_gpu(capsys):
  with patch("pyligprep.quantum.gpu_probe", return_value=(False, "test")):
    assert cli.main(["doctor", "--amber-bin", "/missing"]) == 0
  assert '"available": false' in capsys.readouterr().out


@pytest.mark.amber
def test_cli_resp_and_export_resume(ethanol_pdb, tmp_path, amber_bin):
  root, output = tmp_path / "resp", tmp_path / "export"
  assert (
    cli.main(
      [
        "resp",
        "-i",
        str(ethanol_pdb),
        "-o",
        str(root),
        "--charge",
        "0",
        "--multiplicity",
        "1",
        "--device",
        "cpu",
        "--threads",
        "1",
      ]
    )
    == 0
  )
  assert (
    cli.main(
      [
        "export",
        "--from",
        str(root),
        "-o",
        str(output),
        "--amber-bin",
        str(amber_bin),
      ]
    )
    == 0
  )
  assert cli.main(["export", "--resume", str(output)]) == 0
  (output / "ligand.mol2").write_text("edited by user")
  assert cli.main(["export", "--resume", str(output)]) == 2
  assert (output / "ligand.mol2").read_text() == "edited by user"


def test_run_uses_shared_stages_and_rechecks_children(ethanol_pdb, tmp_path, fake_export):
  from pyligprep.pipeline import run_pipeline

  mol = load_molecule(ethanol_pdb, 0, 1)
  geometry = {"molecule": mol.to_dict()}
  charges = {**geometry, "charges": [0.0] * len(mol.atoms)}
  with (
    patch("pyligprep.pipeline.optimize", return_value=geometry) as opt,
    patch("pyligprep.pipeline.calculate_charges", return_value=charges) as resp,
    patch("pyligprep.pipeline.export_parameters", side_effect=fake_export) as export,
  ):
    root = tmp_path / "run"
    with patch("pyligprep.pipeline.amber_tools"):
      run_pipeline(mol, root, keep_tmp=True)
      result = run_pipeline(mol, root, resume=True, keep_tmp=True)
  assert opt.call_count == resp.call_count == export.call_count == 2
  assert export.call_args.args[1] == (root / "tmp" / "export").resolve()
  assert opt.call_args.kwargs["publish"] is False
  assert sorted(result["files"]) == ["ligand.frcmod", "ligand.mol2", "ligand.prep"]
  assert completed_result(root, "run")["files"] == result["files"]
  for name in result["files"]:
    assert (root / name).read_bytes() == (root / "tmp" / "export" / name).read_bytes()
  assert sorted(p.name for p in root.iterdir()) == [
    "ligand.frcmod",
    "ligand.mol2",
    "ligand.prep",
    "tmp",
    "versions.json",
  ]


def test_run_removes_tmp_after_success_but_keeps_it_after_failure(ethanol_pdb, tmp_path, fake_export):
  from pyligprep.errors import CalculationError, InputError
  from pyligprep.files import read_json
  from pyligprep.pipeline import run_pipeline

  mol = load_molecule(ethanol_pdb, 0, 1)
  charges = {"molecule": mol.to_dict(), "charges": [0.0] * len(mol.atoms)}
  root = tmp_path / "run"
  with (
    patch("pyligprep.pipeline.amber_tools"),
    patch("pyligprep.pipeline.optimize", return_value=charges),
    patch("pyligprep.pipeline.export_parameters", side_effect=fake_export),
  ):
    with patch("pyligprep.pipeline.calculate_charges", side_effect=CalculationError("stopped in RESP")):
      with pytest.raises(CalculationError):
        run_pipeline(mol, root)
    assert read_json(root / "tmp" / "state.json")["status"] == "failed"
    with patch("pyligprep.pipeline.calculate_charges", return_value=charges):
      run_pipeline(mol, root, resume=True)
      assert sorted(p.name for p in root.iterdir()) == [
        "ligand.frcmod",
        "ligand.mol2",
        "ligand.prep",
        "versions.json",
      ]
      assert read_json(root / "versions.json")["binaries"] == {"ambertools": "AmberToolsTest"}
      assert read_json(root / "versions.json")["deliverables"] == ["ligand.mol2", "ligand.prep", "ligand.frcmod"]
      with pytest.raises(InputError, match="tmp/ was removed"):
        run_pipeline(mol, root, resume=True)
    assert cli.main(["run", "--resume", str(root)]) == 2


def test_run_refuses_to_overwrite_edited_deliverable(ethanol_pdb, tmp_path, fake_export):
  from pyligprep.errors import InputError
  from pyligprep.pipeline import run_pipeline

  mol = load_molecule(ethanol_pdb, 0, 1)
  charges = {"molecule": mol.to_dict(), "charges": [0.0] * len(mol.atoms)}
  root = tmp_path / "run"
  with (
    patch("pyligprep.pipeline.amber_tools"),
    patch("pyligprep.pipeline.optimize", return_value=charges),
    patch("pyligprep.pipeline.calculate_charges", return_value=charges),
    patch("pyligprep.pipeline.export_parameters", side_effect=fake_export),
  ):
    run_pipeline(mol, root, keep_tmp=True)
    (root / "ligand.mol2").write_text("edited by user")
    with pytest.raises(InputError, match="modified output"):
      run_pipeline(mol, root, resume=True)
  assert (root / "ligand.mol2").read_text() == "edited by user"


def test_hadd_cli(ethanol_pdb, tmp_path):
  output = tmp_path / "added.pdb"
  assert cli.main(["hadd", "-i", str(ethanol_pdb), "-o", str(output), "--smiles", "CCO"]) == 0
  assert output.exists()


@pytest.mark.amber
def test_export_failure_can_resume_from_saved_resp(ethanol_pdb, tmp_path, amber_bin, monkeypatch):
  from pyligprep import amber
  from pyligprep.errors import CalculationError

  mol = load_molecule(ethanol_pdb, 0, 1)
  result = {"molecule": mol.to_dict(), "charges": [0.0] * len(mol.atoms)}
  root = tmp_path / "export"
  runner = amber._run

  def fail(*args, **kwargs):
    raise CalculationError("simulated AmberTools failure")

  monkeypatch.setattr(amber, "_run", fail)
  with pytest.raises(CalculationError):
    amber.export_parameters(result, root, amber_bin)
  monkeypatch.setattr(amber, "_run", runner)
  assert cli.main(["export", "--resume", str(root)]) == 0


def test_run_succeeds_with_a_warning_when_tmp_cannot_be_removed(ethanol_pdb, tmp_path, fake_export, capsys):
  from pyligprep.pipeline import run_pipeline

  mol = load_molecule(ethanol_pdb, 0, 1)
  charges = {"molecule": mol.to_dict(), "charges": [0.0] * len(mol.atoms)}
  root = tmp_path / "run"
  with (
    patch("pyligprep.pipeline.amber_tools"),
    patch("pyligprep.pipeline.optimize", return_value=charges),
    patch("pyligprep.pipeline.calculate_charges", return_value=charges),
    patch("pyligprep.pipeline.export_parameters", side_effect=fake_export),
    patch("pyligprep.pipeline.shutil.rmtree", side_effect=PermissionError("denied")),
  ):
    run_pipeline(mol, root)
  assert (root / "ligand.mol2").is_file()
  assert "could not remove" in capsys.readouterr().err


def test_export_resume_without_saved_resp_input_is_explained(ethanol_pdb, tmp_path, capsys):
  mol = load_molecule(ethanol_pdb, 0, 1)
  root = tmp_path / "export"
  with pytest.raises(RuntimeError):
    with Stage(root, "export", mol.to_dict(), {"tools": {"antechamber": "/missing/antechamber"}}):
      raise RuntimeError("stopped before resp-source.json")
  assert cli.main(["export", "--resume", str(root)]) == 2
  assert "resp-source.json is missing" in capsys.readouterr().err
