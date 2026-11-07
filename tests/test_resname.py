from dataclasses import asdict
from unittest.mock import patch

import pytest

from pyligprep import cli
from pyligprep.amber import export_parameters, read_mol2
from pyligprep.artifacts import Stage, load_state
from pyligprep.chemistry import load_molecule
from pyligprep.files import digest
from pyligprep.optimization import OptimizationOptions
from pyligprep.pipeline import run_pipeline
from pyligprep.quantum import QuantumOptions


def test_cli_run_resname_and_saved_value(ethanol_pdb, tmp_path):
  mol = load_molecule(ethanol_pdb, 0, 1)
  root = tmp_path / "run"
  settings = {
    "optimization": asdict(OptimizationOptions()),
    "quantum": asdict(QuantumOptions()),
    "resname": "SiN",
  }
  with Stage(root, "run", mol.to_dict(), settings) as stage:
    stage.complete({"molecule": mol.to_dict()})
  with patch("pyligprep.pipeline.run_pipeline") as run:
    assert (
      cli.main(
        [
          "run",
          "-i",
          str(ethanol_pdb),
          "-o",
          str(tmp_path / "new"),
          "--charge",
          "0",
          "--multiplicity",
          "1",
          "-r",
          "SIN",
        ]
      )
      == 0
    )
    assert run.call_args.kwargs["resname"] == "SIN"
    assert run.call_args.kwargs["keep_tmp"] is False
    assert cli.main(["run", "--resume", str(root), "--keep-tmp"]) == 0
    assert run.call_args.kwargs["keep_tmp"] is True
    assert run.call_args.kwargs["resname"] == "SiN"
    assert cli.main(["run", "--resume", str(root), "--resname", "atp"]) == 0
    assert run.call_args.kwargs["keep_tmp"] is False
    assert run.call_args.kwargs["resname"] == "atp"


def test_pipeline_passes_resname_only_to_export_and_retains_on_resume(ethanol_pdb, tmp_path, fake_export):
  mol = load_molecule(ethanol_pdb, 0, 1)
  result = {"molecule": mol.to_dict(), "charges": [0.0] * len(mol.atoms)}
  root = tmp_path / "run"
  with (
    patch("pyligprep.pipeline.amber_tools"),
    patch("pyligprep.pipeline.optimize", return_value=result) as opt,
    patch("pyligprep.pipeline.calculate_charges", return_value=result) as resp,
    patch("pyligprep.pipeline.export_parameters", side_effect=fake_export) as export,
  ):
    run_pipeline(mol, root, resname="SiN", keep_tmp=True)
    run_pipeline(mol, root, resume=True, keep_tmp=True)
    assert export.call_args.kwargs["resname"] == "SiN"
    assert opt.call_args.args[0] == mol
    assert resp.call_args.args[0] == mol
    assert load_state(root)["settings"]["resname"] == "SiN"
    assert all((root / f"SiN.{ext}").is_file() for ext in ("mol2", "prep", "frcmod"))


@pytest.mark.parametrize("name", ["", "../", "A/B", "A B", "A;B", "123", "ABCD", "ＳＩＮ"])
def test_invalid_name_fails_before_quantum_or_amber(ethanol_pdb, tmp_path, name):
  from pyligprep.errors import InputError

  mol = load_molecule(ethanol_pdb, 0, 1)
  result = {"molecule": mol.to_dict(), "charges": [0.0] * len(mol.atoms)}
  with patch("pyligprep.pipeline.optimize", side_effect=AssertionError("QM started")):
    with pytest.raises(InputError, match="residue name"):
      run_pipeline(mol, tmp_path / "run", resname=name)
  with patch("pyligprep.amber._run", side_effect=AssertionError("Amber started")):
    with pytest.raises(InputError, match="residue name"):
      export_parameters(result, tmp_path / "export", resname=name)
  assert not (tmp_path / "run").exists()
  assert not (tmp_path / "export").exists()


@pytest.mark.amber
@pytest.mark.parametrize("name", ["SIN", "SiN", "a1"])
def test_named_export_content_case_and_resume(ethanol_pdb, tmp_path, amber_bin, name):
  mol = load_molecule(ethanol_pdb, 0, 1)
  source = {"molecule": mol.to_dict(), "charges": [0.0] * len(mol.atoms)}
  root = tmp_path / "export"
  result = export_parameters(source, root, amber_bin, resname=name)
  assert all((root / f"{name}.{ext}").exists() for ext in ("mol2", "prep", "frcmod"))
  assert not (root / "ligand.mol2").exists()
  assert source["molecule"] == mol.to_dict()
  assert all(a["residue"] == name for a in result["molecule"]["atoms"])
  atoms, _ = read_mol2(root / f"{name}.mol2")
  assert {a["residue"] for a in atoms} == {name}
  mol2 = (root / f"{name}.mol2").read_text()
  assert mol2.split("@<TRIPOS>MOLECULE\n")[1].splitlines()[0] == name
  assert mol2.split("@<TRIPOS>SUBSTRUCTURE\n")[1].split()[1] == name
  assert [
    line.split()[0]
    for line in (root / f"{name}.prep").read_text().splitlines()
    if len(line.split()) == 3 and line.split()[1] == "INT"
  ] == [name]
  assert sorted(p.name for p in root.iterdir()) == sorted(
    [f"{name}.{ext}" for ext in ("mol2", "prep", "frcmod")] + ["tmp", "versions.json"]
  )
  with patch("pyligprep.amber._run", side_effect=AssertionError("cached export reran AmberTools")):
    assert digest(export_parameters(source, root, amber_bin, resume=True)) == digest(result)
    assert cli.main(["export", "--resume", str(root)]) == 0


@pytest.mark.amber
def test_rename_export_preserves_manual_edits_and_cli_short_option(ethanol_pdb, tmp_path, amber_bin):
  mol = load_molecule(ethanol_pdb, 0, 1)
  source = {"molecule": mol.to_dict(), "charges": [0.0] * len(mol.atoms)}
  root = tmp_path / "export"
  export_parameters(source, root, amber_bin)
  edited = root / "ligand.mol2"
  edited.write_text("user's hand edited old output\n")
  assert cli.main(["export", "--resume", str(root), "-r", "SIN"]) == 0
  assert edited.read_text() == "user's hand edited old output\n"
  assert (root / "SIN.mol2").exists()
  assert load_state(root)["settings"]["resname"] == "SIN"
  assert cli.main(["export", "--resume", str(root), "--resname", "sin"]) == 0
  assert (root / "sin.mol2").exists() and (root / "SIN.mol2").exists()
  (root / "sin.prep").write_text("user's hand edited new output\n")
  assert cli.main(["export", "--resume", str(root)]) == 2
  assert (root / "sin.prep").read_text() == "user's hand edited new output\n"


@pytest.mark.amber
def test_cli_from_completed_resp_with_resname(ethanol_pdb, tmp_path, amber_bin):
  mol = load_molecule(ethanol_pdb, 0, 1)
  source = tmp_path / "resp"
  with Stage(source, "resp", mol.to_dict(), {}) as stage:
    stage.complete({"molecule": mol.to_dict(), "charges": [0.0] * len(mol.atoms)})
  output = tmp_path / "parameters"
  assert (
    cli.main(
      [
        "export",
        "--from",
        str(source),
        "-o",
        str(output),
        "--resname",
        "SIN",
        "--amber-bin",
        str(amber_bin),
      ]
    )
    == 0
  )
  assert (output / "SIN.prep").exists()
  assert {a["residue"] for a in read_mol2(output / "SIN.mol2")[0]} == {"SIN"}


@pytest.mark.parametrize("section", ["MOLECULE", "ATOM", "SUBSTRUCTURE"])
def test_wrong_mol2_residue_field_is_rejected(ethanol_pdb, tmp_path, section):
  from pyligprep.amber import _write_input, validate_mol2
  from pyligprep.errors import CalculationError

  mol = load_molecule(ethanol_pdb, 0, 1)
  charges = [0.0] * len(mol.atoms)
  path = tmp_path / "wrong.mol2"
  _write_input(path, mol, charges)
  marker = f"@<TRIPOS>{section}\n"
  before, content = path.read_text().split(marker, 1)
  content = content.replace(mol.atoms[0].residue, "bad", 1)
  path.write_text(before + marker + content)
  with pytest.raises(CalculationError, match="residue name"):
    validate_mol2(path, mol, charges)


def test_wrong_prep_residue_is_rejected(ethanol_pdb, tmp_path):
  from pyligprep.amber import _validate_prepi
  from pyligprep.errors import CalculationError

  mol = load_molecule(ethanol_pdb, 0, 1)
  path = tmp_path / "wrong.prep"
  path.write_text("bad INT 0\n")
  with pytest.raises(CalculationError, match="residue name"):
    _validate_prepi(path, mol, [0.0] * len(mol.atoms))
