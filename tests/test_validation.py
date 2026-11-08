import sys
from dataclasses import replace

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import rdDistGeom

from pyligprep import chemistry
from pyligprep.charges import RespOptions
from pyligprep.chemistry import add_hydrogens, load_molecule, read_pdb, to_rdkit
from pyligprep.errors import DependencyError, InputError
from pyligprep.esp import surface_points, unit_surface
from pyligprep.model import Molecule
from pyligprep.optimization import OptimizationOptions, optimizer_parameters
from pyligprep.quantum import QuantumOptions
from pyligprep.resp import fit_resp, validate_constraints


@pytest.mark.parametrize("smiles,charge", [("CC(=O)[O-]", -1), ("C[NH3+]", 1), ("Clc1ccccc1", 0)])
def test_hydrogenated_chemical_states(tmp_path, smiles, charge):
  full = Chem.AddHs(Chem.MolFromSmiles(smiles))
  assert rdDistGeom.EmbedMolecule(full, randomSeed=9) == 0
  source, output = tmp_path / "input.pdb", tmp_path / "output.pdb"
  Chem.MolToPDBFile(Chem.RemoveHs(full), str(source))
  prepared = add_hydrogens(source, output, smiles=smiles, charge=charge)
  assert prepared.charge == charge
  assert len(prepared.atoms) == full.GetNumAtoms()
  assert load_molecule(output, charge, 1).charge == charge


@pytest.mark.parametrize("smiles", ["CCO", "OC(=O)c1ccccc1", "O=[N+]([O-])c1ccccc1", "c1ccc2[nH]ccc2c1"])
def test_openbabel_hadd_without_template(tmp_path, smiles):
  full = Chem.AddHs(Chem.MolFromSmiles(smiles))
  rdDistGeom.EmbedMolecule(full, randomSeed=9)
  source = tmp_path / "input.pdb"
  Chem.MolToPDBFile(Chem.RemoveHs(full), str(source))
  with pytest.warns(UserWarning, match="inferred"):
    mol = add_hydrogens(source, tmp_path / "output.pdb")
  assert len(mol.atoms) == full.GetNumAtoms()
  assert Chem.MolToSmiles(Chem.RemoveHs(to_rdkit(mol))) == Chem.MolToSmiles(Chem.MolFromSmiles(smiles))
  # GPL-2.0-only Open Babel runs as a separate `obabel` process and is never imported into this GPL-3.0 package.
  assert "openbabel" not in sys.modules


def test_missing_obabel_is_a_dependency_error(ethanol_pdb, tmp_path, monkeypatch):
  monkeypatch.setattr(chemistry, "obabel_command", lambda: None)
  output = tmp_path / "output.pdb"
  with pytest.raises(DependencyError, match="obabel"):
    add_hydrogens(ethanol_pdb, output)
  assert not output.exists()


def test_charge_mismatch_does_not_write_pdb(ethanol_pdb, tmp_path):
  output = tmp_path / "wrong.pdb"
  with pytest.raises(InputError, match="charge"):
    add_hydrogens(ethanol_pdb, output, smiles="CCO", charge=-1)
  assert not output.exists()


@pytest.mark.parametrize(
  "case",
  ["duplicate_names", "duplicate_serial", "altloc", "models", "residues", "metal"],
)
def test_ambiguous_pdb_rejected(ethanol_pdb, tmp_path, case):
  lines = ethanol_pdb.read_text().splitlines()
  if case == "duplicate_names":
    lines[1] = lines[1][:12] + lines[0][12:16] + lines[1][16:]
  elif case == "duplicate_serial":
    lines[1] = lines[1][:6] + lines[0][6:11] + lines[1][11:]
  elif case == "altloc":
    lines[0] = lines[0][:16] + "A" + lines[0][17:]
  elif case == "models":
    lines = ["MODEL        1", *lines, "MODEL        2"]
  elif case == "residues":
    lines[0] = lines[0][:17] + "OTH" + lines[0][20:]
  else:
    lines[0] = lines[0][:76] + "Zn"
  path = tmp_path / "bad.pdb"
  path.write_text("\n".join(lines) + "\n")
  with pytest.raises(InputError):
    read_pdb(path)


def test_leading_digit_atom_name_rejected(ethanol_pdb, tmp_path):
  lines = ethanol_pdb.read_text().splitlines()
  lines[0] = lines[0][:12] + "1C1 " + lines[0][16:]
  path = tmp_path / "digit.pdb"
  path.write_text("\n".join(lines) + "\n")
  with pytest.raises(InputError, match=r"must not start with a digit.*: 1C1$"):
    read_pdb(path)
  with pytest.raises(InputError, match="must not start with a digit"):
    load_molecule(path, 0, 1)


def test_template_file_supported(ethanol_pdb, tmp_path):
  path = tmp_path / "template.mol"
  Chem.MolToMolFile(Chem.MolFromSmiles("CCO"), str(path))
  mol = add_hydrogens(ethanol_pdb, tmp_path / "output.pdb", template=path)
  assert len(mol.atoms) == 9


def test_bad_options_rejected_without_calculation(tmp_path):
  for options in (
    RespOptions(qwt1=-1),
    RespOptions(density=float("nan")),
    RespOptions(max_iterations=0),
  ):
    with pytest.raises(InputError):
      options.validate()
  with pytest.raises(InputError):
    QuantumOptions(threads=0).validate()
  for options in (
    OptimizationOptions(opt_max_cycles=0),
    OptimizationOptions(optimizer="bogus"),
    OptimizationOptions(constraints="$freeze\nxyz 1\n$end"),
  ):
    with pytest.raises(InputError):
      optimizer_parameters(options, tmp_path)
  with pytest.raises(InputError):
    unit_surface(2)
  with pytest.raises(InputError):
    surface_points(np.zeros((1, 3)), [53])


def test_molecule_artifact_validation(ethanol_pdb):
  mol = load_molecule(ethanol_pdb, 0, 1)
  for invalid in (
    replace(mol, charge=2),
    replace(mol, atoms=()),
    replace(mol, bonds=((0, 0, 1),)),
    replace(mol, atoms=(replace(mol.atoms[0], xyz=(np.nan, 0, 0)), *mol.atoms[1:])),
  ):
    with pytest.raises(InputError):
      invalid.validate()
  with pytest.raises(InputError):
    Molecule.from_dict({})


@pytest.mark.parametrize("ivary", [[0, 2], [0, 0.5], [-1, 1], [True, 0]])
def test_invalid_resp_constraints_rejected(ivary):
  with pytest.raises(InputError):
    fit_resp(np.ones((3, 2)), np.zeros(3), [6, 1], 0, ivary)


def test_resp_constraint_identity_validated(ethanol_pdb):
  mol = load_molecule(ethanol_pdb, 0, 1)
  constraints = {
    "stage1": [0] * 9,
    "stage2": [0] * 9,
    "atoms": [a.name for a in reversed(mol.atoms)],
  }
  with pytest.raises(InputError, match="atom names/order"):
    validate_constraints(mol, constraints)
