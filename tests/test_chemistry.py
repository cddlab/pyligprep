import numpy as np
import pytest
from rdkit import Chem

from pyligprep.chemistry import add_hydrogens, load_molecule
from pyligprep.errors import InputError


def test_hadd_preserves_existing_atoms_and_is_idempotent(ethanol_pdb, tmp_path):
  mol = Chem.MolFromPDBFile(str(ethanol_pdb), removeHs=False)
  heavy = Chem.RemoveHs(mol)
  source = tmp_path / "heavy.pdb"
  Chem.MolToPDBFile(heavy, str(source))
  output = tmp_path / "added.pdb"
  first = add_hydrogens(source, output, smiles="CCO", charge=0)
  assert len(first.atoms) == 9
  assert len(set(a.name for a in first.atoms)) == 9
  np.testing.assert_allclose(first.coords[:3], heavy.GetConformer().GetPositions(), atol=1e-12)
  second = add_hydrogens(output, tmp_path / "again.pdb", charge=0)
  assert len(second.atoms) == 9
  assert [a.name for a in first.atoms] == [a.name for a in second.atoms]
  loaded = load_molecule(output, charge=0, multiplicity=1)
  assert loaded.charge == 0


def test_stale_sidecar_rejected(ethanol_pdb, tmp_path):
  output = tmp_path / "added.pdb"
  add_hydrogens(ethanol_pdb, output, smiles="CCO")
  output.write_text(output.read_text() + "REMARK edited\n")
  with pytest.raises(InputError, match="sidecar"):
    load_molecule(output, charge=0, multiplicity=1)


def test_charge_and_spin_are_validated(ethanol_pdb):
  with pytest.raises(InputError):
    load_molecule(ethanol_pdb, charge=0, multiplicity=2)


def test_hadd_will_not_overwrite_input(ethanol_pdb):
  original = ethanol_pdb.read_bytes()
  with pytest.raises((FileExistsError, InputError)):
    add_hydrogens(ethanol_pdb, ethanol_pdb, smiles="CCO")
  assert ethanol_pdb.read_bytes() == original
