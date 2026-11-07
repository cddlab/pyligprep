import numpy as np
import pytest
from pyscf import gto, lib, scf

from pyligprep.checkpoint import checkpoint_density, load_restart


@pytest.fixture
def checkpoints(tmp_path):
  lib.num_threads(1)
  mol = gto.M(atom="O 0 0 0; H 0 -0.75 0.58; H 0 0.75 0.58", basis="sto-3g", verbose=0)
  methods = {}
  for kind, factory in (("rks", scf.RHF), ("uks", scf.UHF)):
    mf = factory(mol)
    mf.chkfile = str(tmp_path / f"{kind}.chk")
    mf.kernel()
    assert mf.converged
    methods[kind] = mf
  return mol, methods


@pytest.mark.parametrize("source,target", [(s, t) for s in ("rks", "uks") for t in ("rks", "uks")])
def test_restricted_unrestricted_restart(checkpoints, source, target):
  from pathlib import Path

  mol, methods = checkpoints
  shifted = mol.copy().set_geom_(mol.atom_coords() + 0.2)
  dm = load_restart(shifted, Path(methods[source].chkfile), target)
  np.testing.assert_allclose(shifted.atom_coords(), mol.atom_coords())
  total = methods[source].make_rdm1()
  if total.ndim == 3:
    total = total.sum(axis=0)
  expected = total if target == "rks" else np.stack((total / 2, total / 2))
  np.testing.assert_allclose(dm, expected, atol=1e-6)


def test_corrupt_checkpoint_rejected(checkpoints):
  mol, methods = checkpoints
  _, data = scf.chkfile.load_scf(methods["rks"].chkfile)
  for updates in (
    {"mo_occ": data["mo_occ"] * 0.5},
    {"mo_coeff": data["mo_coeff"] * 2},
    {"mo_coeff": np.full_like(data["mo_coeff"], np.nan)},
    {"mo_occ": [2]},
  ):
    with pytest.raises(ValueError):
      checkpoint_density(mol, {**data, **updates})
