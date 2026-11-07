import numpy as np
import pytest

from pyligprep import dispersion
from pyligprep.errors import DependencyError

WATER = "O 0 0 0; H 0 0 0.96; H 0.93 0 -0.24"


def test_dispersion_version_parsing():
  assert dispersion.dispersion_version("B3LYP-D3BJ") == "d3bj"
  assert dispersion.dispersion_version("b3lyp") is None
  assert dispersion.ensure_dispersion("B3LYP") is None


def test_d3bj_energy_and_gradient_are_applied():
  """Whatever backend this platform has, PySCF must include D3(BJ) energy and gradient."""
  from pyscf import dft, gto

  assert dispersion.ensure_dispersion("B3LYP-D3BJ") in (dispersion.NATIVE, dispersion.SIMPLE_DFTD3)
  from pyscf.dispersion import dftd3

  mol = gto.M(atom=WATER, basis="sto-3g", verbose=0)
  mf = dft.RKS(mol, xc="b3lyp-d3bj").density_fit()
  mf.kernel()
  model = dftd3.DFTD3Dispersion(mol, xc="b3lyp", version="d3bj", atm=False)
  result = model.get_dispersion(grad=True)
  assert result["energy"] < 0
  assert mf.scf_summary["dispersion"] == pytest.approx(result["energy"], abs=1e-12)

  step = 1e-5
  coords = mol.atom_coords()
  numeric = np.zeros_like(coords)
  for atom in range(mol.natm):
    for axis in range(3):
      energies = []
      for sign in (1, -1):
        shifted = coords.copy()
        shifted[atom, axis] += sign * step
        moved = mol.copy()
        moved.set_geom_(shifted, unit="Bohr")
        energies.append(dftd3.DFTD3Dispersion(moved, xc="b3lyp").get_dispersion()["energy"])
      numeric[atom, axis] = (energies[0] - energies[1]) / (2 * step)
  assert np.allclose(result["gradient"], numeric, atol=1e-8)


def test_missing_backend_is_dependency_error(monkeypatch):
  monkeypatch.setattr(dispersion, "_native_available", lambda: False)

  def unavailable():
    raise ImportError("no dftd3")

  monkeypatch.setattr(dispersion, "_simple_dftd3_module", unavailable)
  with pytest.raises(DependencyError, match="dftd3"):
    dispersion.ensure_dispersion("B3LYP-D3BJ")


def test_d4_without_native_backend_is_dependency_error(monkeypatch):
  monkeypatch.setattr(dispersion, "_native_available", lambda: False)
  with pytest.raises(DependencyError, match="D4"):
    dispersion.ensure_dispersion("B3LYP-D4")
