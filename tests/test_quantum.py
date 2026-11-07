from unittest.mock import patch

import numpy as np
import pytest

from pyligprep.errors import DependencyError
from pyligprep.esp import BOHR, evaluate_esp, surface_points
from pyligprep.quantum import choose_device


def test_device_selection_and_forced_gpu_failure():
  with patch("pyligprep.quantum.gpu_probe", return_value=(False, "test unavailable")):
    with pytest.warns(UserWarning, match="CPU"):
      assert choose_device("auto") == "cpu"
    assert choose_device("cpu") == "cpu"
    with pytest.raises(DependencyError):
      choose_device("gpu")


def test_cpu_esp_matches_pointwise_integrals():
  # Single-point HF only: deliberately no CPU geometry optimization.
  from pyscf import gto, lib, scf

  lib.num_threads(1)
  mol = gto.M(
    atom="O 0 0 0; H 0 -0.757 0.587; H 0 0.757 0.587",
    basis="6-31g*",
    cart=True,
    verbose=0,
  )
  mf = scf.RHF(mol).run(conv_tol=1e-11)
  points = surface_points(mol.atom_coords(), [8, 1, 1])[::17]
  actual = evaluate_esp(mf, points, "cpu", batch_size=7)
  reference = []
  for point in points:
    with mol.with_rinv_origin(point):
      electron = np.einsum("ij,ji", mol.intor("int1e_rinv"), mf.make_rdm1())
    nuclear = sum(mol.atom_charges() / np.linalg.norm(mol.atom_coords() - point, axis=1))
    reference.append(nuclear - electron)
  np.testing.assert_allclose(actual, reference, atol=1e-10)
  assert BOHR == lib.param.BOHR


@pytest.mark.gpu
def test_cpu_gpu_hf_esp_agree():
  from pyscf import gto, lib, scf

  lib.num_threads(2)
  mol = gto.M(
    atom="O 0 0 0; H 0 -0.757 0.587; H 0 0.757 0.587",
    basis="6-31g*",
    cart=True,
    verbose=0,
  )
  cpu = scf.RHF(mol).run(conv_tol=1e-11)
  gpu = scf.RHF(mol).to_gpu().run(conv_tol=1e-11)
  assert gpu.converged
  points = surface_points(mol.atom_coords(), [8, 1, 1])
  np.testing.assert_allclose(evaluate_esp(cpu, points, "cpu"), evaluate_esp(gpu, points, "gpu"), atol=2e-7)
