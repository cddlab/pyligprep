import numpy as np
import pytest

from pyligprep.errors import CalculationError
from pyligprep.resp import fit_resp


def test_fit_recovers_known_charges_and_enforces_constraints():
  rng = np.random.default_rng(42)
  design = rng.uniform(0.1, 1.0, (200, 4))
  expected = np.array([-0.6, 0.2, 0.2, 0.2])
  result = fit_resp(design, design @ expected, [6, 1, 1, 1], 0, [0, 0, 2, 2], qwt=0)
  np.testing.assert_allclose(result.charges, expected, atol=1e-11)
  assert abs(sum(result.charges)) < 1e-12


def test_frozen_charges_survive_second_stage():
  rng = np.random.default_rng(8)
  design = rng.uniform(0.1, 1, (100, 3))
  initial = np.array([-0.8, 0.4, 0.4])
  result = fit_resp(design, design @ initial, [8, 1, 1], 0, [-1, 0, 2], initial=initial)
  assert result.charges[0] == initial[0]
  np.testing.assert_allclose(result.charges, initial, atol=1e-12)


def test_failed_fit_is_not_success():
  rng = np.random.default_rng(9)
  design = rng.uniform(0.1, 1, (100, 3))
  with pytest.raises(CalculationError, match="converg"):
    fit_resp(design, rng.normal(size=100), [6, 1, 1], 0, [0, 0, 0], max_iterations=1)
