from decimal import Decimal

import numpy as np
import pytest

from pyligprep.charge_rounding import (
  final_equivalence_groups,
  round_charges,
  validate_export_charges,
)
from pyligprep.chemistry import load_molecule
from pyligprep.errors import CalculationError, InputError


def test_round_half_up_positive_and_negative():
  charges, report = round_charges([0.047205, -0.047205], 0, [[0], [1]])
  assert charges.tolist() == [0.04721, -0.04721]
  assert report["correction_units"] == [0, 0]


@pytest.mark.parametrize("sign", [-1, 1])
@pytest.mark.parametrize("anchor", [-1, 0, 1])
def test_correction_prioritizes_nearest_anchor(sign, anchor):
  values = np.array([0.000004, 0.300004, -0.300008]) * sign
  values[0] += anchor
  charges, report = round_charges(values, anchor, [[0], [1], [2]])
  assert report["correction_units"] == [sign, 0, 0]
  assert sum(Decimal(f"{q:.6f}") for q in charges) == anchor


def test_skip_triplet_when_single_atom_can_absorb_one_unit():
  charges, report = round_charges([0.010004] * 3 + [-0.030012], 0, [[0, 1, 2], [3]])
  assert report["correction_units"] == [0, 0, 0, 1]
  assert charges.tolist() == [0.01, 0.01, 0.01, -0.03]


def test_equivalent_oxygens_change_together():
  charges, report = round_charges(
    [-0.999996, -0.999996, 0.333334, 0.333334, 0.333324],
    -1,
    [[0, 1], [2], [3], [4]],
  )
  assert report["correction_units"] == [1, 1, 0, 0, 0]
  assert charges[0] == charges[1] == -0.99999
  assert sum(Decimal(f"{q:.6f}") for q in charges) == -1


def test_opposite_direction_group_corrections_can_make_one_unit():
  charges, report = round_charges(
    [0.100006] * 3 + [-0.150009] * 2,
    0,
    [[0, 1, 2], [3, 4]],
  )
  assert report["correction_units"] == [-1, -1, -1, 1, 1]
  assert sum(Decimal(f"{q:.6f}") for q in charges) == 0
  assert charges[0] == charges[1] == charges[2]
  assert charges[3] == charges[4]


def test_multiple_units_fallback_and_impossible_lattice():
  # Groups of 6, 9, 20 cannot correct +1 with just +/-1 per group.
  raw = [(2 - 1 / 35) * 1e-5] * 6 + [(1 - 1 / 35) * 1e-5] * 9 + [(-1 - 1 / 35) * 1e-5] * 20
  charges, report = round_charges(raw, 0, [list(range(6)), list(range(6, 15)), list(range(15, 35))])
  assert max(abs(v) for v in report["correction_units"]) >= 2
  assert sum(Decimal(f"{q:.6f}") for q in charges) == 0
  assert len(set(charges[:6])) == len(set(charges[6:15])) == len(set(charges[15:])) == 1
  with pytest.raises(CalculationError, match="equivalence"):
    round_charges([1 / 3] * 3, 1, [[0, 1, 2]])


def test_final_groups_include_frozen_first_stage_but_not_released_ties(ethanol_pdb):
  mol = load_molecule(ethanol_pdb, 0, 1)
  constraints = {
    "stage1": [0, 1, 1, 0, 4, 4, 0, 7, 0],
    "stage2": [-1, -1, 0, 0, 4, 5, 0, 0, -1],
  }
  assert final_equivalence_groups(mol, constraints) == [
    [0, 1],
    [2],
    [3, 4, 5],
    [6],
    [7],
    [8],
  ]


@pytest.mark.parametrize(
  "values,total,groups",
  [
    ([float("nan")], 0, [[0]]),
    ([0.1], 0, [[0]]),
    ([0.1, -0.1], 0, [[0, 1]]),
    ([0.0, 0.0], 0, [[0], [0]]),
    ([0.0, 0.0], 0, [[0]]),
    ([0.0], 0.1, [[0]]),
  ],
)
def test_invalid_rounding_input_rejected(values, total, groups):
  with pytest.raises(InputError):
    round_charges(values, total, groups)


def test_zero_residual_and_determinism():
  raw = [0.04, 0.04, -0.08]
  first, report = round_charges(raw, 0, [[0], [1], [2]])
  second, _ = round_charges(first, 0, [[0], [1], [2]])
  assert first.tolist() == second.tolist() == raw
  assert report["correction_units"] == [0, 0, 0]


@pytest.mark.parametrize(
  "texts,expected,total",
  [
    (["0.100001", "-0.100001"], [0.1, -0.1], 0),
    (["0.10000", "-0.10000"], [0.1, -0.1], 0),
    (["0.100000", "-0.099990"], [0.1, -0.09999], 0),
    (["0.100000", "-0.100000"], [0.2, -0.2], 0),
    (["0.000000"], [0.0, 0.0], 0),
  ],
)
def test_on_disk_charges_reject_wrong_precision_sum_or_changed_values(texts, expected, total):
  with pytest.raises(CalculationError):
    validate_export_charges(texts, expected, total)


def test_equal_numbers_do_not_imply_equivalent_atoms():
  charges, report = round_charges([0.010004, 0.010004, -0.020008], 0, [[0], [1], [2]])
  assert report["correction_units"] == [1, 0, 0]
  assert charges[0] != charges[1]


def test_many_group_sizes_preserve_charge_and_symmetry():
  rng = np.random.default_rng(47)
  for _ in range(100):
    sizes = rng.integers(1, 12, size=10)
    groups, raw = [], []
    values = rng.uniform(-0.5, 0.5, len(sizes))
    values[-1] = -np.dot(sizes[:-1], values[:-1]) / sizes[-1]
    for size, value in zip(sizes, values, strict=True):
      groups.append(list(range(len(raw), len(raw) + size)))
      raw.extend([value] * size)
    actual, _ = round_charges(raw, 0, groups)
    assert sum(Decimal(f"{q:.6f}") for q in actual) == 0
    assert all(len(set(actual[g])) == 1 for g in groups)
