"""Five-decimal charge export with exact totals and RESP equivalence groups."""

from decimal import ROUND_HALF_UP, Decimal
from functools import reduce
from heapq import heappop, heappush
from math import gcd

import numpy as np

from pyligprep.errors import CalculationError, InputError
from pyligprep.model import Molecule
from pyligprep.resp import stage_constraints, validate_constraints

SCALE = 100_000
POLICY = "five-decimal-half-up-equivalent-v1"


def final_equivalence_groups(mol: Molecule, constraints: dict | None) -> list[list[int]]:
  """Retain stage-2 ties and stage-1 ties among atoms frozen in stage 2."""
  constraints = stage_constraints(mol) if constraints is None else constraints
  validate_constraints(mol, constraints)
  roots = []
  for name in ("stage1", "stage2"):
    current = []
    for i, value in enumerate(constraints[name]):
      current.append(current[value - 1] if value > 0 else i)
    roots.append(current)
  groups = {}
  for i, value in enumerate(constraints["stage2"]):
    key = ("frozen", roots[0][i]) if value == -1 else ("active", roots[1][i])
    groups.setdefault(key, []).append(i)
  return list(groups.values())


def _rounded_groups(charges, total, groups):
  charges = np.asarray(charges, dtype=float)
  if (
    charges.ndim != 1
    or not len(charges)
    or not np.isfinite(charges).all()
    or isinstance(total, bool)
    or not isinstance(total, (int, np.integer))
    or abs(charges.sum() - total) > 1e-8
  ):
    raise InputError("Invalid RESP charges or integer total for rounding")
  indices = [i for group in groups for i in group]
  if (
    not groups
    or any(not group for group in groups)
    or any(isinstance(i, bool) or not isinstance(i, (int, np.integer)) for i in indices)
    or sorted(indices) != list(range(len(charges)))
  ):
    raise InputError("Equivalence groups must partition all atoms exactly once")
  rounded, distances = [], []
  for group in groups:
    if np.ptp(charges[group]) > 1e-8:
      raise InputError("RESP charges disagree within an equivalence group")
    mean = sum((Decimal(str(charges[i])) for i in group), Decimal(0)) / len(group)
    rounded.append(int((mean * SCALE).to_integral_value(rounding=ROUND_HALF_UP)))
    distances.append(min(abs(mean - anchor) for anchor in (-1, 0, 1)))
  return charges, rounded, distances


def _single_step_subset(sizes: list[int], target: int) -> list[int] | None:
  """Prefer earlier groups, skipping any that prevent an exact remainder."""
  sign = 1 if target >= 0 else -1
  remaining = abs(target)
  mask = (1 << (remaining + 1)) - 1
  suffix = [1] * (len(sizes) + 1)
  for i in range(len(sizes) - 1, -1, -1):
    suffix[i] = (suffix[i + 1] | (suffix[i + 1] << sizes[i])) & mask
  if not (suffix[0] >> remaining) & 1:
    return None
  steps = [0] * len(sizes)
  for i, size in enumerate(sizes):
    if size <= remaining and (suffix[i + 1] >> (remaining - size)) & 1:
      steps[i] = sign
      remaining -= size
  return steps


def _balanced_steps(sizes: list[int], target: int) -> list[int]:
  """Find signed whole-group corrections with minimum total atom steps.

  A finite shortest-path search permits opposite signs and repeated steps.
  Among equal-cost paths, prefer groups nearer 0/-1/+1 (input order here).
  Any signed sum can be reordered to stay within target +/- max group size.
  """
  common = reduce(gcd, sizes)
  if target % common:
    raise CalculationError(
      "Cannot conserve integer total charge at five decimal places while "
      f"preserving equivalence groups (group-size gcd={common})"
    )
  choices = {}
  for i, size in enumerate(sizes):
    choices.setdefault(size, i)
  limit = max(sizes)
  lower, upper = min(0, target) - limit, max(0, target) + limit
  costs, previous, queue = {0: (0, 0)}, {}, [(0, 0, 0)]
  while queue:
    cost, priority, value = heappop(queue)
    if (cost, priority) != costs[value]:
      continue
    if value == target:
      break
    for size, i in choices.items():
      for sign in (-1, 1):
        nxt = value + sign * size
        candidate = (cost + size, priority + (i + 1) * size)
        if lower <= nxt <= upper and (nxt not in costs or candidate < costs[nxt]):
          costs[nxt] = candidate
          previous[nxt] = (value, i, sign)
          heappush(queue, (*candidate, nxt))
  if target not in costs:
    raise CalculationError("No integer charge correction preserving equivalence")
  steps = [0] * len(sizes)
  value = target
  while value:
    value, i, sign = previous[value]
    steps[i] += sign
  return steps


def round_charges(charges, total: int, groups: list[list[int]]) -> tuple[np.ndarray, dict]:
  """Round half away from zero, then conserve charge on the 1e-5 lattice."""
  raw, rounded, distances = _rounded_groups(charges, total, groups)
  order = sorted(range(len(groups)), key=lambda i: (distances[i], groups[i][0]))
  sizes = [len(groups[i]) for i in order]
  residual = int(total) * SCALE - sum(q * len(g) for q, g in zip(rounded, groups, strict=True))
  steps = _single_step_subset(sizes, residual)
  if steps is None:
    steps = _balanced_steps(sizes, residual)
  before, after = [0] * len(raw), [0] * len(raw)
  for index, step in zip(order, steps, strict=True):
    for atom in groups[index]:
      before[atom] = rounded[index]
      after[atom] = rounded[index] + step
  if sum(after) != int(total) * SCALE:
    raise CalculationError("Internal error: rounded charge sum differs from target")
  result = np.array(after, dtype=float) / SCALE
  return result, {
    "policy": POLICY,
    "decimal_places": 5,
    "output_decimal_places": 6,
    "unit_e": "0.000010",
    "total_charge": int(total),
    "resp_charges": raw.tolist(),
    "groups_zero_based": groups,
    "rounded_units": before,
    "corrected_units": after,
    "residual_before_correction_units": residual,
    "correction_units": [a - b for a, b in zip(after, before, strict=True)],
    "maximum_change_from_resp_e": float(np.max(np.abs(result - raw))),
  }


def validate_export_charges(texts: list[str], expected, total: int) -> None:
  """Check the actual on-disk decimal numbers, not a float sum tolerance."""
  import re

  if len(texts) != len(expected) or any(re.fullmatch(r"[+-]?\d+\.\d{5}0", value) is None for value in texts):
    raise CalculationError("Export charges must have six decimals with final digit zero")
  values = [Decimal(value) for value in texts]
  if sum(values) != total or values != [Decimal(f"{q:.6f}") for q in expected]:
    raise CalculationError("Export changed corrected charges or integer total charge")
