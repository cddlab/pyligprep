"""Two-stage RESP fitting and automatic equivalence constraints.

Adapted and modified for pyligprep from AmberTools 26 PyRESP (py_resp.py) and PyRESP_GEN (pyresp_gen.py),
GPL-3.0-only (see AmberTools' LICENSE).
"""

from collections import Counter
from dataclasses import dataclass

import numpy as np
from scipy.linalg import LinAlgError, solve

from pyligprep.chemistry import to_rdkit
from pyligprep.errors import CalculationError, InputError
from pyligprep.model import Molecule


@dataclass(frozen=True)
class FitResult:
  charges: np.ndarray
  iterations: int
  rms_esp_error: float


def stage_constraints(mol: Molecule, depth: int = 3) -> dict:
  """PyRESP_GEN depth-based equivalence with local chemistry conflict checks."""
  if depth < 1 or depth > 5:
    raise InputError("Equivalence depth must be between 1 and 5")
  neighbors = {i: [] for i in range(len(mol.atoms))}
  for i, j, _ in mol.bonds:
    neighbors[i].append(j)
    neighbors[j].append(i)
  expanded = neighbors
  for _ in range(1, depth):
    expanded = {i: [k for j in row for k in expanded[j] if k != i] for i, row in expanded.items()}
  rd_mol = to_rdkit(mol)
  # PyPE_RESP checks hybridization conflicts after depth equivalence. Derive
  # elements from atomic numbers, never from the first letter of atom names.
  signatures = []
  carbon_h = set()
  hydrogens = set()
  for i, atom in enumerate(rd_mol.GetAtoms()):
    counts = Counter(mol.atoms[j].element for j in neighbors[i])
    if atom.GetAtomicNum() == 6 and len(neighbors[i]) == 4 and counts["H"] >= 2:
      carbon_h.add(i)
      hydrogens.update(j for j in neighbors[i] if mol.atoms[j].element == "H")
    signatures.append(
      (
        atom.GetAtomicNum(),
        str(atom.GetHybridization()),
        tuple(sorted(counts.items())),
        tuple(sorted(Counter(mol.atoms[j].element for j in expanded[i]).items())),
      )
    )
  groups = {}
  for i, sig in enumerate(signatures):
    groups.setdefault(sig, []).append(i)
  first = [0] * len(mol.atoms)
  second = [-1] * len(mol.atoms)
  for i in carbon_h | hydrogens:
    second[i] = 0
  for group in groups.values():
    for members, target in (
      ([i for i in group if i not in hydrogens], first),
      ([i for i in group if i in carbon_h | hydrogens], second),
    ):
      for i in members[1:]:
        target[i] = members[0] + 1
  return {
    "stage1": first,
    "stage2": second,
    "depth": depth,
    "policy": "pyresp-gen-depth-with-element-hybridization-neighbor-check",
    "indexing": "ivary: 0=free, -1=fixed, positive=1-based equal atom",
    "atoms": [a.name for a in mol.atoms],
  }


def _parameter_map(ivary, initial):
  n = len(ivary)
  roots = []
  for i, value in enumerate(ivary):
    if not isinstance(value, (int, np.integer)) or isinstance(value, bool):
      raise InputError("RESP ivary entries must be integers")
    if value == -1:
      roots.append(-1)
    elif value == 0:
      roots.append(i)
    elif 0 < value <= i and roots[value - 1] >= 0:
      roots.append(roots[value - 1])
    else:
      raise InputError("Equivalent atoms must reference an earlier unfrozen atom")
  active = sorted(set(roots) - {-1})
  transform = np.zeros((n, len(active)))
  columns = {root: i for i, root in enumerate(active)}
  fixed = np.zeros(n)
  for i, root in enumerate(roots):
    if root == -1:
      fixed[i] = initial[i]
    else:
      transform[i, columns[root]] = 1
  return transform, fixed


def validate_constraints(molecule: Molecule, constraints: dict) -> None:
  names = [a.name for a in molecule.atoms]
  if constraints.get("atoms", names) != names:
    raise InputError("RESP constraint atom names/order differ from the molecule")
  for stage in ("stage1", "stage2"):
    if stage not in constraints or len(constraints[stage]) != len(names):
      raise InputError("RESP constraint length differs from atom count")
    _parameter_map(constraints[stage], np.zeros(len(names)))


def fit_resp(
  design,
  potential,
  numbers,
  charge,
  ivary,
  initial=None,
  qwt=0.0005,
  max_iterations=42,
  tolerance=1e-6,
) -> FitResult:
  """Constrained least squares with PyRESP's harmonic start and hyperbola."""
  design = np.asarray(design, dtype=float)
  potential = np.asarray(potential, dtype=float)
  n = len(numbers)
  initial = np.zeros(n) if initial is None else np.asarray(initial, dtype=float)
  if (
    design.ndim != 2
    or design.shape[1] != n
    or potential.shape != (len(design),)
    or len(ivary) != n
    or initial.shape != (n,)
    or not len(design)
    or not np.isfinite([qwt, tolerance]).all()
    or qwt < 0
    or tolerance <= 0
    or max_iterations < 1
    or not all(np.isfinite(x).all() for x in (design, potential, initial))
  ):
    raise InputError("Invalid RESP data, controls or initial charges")
  transform, fixed = _parameter_map(ivary, initial)
  if transform.shape[1] == 0:
    if not np.isclose(fixed.sum(), charge, atol=1e-9, rtol=0):
      raise InputError("Frozen charges violate total charge")
    return FitResult(fixed, 0, float(np.sqrt(np.mean((design @ fixed - potential) ** 2))))
  normal = design.T @ design
  rhs = design.T @ potential
  charges = initial.copy()
  previous = np.zeros(n)
  weights = np.where(np.asarray(numbers) == 1, 0.0, qwt)
  total = transform.sum(axis=0)
  for iteration in range(max_iterations):
    diagonal = weights if iteration == 0 else weights / np.sqrt(charges**2 + 0.01)
    matrix = normal + np.diag(diagonal)
    vector = rhs + (weights * initial if iteration == 0 else 0)
    reduced = transform.T @ matrix @ transform
    kkt = np.zeros((len(total) + 1, len(total) + 1))
    kkt[:-1, :-1] = reduced
    kkt[:-1, -1] = kkt[-1, :-1] = total
    target = np.r_[transform.T @ (vector - matrix @ fixed), charge - fixed.sum()]
    try:
      solution = solve(kkt, target, assume_a="sym")[:-1]
    except LinAlgError as exc:
      raise CalculationError("RESP constraint matrix is singular") from exc
    charges = transform @ solution + fixed
    change = np.linalg.norm(charges - previous) / n
    if not np.isfinite(charges).all():
      raise CalculationError("RESP produced non-finite charges")
    if not np.any(weights) or (iteration > 1 and change < tolerance):
      if abs(charges.sum() - charge) > 1e-8:
        raise CalculationError("RESP total-charge constraint was not satisfied")
      rms = float(np.sqrt(np.mean((design @ charges - potential) ** 2)))
      return FitResult(charges, iteration, rms)
    previous = charges.copy()
  raise CalculationError(f"RESP did not converge in {max_iterations} iterations")


def write_resp_input(path, mol, ivary, qwt, stage):
  """Human-readable reference input; not executed by the production pipeline."""
  with path.open("w", encoding="utf-8") as handle:
    handle.write(
      f"pyligprep stage {stage}\n &cntrl\n nmol=1, ihfree=1, iqopt={stage},\n qwt={qwt}, ipol=0, ipermdip=0,\n &end\n 1.0\npyligprep\n"
    )
    handle.write(f"{mol.charge:5d}{len(mol.atoms):5d}\n")
    for z, vary in zip(mol.numbers, ivary, strict=True):
      handle.write(f"{z:5d}{vary:5d}\n")
    handle.write("\n")
