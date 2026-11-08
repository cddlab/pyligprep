"""Instance-local AO overlap truncation for geometry optimization."""

import math
from numbers import Real
from typing import Any

import numpy as np

from pyligprep.errors import CalculationError, DependencyError, InputError


def validate_overlap_threshold(value: float | None) -> None:
  if value is not None and (
    isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) or value <= 0
  ):
    raise InputError("opt-overlap-threshold must be a finite positive number")


class _OverlapThreshold:
  """Mixin retained by PySCF's Newton and scanner class composition."""

  opt_overlap_threshold: float
  mol: Any

  def check_linear_dependency(self, overlap, verbose=None):
    from pyscf.lib import logger

    if isinstance(overlap, np.ndarray):
      from scipy.linalg import eigh

      xp = np
    else:
      import cupy as xp
      from gpu4pyscf.lib.cupy_helper import eigh

    if not bool(xp.isfinite(overlap).all()):
      raise CalculationError("Nonfinite AO overlap matrix")
    eigenvalues, vectors = eigh(overlap)
    if not bool(xp.isfinite(eigenvalues).all()):
      raise CalculationError("Nonfinite AO overlap eigenvalues")
    keep = eigenvalues > self.opt_overlap_threshold
    retained = int(keep.sum())
    if retained == 0 or retained < max(self.mol.nelec):
      raise CalculationError("AO overlap threshold leaves too few orbitals for the electrons")
    log = logger.new_logger(self, verbose)
    log.info(
      "AO overlap threshold = %.6g; min eigenvalue = %.6g; retained = %d/%d",
      self.opt_overlap_threshold,
      float(eigenvalues[0]),
      retained,
      len(eigenvalues),
    )
    smallest = float(xp.abs(eigenvalues).min())
    condition = float(xp.abs(eigenvalues).max()) / smallest if smallest else math.inf
    if condition > 1e10:
      log.warn("Ill-conditioned AO overlap matrix (cond = %.6g); check SCF and gradient accuracy", condition)
    if retained < len(eigenvalues):
      log.warn(
        "%d small eigenvectors of overlap matrix removed because of linear dependency between AOs.",
        len(eigenvalues) - retained,
      )
    return vectors[:, keep] / xp.sqrt(eigenvalues[keep])


def apply_overlap_threshold(mf, threshold: float | None):
  """Attach after device conversion and before Newton wrapping; leave defaults untouched."""
  validate_overlap_threshold(threshold)
  if threshold is None:
    return mf
  if not callable(getattr(mf, "check_linear_dependency", None)):
    raise DependencyError("opt-overlap-threshold requires a PySCF/GPU4PySCF version with check_linear_dependency")
  from pyscf import lib

  mf.opt_overlap_threshold = float(threshold)
  mf._keys = mf._keys | {"opt_overlap_threshold"}
  return lib.set_class(mf, (_OverlapThreshold, mf.__class__))
