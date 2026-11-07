"""DFT-D3 backend selection for PySCF.

PySCF evaluates ``-D3BJ`` functionals through ``pyscf.dispersion`` (the pyscf-dispersion
package), which publishes Linux wheels only. On macOS the same simple-dftd3 library is
available as the ``dftd3`` package; this module exposes it under the interface PySCF calls
(``DFTD3Dispersion(mol, xc, version, atm).get_dispersion(grad)``). Both wrap the s-dftd3
parameter tables, so energies and gradients are the same model.
"""

import sys
import types

from pyligprep.errors import DependencyError

NATIVE = "pyscf-dispersion"
SIMPLE_DFTD3 = "simple-dftd3"


def dispersion_version(method: str) -> str | None:
  """Return the dispersion version PySCF parses from an XC name, e.g. ``d3bj``."""
  from pyscf.scf.dispersion import parse_disp

  return parse_disp(method.lower())[1]


def _native_available() -> bool:
  try:
    import pyscf.dispersion  # noqa: F401
  except ImportError:
    return False
  # Our own shim registers a module without __file__; only count the real package.
  return getattr(sys.modules["pyscf.dispersion"], "__file__", None) is not None


def _simple_dftd3_module():
  from dftd3 import pyscf as sdftd3

  class DFTD3Dispersion(sdftd3.DFTD3Dispersion):
    """pyscf-dispersion compatible facade over simple-dftd3's PySCF interface."""

    def __init__(self, mol, xc="hf", version="d3bj", atm=False):
      super().__init__(mol, xc=xc, version=version.lower(), atm=atm)

    def get_dispersion(self, grad=False):
      energy, gradient, virial = self.kernel()
      result = {"energy": float(energy)}
      if grad:
        result.update(gradient=gradient, virial=virial)
      return result

  module = types.ModuleType("pyscf.dispersion.dftd3")
  module.__dict__["DFTD3Dispersion"] = DFTD3Dispersion
  return module


def dispersion_backend() -> str | None:
  """Report the D3 backend that is (or would be) used, without modifying PySCF."""
  if _native_available():
    return NATIVE
  try:
    import dftd3.pyscf  # noqa: F401
  except ImportError:
    return None
  return SIMPLE_DFTD3


def ensure_dispersion(method: str) -> str | None:
  """Make PySCF able to evaluate the dispersion term requested by ``method``.

  Returns the backend name, or None when ``method`` has no dispersion correction.
  """
  version = dispersion_version(method)
  if version is None:
    return None
  if _native_available():
    return NATIVE
  if not version.startswith("d3"):
    raise DependencyError(f"{method} requires pyscf-dispersion (DFT-D4 is not provided on this platform)")
  try:
    dftd3 = _simple_dftd3_module()
  except ImportError as exc:
    raise DependencyError(f"{method} requires pyscf-dispersion (Linux) or dftd3 (macOS)") from exc
  package = types.ModuleType("pyscf.dispersion")
  package.__dict__.update(dftd3=dftd3, dftd4=None)
  sys.modules["pyscf.dispersion"] = package
  sys.modules["pyscf.dispersion.dftd3"] = dftd3
  # pyscf.scf.dispersion bound these names when it was first imported.
  import pyscf.scf.dispersion as scf_dispersion

  scf_dispersion.dispersion = package
  scf_dispersion.dftd3 = dftd3
  return SIMPLE_DFTD3
