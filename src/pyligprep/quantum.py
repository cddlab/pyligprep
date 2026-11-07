"""Lazy CPU/GPU selection and electronic structure factories."""

import warnings
from dataclasses import dataclass
from pathlib import Path

from pyligprep.errors import CalculationError, DependencyError, InputError
from pyligprep.model import Molecule


@dataclass(frozen=True)
class QuantumOptions:
  device: str = "auto"
  threads: int = 8
  memory_mb: int = 22000
  max_scf_cycles: int = 512

  def validate(self):
    if self.device not in ("auto", "gpu", "cpu") or min(self.threads, self.memory_mb, self.max_scf_cycles) < 1:
      raise InputError("Invalid device, threads, memory or SCF cycle limit")


def gpu_probe() -> tuple[bool, str]:
  try:
    import cupy
    import gpu4pyscf

    if cupy.cuda.runtime.getDeviceCount() < 1:
      return False, "No visible CUDA device"
    cupy.zeros(1).sum().item()
    return (
      True,
      f"GPU4PySCF {gpu4pyscf.__version__}: {cupy.cuda.runtime.getDeviceProperties(0)['name']}",
    )
  except (ImportError, OSError, RuntimeError) as exc:
    return False, f"{type(exc).__name__}: {exc}"


def choose_device(requested: str) -> str:
  if requested == "cpu":
    return "cpu"
  if requested not in ("auto", "gpu"):
    raise InputError(f"Unknown device: {requested}")
  available, detail = gpu_probe()
  if available:
    print(detail)
    return "gpu"
  if requested == "gpu":
    raise DependencyError(f"GPU was requested but is unavailable: {detail}")
  warnings.warn(f"Using CPU: {detail}", stacklevel=2)
  return "cpu"


def build_molecule(
  molecule: Molecule,
  basis: str,
  options: QuantumOptions,
  log: Path,
  cart: bool = False,
):
  from pyscf import gto, lib

  options.validate()
  molecule.validate()
  lib.num_threads(options.threads)
  return gto.M(
    atom=[(a.element, a.xyz) for a in molecule.atoms],
    basis=basis,
    unit="Angstrom",
    charge=molecule.charge,
    spin=molecule.multiplicity - 1,
    cart=cart,
    symmetry=False,
    max_memory=options.memory_mb,
    output=str(log),
    verbose=4,
  )


def make_method(
  mol,
  method: str,
  device: str,
  checkpoint: Path,
  options: QuantumOptions,
  scf_type="auto",
):
  from pyscf import dft, scf

  unrestricted = scf_type == "uks" or (scf_type == "auto" and mol.spin != 0)
  if scf_type not in ("auto", "rks", "uks") or (scf_type == "rks" and mol.spin):
    raise InputError("Restricted SCF requires a closed-shell singlet")
  if method.upper() == "HF":
    mf = scf.UHF(mol) if unrestricted else scf.RHF(mol)
  else:
    from pyligprep.dispersion import ensure_dispersion

    # -D3BJ needs a dispersion backend; on macOS this installs the simple-dftd3 shim.
    ensure_dispersion(method)
    factory = dft.UKS if unrestricted else dft.RKS
    mf = factory(mol, xc=method.lower()).density_fit()
    mf.grids.level = 3
  if device == "gpu":
    mf = mf.to_gpu()
  mf.conv_tol = 1e-10
  mf.conv_tol_grad = 1e-6
  mf.max_cycle = options.max_scf_cycles
  mf.chkfile = str(checkpoint)
  if method.upper() != "HF":
    mf = mf.newton()
    mf.max_cycle = min(100, options.max_scf_cycles)
    mf.max_stepsize = 0.03
  return mf


def run_scf(mf, dm0=None, device="cpu"):
  if dm0 is not None and device == "gpu":
    import cupy

    dm0 = cupy.asarray(dm0)
  mf.kernel(dm0=dm0)
  if not mf.converged:
    raise CalculationError("SCF did not converge; checkpoint retained, no numerical CPU fallback")
  return mf
