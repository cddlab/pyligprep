"""Validated restricted/unrestricted checkpoint conversion, adapted from ligcl3_opt.py."""

from pathlib import Path

import numpy as np
from pyscf import scf


def validate_scf_type(mol, scf_type: str) -> str:
  """Require an explicit supported spin treatment before GPU work."""
  scf_type = scf_type.lower()
  if scf_type not in ("rks", "uks"):
    raise ValueError(f"Unknown SCF type: {scf_type!r}")
  if scf_type == "rks" and mol.spin != 0:
    raise ValueError("RKS requires a closed-shell singlet (spin=0); use --scf-type uks")
  return scf_type


def validate_checkpoint_molecule(mol, saved) -> None:
  """Keep the existing basis/order/spin checks for both checkpoint formats."""
  if (
    [mol.atom_symbol(i) for i in range(mol.natm)] != [saved.atom_symbol(i) for i in range(saved.natm)]
    or mol.charge != saved.charge
    or mol.spin != saved.spin
    or mol.cart != saved.cart
    or mol._basis != saved._basis
    or mol._ecp != saved._ecp
  ):
    raise ValueError("Checkpoint atoms/order, charge, spin, basis or ECP differ from input.")


def checkpoint_orbitals(mol, data: dict) -> tuple[str, np.ndarray, np.ndarray]:
  """Detect restricted/unrestricted arrays and validate their occupations."""
  try:
    coeff = np.asarray(data["mo_coeff"])
    occ = np.asarray(data["mo_occ"])
    finite = np.isfinite(coeff).all() and np.isfinite(occ).all()
  except (KeyError, TypeError, ValueError) as exc:
    raise ValueError("Checkpoint has missing or invalid SCF orbitals.") from exc
  if not finite or np.iscomplexobj(occ):
    raise ValueError("Checkpoint has nonfinite orbitals or invalid occupations.")
  nao = mol.nao_nr()
  if coeff.ndim == 2 and coeff.shape[0] == nao and occ.shape == (coeff.shape[1],) and mol.spin == 0:
    source, max_occ, expected = "rks", 2, mol.nelectron
  elif coeff.ndim == 3 and coeff.shape[:2] == (2, nao) and occ.shape == (2, coeff.shape[2]):
    source, max_occ, expected = "uks", 1, mol.nelec
  else:
    raise ValueError("Checkpoint does not contain compatible RKS/UKS orbital shapes.")
  if (
    not 0 < coeff.shape[-1] <= nao
    or np.any(occ < 0)
    or np.any(occ > max_occ)
    or not np.allclose(occ.sum(axis=-1), expected, rtol=0, atol=1e-6)
  ):
    raise ValueError("Checkpoint occupations disagree with the electron/spin counts.")
  return source, coeff, occ


def checkpoint_density(mol, data: dict) -> tuple[str, np.ndarray]:
  """Build a CPU density in the saved AO basis and verify electron counts."""
  source, coeff, occ = checkpoint_orbitals(mol, data)
  make_rdm1 = scf.hf.make_rdm1 if source == "rks" else scf.uhf.make_rdm1
  with np.errstate(over="ignore", invalid="ignore"):
    dm = np.asarray(make_rdm1(coeff, occ))
  if not np.isfinite(dm).all():
    raise ValueError("Checkpoint produces a nonfinite density matrix.")
  overlap = mol.intor_symmetric("int1e_ovlp")
  counts = np.einsum("...ij,ji->...", dm, overlap)
  expected = mol.nelectron if source == "rks" else mol.nelec
  if not np.allclose(counts, expected, rtol=0, atol=1e-5):
    raise ValueError("Checkpoint density electron counts disagree with the molecule.")
  return source, dm


def load_restart(mol, checkpoint: Path, scf_type: str = "rks") -> np.ndarray:
  """Restore either checkpoint format as an initial density for the chosen SCF."""
  scf_type = validate_scf_type(mol, scf_type)
  if not checkpoint.is_file():
    raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
  try:
    saved, data = scf.chkfile.load_scf(str(checkpoint))
  except (OSError, KeyError, TypeError, ValueError) as exc:
    raise ValueError(f"Cannot read SCF checkpoint: {checkpoint}") from exc
  validate_checkpoint_molecule(mol, saved)
  coords = saved.atom_coords()
  if not np.isfinite(coords).all():
    raise ValueError("Checkpoint contains invalid coordinates.")
  source, dm0 = checkpoint_density(saved, data)
  if source == "uks" and scf_type == "rks":
    dm0 = dm0.sum(axis=0)
    print("UKS -> RKS: summing alpha/beta densities; spin polarization is discarded.")
  elif source == "rks" and scf_type == "uks":
    dm0 = np.stack((dm0 * 0.5, dm0 * 0.5))
    print("RKS -> UKS: using equal alpha/beta initial densities.")
  # Commit geometry only after all checkpoint checks and conversion succeed.
  mol.set_geom_(coords, unit="Bohr")
  print(f"Restart {source.upper()} -> {scf_type.upper()}: geometry and density from {checkpoint}")
  print("SCF will be reconverged; geometry optimizer history is not restored.")
  return dm0
