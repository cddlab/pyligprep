"""Shared CPU/GPU ESP grids and evaluation in atomic units.

Surface algorithm and GAMESS radii adapted from GPU4PySCF 1.9.1 pop/esp.py,
Copyright 2021-2024 The PySCF Developers, Apache-2.0. See THIRD_PARTY.md.
"""

import numpy as np
from scipy.spatial.distance import cdist

from pyligprep.errors import InputError

BOHR = 0.52917721092
RADII = {1: 1.20, 6: 1.50, 7: 1.50, 8: 1.40, 9: 1.35, 15: 1.80, 16: 1.75, 17: 1.70}
SCALES = (1.4, 1.6, 1.8, 2.0)


def unit_surface(n: int) -> np.ndarray:
  if n < 8:
    raise InputError("ESP density too small to construct a sphere")
  equator = int(np.sqrt(np.pi * n))
  vertical = equator // 2
  rings = []
  for i in range(vertical + 1):
    phi = np.pi * i / vertical
    xy, z = np.sin(phi), np.cos(phi)
    count = max(1, int(equator * xy + 1e-10))
    theta = 2 * np.pi * np.arange(count) / count
    rings.append(np.column_stack((np.cos(theta) * xy, np.sin(theta) * xy, np.full(count, z))))
  return np.concatenate(rings)[:n]


def surface_points(coords_bohr, numbers, density=1.0) -> np.ndarray:
  if not np.isfinite(density) or density <= 0:
    raise InputError("ESP density must be positive (points/angstrom^2)")
  try:
    radii = np.array([RADII[z] for z in numbers]) / BOHR
  except KeyError as exc:
    raise InputError(f"No validated ESP radius for atomic number {exc.args[0]}") from exc
  points = []
  for scale in SCALES:
    scaled = radii * scale
    for i, coord in enumerate(coords_bohr):
      count = int(density * BOHR**2 * 4 * np.pi * scaled[i] ** 2)
      candidate = coord + scaled[i] * unit_surface(count)
      keep = np.all(cdist(candidate, coords_bohr) + 1e-10 >= scaled, axis=1)
      points.append(candidate[keep])
  result = np.concatenate(points)
  if len(result) <= len(numbers):
    raise InputError("Insufficient ESP grid points")
  return result


def evaluate_esp(mf, points, device: str, batch_size=256) -> np.ndarray:
  mol = mf.mol
  dm = mf.make_rdm1()
  if dm.ndim == 3:
    dm = dm.sum(axis=0)
  values = []
  for offset in range(0, len(points), batch_size):
    grid = points[offset : offset + batch_size]
    nuclear = (1.0 / cdist(grid, mol.atom_coords())) @ mol.atom_charges()
    if device == "gpu":
      from gpu4pyscf.gto.int3c1e import int1e_grids

      electron = int1e_grids(mol, grid, dm=dm, direct_scf_tol=1e-14).get()
    else:
      integrals = mol.intor("int1e_grids", grids=grid)
      electron = np.einsum("pij,ji->p", integrals, dm)
    values.append(nuclear - electron)
  return np.concatenate(values)


def write_esp(path, mol, coords_bohr, points, potential):
  if max(len(mol.atoms), len(points)) > 99999:
    raise InputError("Amber ESP text format supports at most 99999 atoms/points")
  with path.open("w", encoding="utf-8") as handle:
    handle.write(f"{len(mol.atoms):5d}{len(points):5d}\n")
    for atom, z, xyz in zip(mol.atoms, mol.numbers, coords_bohr, strict=True):
      handle.write(" " * 16 + "".join(f"{v:16.7E}" for v in xyz) + f" {z:d} {atom.element}\n")
    for value, xyz in zip(potential, points, strict=True):
      handle.write("".join(f"{v:16.7E}" for v in (value, *xyz)) + "\n")
