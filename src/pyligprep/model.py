"""Portable chemical identity and full precision coordinates, independent of QM."""

from dataclasses import asdict, dataclass, replace
from typing import Any

import numpy as np

from pyligprep.errors import InputError

ELEMENTS = {"H": 1, "C": 6, "N": 7, "O": 8, "F": 9, "P": 15, "S": 16, "Cl": 17}


@dataclass(frozen=True)
class Atom:
  name: str
  element: str
  xyz: tuple[float, float, float]
  serial: int
  residue: str = "LIG"
  chain: str = "A"
  residue_id: int = 1
  insertion: str = ""
  formal_charge: int = 0


@dataclass(frozen=True)
class Molecule:
  atoms: tuple[Atom, ...]
  bonds: tuple[tuple[int, int, float], ...]
  charge: int = 0
  multiplicity: int = 1

  @property
  def coords(self) -> np.ndarray:
    return np.asarray([a.xyz for a in self.atoms], dtype=float)

  @property
  def numbers(self) -> list[int]:
    return [ELEMENTS[a.element] for a in self.atoms]

  def validate(self) -> None:
    if not self.atoms or len(self.atoms) > 99999:
      raise InputError("Require 1..99999 atoms in a single ligand")
    names = [a.name for a in self.atoms]
    if len(set(names)) != len(names) or any(
      not n or len(n) > 4 or not n.isascii() or any(c.isspace() for c in n) for n in names
    ):
      raise InputError("Atom names must be unique ASCII names of 1..4 characters")
    if any(a.element not in ELEMENTS for a in self.atoms):
      raise InputError("Supported elements: H C N O F P S Cl")
    if self.coords.shape != (len(self.atoms), 3) or not np.isfinite(self.coords).all():
      raise InputError("Invalid atomic coordinates")
    electrons = sum(self.numbers) - self.charge
    spin = self.multiplicity - 1
    if spin < 0 or electrons < spin or (electrons - spin) % 2:
      raise InputError("Charge and multiplicity are inconsistent with electron count")
    if sum(a.formal_charge for a in self.atoms) != self.charge:
      raise InputError(
        "Formal charges disagree with --charge; specify the protonation state using --smiles or --template"
      )
    seen = set()
    for i, j, order in self.bonds:
      if not 0 <= i < j < len(self.atoms) or (i, j) in seen or order not in (1, 1.5, 2, 3):
        raise InputError("Invalid or duplicate bond")
      seen.add((i, j))

  def with_coords(self, coords) -> "Molecule":
    coords = np.asarray(coords, dtype=float)
    if coords.shape != self.coords.shape or not np.isfinite(coords).all():
      raise InputError("Geometry shape or coordinates are invalid")
    return replace(
      self,
      atoms=tuple(replace(a, xyz=tuple(p)) for a, p in zip(self.atoms, coords, strict=True)),
    )

  def to_dict(self) -> dict:
    return asdict(self)

  @classmethod
  def from_dict(cls, data: dict) -> "Molecule":
    try:
      fields: list[dict[str, Any]] = [{**a, "xyz": tuple(a["xyz"])} for a in data["atoms"]]
      atoms = tuple(Atom(**f) for f in fields)
      mol = cls(
        atoms,
        tuple(tuple(b) for b in data["bonds"]),
        data["charge"],
        data["multiplicity"],
      )
      mol.validate()
      return mol
    except (KeyError, TypeError, ValueError) as exc:
      raise InputError("Invalid molecule artifact") from exc
