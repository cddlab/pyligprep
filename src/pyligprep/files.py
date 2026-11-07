"""Deterministic, atomic, non-pickle artifacts."""

import hashlib
import json
import os
import tempfile
from pathlib import Path

from pyligprep.errors import InputError


def digest(data: dict) -> str:
  return hashlib.sha256(json.dumps(data, sort_keys=True, allow_nan=False).encode()).hexdigest()


def file_hash(path: Path) -> str:
  result = hashlib.sha256()
  with path.open("rb") as handle:
    for block in iter(lambda: handle.read(1024 * 1024), b""):
      result.update(block)
  return result.hexdigest()


def read_json(path: Path) -> dict:
  try:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
      raise ValueError("Expected JSON object")
    return data
  except (OSError, ValueError) as exc:
    raise InputError(f"Cannot read artifact {path}: {exc}") from exc


def write_json(path: Path, data: dict) -> None:
  text = json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + "\n"
  with tempfile.NamedTemporaryFile(
    mode="w", dir=path.parent, prefix=".writing-", encoding="utf-8", delete=False
  ) as handle:
    temporary = Path(handle.name)
    handle.write(text)
    handle.flush()
    os.fsync(handle.fileno())
  temporary.replace(path)


def write_xyz(path: Path, mol, comment: str = "", append: bool = False) -> None:
  with path.open("a" if append else "w", encoding="utf-8") as handle:
    handle.write(f"{len(mol.atoms)}\n{comment}\n")
    for atom in mol.atoms:
      x, y, z = atom.xyz
      handle.write(f"{atom.element:2s} {x:20.12f} {y:20.12f} {z:20.12f}\n")
