"""PDB identity preservation and explicit hydrogen preparation."""

import importlib.util
import os
import subprocess
import warnings
from collections import Counter
from dataclasses import replace
from pathlib import Path

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, rdDetermineBonds, rdMolDescriptors

from pyligprep.errors import DependencyError, InputError
from pyligprep.files import file_hash, read_json, write_json
from pyligprep.model import ELEMENTS, Atom, Molecule

OBABEL_TIMEOUT = 120  # seconds; a single ligand converts in well under a second


def read_pdb(path: Path) -> tuple[tuple[Atom, ...], str]:
  text = path.read_text(encoding="utf-8")
  atoms = []
  models = 0
  for line in text.splitlines():
    if line.startswith("MODEL "):
      models += 1
    if line[:6].strip() not in ("ATOM", "HETATM"):
      continue
    if line[16:17].strip():
      raise InputError("Resolve PDB alternate locations before preparation")
    try:
      name = line[12:16].strip()
      element = line[76:78].strip().capitalize()
      if not element:
        raw = line[12:16]
        element = raw.strip().lstrip("0123456789")[:1].upper()
        if len(raw) >= 2 and raw[0].isalpha() and raw[1].isalpha():
          element = raw[:2].capitalize()
      field = line[78:80].strip()
      formal = (int(field[0]) * (1 if field[1] == "+" else -1)) if field else 0
      atom = Atom(
        name,
        element,
        (float(line[30:38]), float(line[38:46]), float(line[46:54])),
        int(line[6:11]),
        line[17:20].strip(),
        line[21:22].strip(),
        int(line[22:26]),
        line[26:27].strip(),
        formal,
      )
    except (ValueError, IndexError) as exc:
      raise InputError(f"Malformed PDB atom: {line}") from exc
    atoms.append(atom)
  residues = {(a.residue, a.chain, a.residue_id, a.insertion) for a in atoms}
  if not atoms or models > 1 or len(residues) != 1:
    raise InputError("Input must contain one model and one ligand residue")
  if len({a.serial for a in atoms}) != len(atoms):
    raise InputError("PDB atom serials must be unique")
  if any(a.element not in ELEMENTS for a in atoms):
    raise InputError("Unsupported or ambiguous element; fill PDB element columns (H C N O F P S Cl)")
  if len({a.name for a in atoms}) != len(atoms):
    raise InputError("PDB atom names must be unique within the ligand")
  # antechamber infers elements from atom names when writing MOL2 and fails on
  # names such as "1H2" (old PDB hydrogen naming), so reject them up front.
  if digit_names := [a.name for a in atoms if a.name[:1].isdigit()]:
    raise InputError(
      "PDB atom names must not start with a digit (antechamber cannot infer their elements); rename: "
      + ", ".join(digit_names)
    )
  return tuple(atoms), text


def to_rdkit(mol: Molecule):
  rw = Chem.RWMol()
  for atom in mol.atoms:
    item = Chem.Atom(atom.element)
    item.SetFormalCharge(atom.formal_charge)
    rw.AddAtom(item)
  orders = {
    1: Chem.BondType.SINGLE,
    2: Chem.BondType.DOUBLE,
    3: Chem.BondType.TRIPLE,
    1.5: Chem.BondType.AROMATIC,
  }
  for i, j, order in mol.bonds:
    rw.AddBond(i, j, orders[order])
  result = rw.GetMol()
  conf = Chem.Conformer(len(mol.atoms))
  for i, atom in enumerate(mol.atoms):
    conf.SetAtomPosition(i, atom.xyz)
  result.AddConformer(conf)
  Chem.SanitizeMol(result)
  Chem.AssignStereochemistryFrom3D(result)
  return result


def from_rdkit(rd_mol, original: tuple[Atom, ...], multiplicity: int = 1) -> Molecule:
  if rd_mol.GetNumAtoms() < len(original):
    raise InputError("Chemical perception removed input atoms")
  atoms = []
  used = {a.name for a in original}
  serial = max(a.serial for a in original)
  positions = rd_mol.GetConformer().GetPositions()
  counter = 1
  for i, item in enumerate(rd_mol.GetAtoms()):
    if i < len(original):
      atom = original[i]
      if atom.element != item.GetSymbol():
        raise InputError("Chemical perception changed atom ordering")
    else:
      if item.GetAtomicNum() != 1:
        raise InputError("Only hydrogen atoms may be added")
      while f"H{counter}" in used:
        counter += 1
      name = f"H{counter}"
      used.add(name)
      serial += 1
      atom = replace(original[0], name=name, element="H", serial=serial)
    atoms.append(replace(atom, xyz=tuple(positions[i]), formal_charge=item.GetFormalCharge()))
  bonds = tuple(
    sorted(
      (
        min(b.GetBeginAtomIdx(), b.GetEndAtomIdx()),
        max(b.GetBeginAtomIdx(), b.GetEndAtomIdx()),
        b.GetBondTypeAsDouble(),
      )
      for b in rd_mol.GetBonds()
    )
  )
  result = Molecule(tuple(atoms), bonds, sum(a.formal_charge for a in atoms), multiplicity)
  result.validate()
  if len(Chem.GetMolFrags(rd_mol)) != 1:
    raise InputError("Disconnected fragments/salts are not supported")
  return result


def _template_molecule(smiles: str | None, template: Path | None):
  if smiles and template:
    raise InputError("Choose either --smiles or --template")
  if smiles:
    result = Chem.MolFromSmiles(smiles)
  elif template:
    suffix = template.suffix.lower()
    if suffix == ".sdf":
      supplier = Chem.SDMolSupplier(str(template), removeHs=True)
      result = supplier[0] if len(supplier) == 1 else None
    elif suffix == ".mol2":
      result = Chem.MolFromMol2File(str(template), removeHs=True)
    else:
      result = Chem.MolFromMolFile(str(template), removeHs=True)
  else:
    return None
  if result is None:
    raise InputError("Cannot parse template; use one SDF/MOL/MOL2 molecule or valid SMILES")
  return Chem.RemoveHs(result)


def _single_subdirectory(path: Path) -> Path | None:
  found = [item for item in path.glob("*") if item.is_dir()] if path.is_dir() else []
  return found[0] if len(found) == 1 else None


def obabel_command() -> tuple[Path, dict[str, str]] | None:
  """Return the `obabel` program shipped in openbabel-wheel and its environment, without importing Open Babel.

  Open Babel is GPL-2.0-only, which is incompatible with this GPL-3.0 package, so it is only ever run as a
  separate program over stdin/stdout. Never `import openbabel`. The wheel's own `obabel` wrapper imports the
  package to set BABEL_LIBDIR/BABEL_DATADIR; here the plugin and data directories are passed to the binary.
  """
  spec = importlib.util.find_spec("openbabel")
  if spec is None or not spec.submodule_search_locations:
    return None
  name = "obabel.exe" if os.name == "nt" else "obabel"
  for root in map(Path, spec.submodule_search_locations):
    program = root / "bin" / name
    plugins = _single_subdirectory(root / "lib" / "openbabel")
    data = _single_subdirectory(root / "share" / "openbabel")
    if program.is_file() and plugins and data:
      return program, {**os.environ, "BABEL_LIBDIR": str(plugins), "BABEL_DATADIR": str(data)}
  return None


def _obabel_pdb_to_mol(text: str) -> str:
  """Perceive bond orders of a PDB block with the `obabel` program; return an MDL mol block."""
  command = obabel_command()
  if command is None:
    raise DependencyError(
      "PDB-only hydrogen addition needs the obabel program of openbabel-wheel; "
      "reinstall pyligprep or provide --smiles/--template"
    )
  program, environment = command
  try:
    run = subprocess.run(
      [str(program), "-ipdb", "-omol"],
      input=text,
      capture_output=True,
      text=True,
      env=environment,
      timeout=OBABEL_TIMEOUT,
      check=False,
    )
  except (OSError, subprocess.TimeoutExpired) as exc:
    raise DependencyError(f"Cannot run {program}: {exc}") from exc
  if run.returncode != 0 or not run.stdout.strip():
    detail = run.stderr.strip()
    raise InputError("Open Babel could not read the PDB" + (f": {detail}" if detail else ""))
  return run.stdout


def _perceive(text: str, atoms, charge: int | None, smiles=None, template=None, complete=False):
  reference = _template_molecule(smiles, template)
  if reference is not None:
    raw = Chem.MolFromPDBBlock(text, sanitize=False, removeHs=False, proximityBonding=True)
    if raw is None:
      raise InputError("Cannot parse PDB connectivity")
    try:
      return AllChem.AssignBondOrdersFromTemplate(reference, raw)
    except ValueError as exc:
      raise InputError("Template does not match PDB connectivity/atoms") from exc
  if complete and charge is not None:
    xyz = f"{len(atoms)}\n\n" + "\n".join(f"{a.element} {' '.join(map(str, a.xyz))}" for a in atoms)
    raw = Chem.MolFromXYZBlock(xyz)
    try:
      rdDetermineBonds.DetermineBonds(raw, charge=charge, allowChargedFragments=True)
      return raw
    except ValueError as exc:
      raise InputError(
        "Cannot assign bonds for this complete structure; use --smiles/--template or prepare it with hadd"
      ) from exc
  result = Chem.MolFromMolBlock(_obabel_pdb_to_mol(text), removeHs=False, sanitize=True)
  if result is None:
    raise InputError("Bond perception failed; supply a chemical template")
  warnings.warn(
    "Bond orders/protonation were inferred from PDB. Review the output and use --smiles/--template for a specified state.",
    stacklevel=2,
  )
  return result


def sidecar(path: Path) -> Path:
  return path.with_suffix(".pyligprep.json")


def _read_sidecar(path: Path, atoms) -> Molecule | None:
  if not sidecar(path).exists():
    return None
  data = read_json(sidecar(path))
  if data.get("schema") != 1 or data.get("pdb_sha256") != file_hash(path):
    raise InputError("PDB sidecar is stale or unsupported; regenerate preparation after editing")
  mol = Molecule.from_dict(data["molecule"])
  if [(a.name, a.element) for a in mol.atoms] != [(a.name, a.element) for a in atoms]:
    raise InputError("PDB sidecar atom identities do not match")
  return mol.with_coords([a.xyz for a in atoms])


def write_pdb(path: Path, mol: Molecule, exclusive: bool = True) -> None:
  with path.open("x" if exclusive else "w", encoding="utf-8") as handle:
    for atom in mol.atoms:
      if not 0 < atom.serial <= 99999 or any(not -999.999 <= c <= 9999.999 for c in atom.xyz):
        raise InputError("Coordinates or serials exceed PDB field widths")
      name = f" {atom.name:<3}" if len(atom.element) == 1 and len(atom.name) < 4 else f"{atom.name:<4}"
      formal = f"{abs(atom.formal_charge)}{'+' if atom.formal_charge > 0 else '-'}" if atom.formal_charge else "  "
      x, y, z = atom.xyz
      handle.write(
        f"HETATM{atom.serial:5d} {name} {atom.residue:>3} {atom.chain:1}{atom.residue_id:4d}{atom.insertion:1}   {x:8.3f}{y:8.3f}{z:8.3f}{1.0:6.2f}{0.0:6.2f}          {atom.element:>2}{formal}\n"
      )
    for i, j, _ in mol.bonds:
      handle.write(f"CONECT{mol.atoms[i].serial:5d}{mol.atoms[j].serial:5d}\n")
    handle.write("END\n")


def add_hydrogens(source: Path, output: Path, smiles=None, template=None, charge=None) -> Molecule:
  if output.exists() or sidecar(output).exists():
    raise FileExistsError(f"Refusing to overwrite {output} or its sidecar")
  atoms, text = read_pdb(source)
  saved = _read_sidecar(source, atoms)
  raw = to_rdkit(saved) if saved and not (smiles or template) else _perceive(text, atoms, charge, smiles, template)
  try:
    # We assign residue metadata ourselves. RDKit's addResidueInfo path can
    # stall on existing named PDB hydrogens in partially/fully protonated files.
    added = Chem.AddHs(raw, addCoords=True, addResidueInfo=False)
    mol = from_rdkit(added, atoms)
  except ValueError as exc:
    raise InputError(f"Hydrogen addition failed: {exc}") from exc
  if charge is not None and charge != mol.charge:
    raise InputError(
      f"Inferred charge {mol.charge} differs from requested {charge}; specify protonation with --smiles/--template"
    )
  if not np.allclose(mol.coords[: len(atoms)], [a.xyz for a in atoms], rtol=0, atol=1e-6):
    raise InputError("Hydrogen addition moved existing atoms")
  output.parent.mkdir(parents=True, exist_ok=True)
  write_pdb(output, mol)
  write_json(
    sidecar(output),
    {
      "schema": 1,
      "pdb_sha256": file_hash(output),
      "molecule": mol.to_dict(),
      "source_sha256": file_hash(source),
      "smiles": Chem.MolToSmiles(Chem.RemoveHs(added)),
      "formula": rdMolDescriptors.CalcMolFormula(added),
      "added_hydrogens": [
        {
          "name": mol.atoms[i].name,
          "parent": mol.atoms[a.GetNeighbors()[0].GetIdx()].name,
        }
        for i, a in enumerate(added.GetAtoms())
        if i >= len(atoms)
      ],
    },
  )
  counts = Counter(a.element for a in mol.atoms)
  print(f"Wrote {output}: {len(mol.atoms)} atoms ({dict(counts)}), charge={mol.charge}; review before opt/run")
  return mol


def load_molecule(path: Path, charge: int, multiplicity: int, smiles=None, template=None) -> Molecule:
  atoms, text = read_pdb(path)
  saved = _read_sidecar(path, atoms)
  if saved is None or smiles or template:
    raw = _perceive(text, atoms, charge, smiles, template, complete=True)
    mol = from_rdkit(raw, atoms, multiplicity)
  else:
    mol = replace(saved, multiplicity=multiplicity)
    raw = to_rdkit(mol)
  if any(a.GetNumImplicitHs() or a.GetNumExplicitHs() for a in raw.GetAtoms()):
    raise InputError("Input lacks explicit hydrogen atoms; run hadd and review first")
  mol = replace(mol, charge=charge, multiplicity=multiplicity)
  mol.validate()
  return mol
