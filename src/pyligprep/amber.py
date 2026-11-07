"""AmberTools binaries with explicit charges and verified atom identities."""

import os
import re
import shutil
import subprocess
from dataclasses import replace
from pathlib import Path

import numpy as np

from pyligprep.artifacts import STATE, Stage, load_state
from pyligprep.charge_rounding import POLICY, final_equivalence_groups, round_charges, validate_export_charges
from pyligprep.errors import CalculationError, DependencyError, InputError
from pyligprep.files import digest, file_hash, read_json, write_json
from pyligprep.model import Molecule


def resolve_resname(resname: str | None, output: Path, resume: bool, publish: bool = True) -> str | None:
  """Validate a case-sensitive override, inheriting it on stage resume."""
  if resname is None and resume:
    state = load_state(output) if publish else read_json(output / STATE)
    resname = state["settings"].get("resname")
  if resname is not None:
    _validate_resname(resname)
  return resname


def _validate_resname(resname: str) -> None:
  if not isinstance(resname, str) or not re.fullmatch(r"[A-Za-z][A-Za-z0-9]{0,2}", resname):
    raise InputError("LEaP residue name must start with a letter and contain 1..3 ASCII alphanumeric characters")


def amber_tools(directory: Path | None = None) -> dict[str, str]:
  tools = {}
  for name in ("antechamber", "parmchk2", "tleap"):
    candidate = str(directory / name) if directory else shutil.which(name)
    if not candidate or not os.access(candidate, os.X_OK):
      raise DependencyError(f"Missing {name}; module load ambertools26 or use --amber-bin")
    tools[name] = str(Path(candidate).resolve())
  return tools


def amber_versions(tools: dict[str, str]) -> dict:
  """Paths and hashes of the AmberTools binaries, plus the release named in AMBERHOME/README."""
  release = None
  try:
    readme = (Path(tools["antechamber"]).parent.parent / "README").read_text(errors="replace")
    match = re.search(r"AmberTools\s?\d+", readme)
    release = match.group(0) if match else None
  except OSError:
    pass
  binaries = {name: {"path": path, "sha256": file_hash(Path(path))} for name, path in tools.items()}
  return {"ambertools": release, **binaries}


def _run(arguments: list[str], directory: Path, log_name: str) -> str:
  with (directory / log_name).open("w", encoding="utf-8") as log:
    log.write("Command: " + repr(arguments) + "\n")
    log.flush()
    result = subprocess.run(arguments, cwd=directory, stdout=log, stderr=subprocess.STDOUT, check=False)
  text = (directory / log_name).read_text(encoding="utf-8", errors="replace")
  if result.returncode:
    raise CalculationError(f"{Path(arguments[0]).name} failed ({result.returncode}); inspect {directory / log_name}")
  return text


def _write_input(path, mol, charges):
  with path.open("w", encoding="utf-8") as handle:
    handle.write(
      f"@<TRIPOS>MOLECULE\n{mol.atoms[0].residue}\n{len(mol.atoms)} {len(mol.bonds)} 1 0 0\nSMALL\nUSER_CHARGES\n\n@<TRIPOS>ATOM\n"
    )
    for i, (atom, charge) in enumerate(zip(mol.atoms, charges, strict=True), 1):
      x, y, z = atom.xyz
      handle.write(
        f"{i:7d} {atom.name:<4} {x:16.10f} {y:16.10f} {z:16.10f} {atom.element:<4} 1 {atom.residue} {charge:.6f}\n"
      )
    handle.write("@<TRIPOS>BOND\n")
    for b, (i, j, order) in enumerate(mol.bonds, 1):
      kind = "ar" if order == 1.5 else str(int(order))
      handle.write(f"{b:6d} {i + 1:6d} {j + 1:6d} {kind}\n")
    handle.write(f"@<TRIPOS>SUBSTRUCTURE\n1 {mol.atoms[0].residue} 1 GROUP 0 **** **** 0 ROOT\n")


def read_mol2(path: Path):
  atoms, bonds = [], []
  section = ""
  for line in path.read_text().splitlines():
    if line.startswith("@<TRIPOS>"):
      section = line[9:]
      continue
    parts = line.split()
    if not parts or parts[0].startswith("#"):
      continue
    if section == "ATOM":
      if len(parts) < 9:
        raise InputError(f"MOL2 lacks atom charges: {path}")
      atoms.append(
        {
          "id": int(parts[0]),
          "name": parts[1],
          "xyz": list(map(float, parts[2:5])),
          "type": parts[5],
          "residue": parts[7],
          "charge": float(parts[8]),
          "charge_text": parts[8],
        }
      )
    elif section == "BOND":
      bonds.append((int(parts[1]), int(parts[2]), parts[3]))
  return atoms, bonds


def validate_mol2(path, mol, charges, ordered=True, check_coordinates=True):
  atoms, bonds = read_mol2(path)
  _validate_mol2_residue(path, atoms, mol.atoms[0].residue)
  names = [a["name"] for a in atoms]
  expected = [a.name for a in mol.atoms]
  if len(set(names)) != len(names) or set(names) != set(expected) or (ordered and names != expected):
    raise CalculationError("AmberTools changed atom names/order")
  by_name = {a["name"]: a for a in atoms}
  validate_export_charges([by_name[a.name]["charge_text"] for a in mol.atoms], charges, mol.charge)
  if check_coordinates and not np.allclose([by_name[a.name]["xyz"] for a in mol.atoms], mol.coords, atol=6e-4, rtol=0):
    raise CalculationError("AmberTools changed coordinates")
  ids = {a["id"]: a["name"] for a in atoms}
  graph = {tuple(sorted((ids[i], ids[j]))) for i, j, _ in bonds}
  original = {tuple(sorted((mol.atoms[i].name, mol.atoms[j].name))) for i, j, _ in mol.bonds}
  if graph != original:
    raise CalculationError("AmberTools changed bond connectivity")
  if any(a["type"].lower() in ("du", "xx", "x") for a in atoms):
    raise CalculationError("Unassigned GAFF2 atom type")
  return atoms


def _validate_mol2_residue(path: Path, atoms: list[dict], residue: str) -> None:
  sections = {}
  for block in path.read_text().split("@<TRIPOS>")[1:]:
    title, _, content = block.partition("\n")
    sections[title.strip()] = content.strip().splitlines()
  title = (sections.get("MOLECULE") or [""])[0].strip()
  substructures = [
    line.split()[1]
    for line in sections.get("SUBSTRUCTURE", [])
    if len(line.split()) >= 2 and not line.lstrip().startswith("#")
  ]
  if title != residue or substructures != [residue] or any(atom["residue"] != residue for atom in atoms):
    raise CalculationError("MOL2 residue name differs (case-sensitive)")


def _validate_prepi(path, mol, charges):
  entries = {}
  residues = []
  for line in path.read_text().splitlines():
    words = line.split()
    if len(words) == 3 and words[1] == "INT":
      residues.append(words[0])
    if len(words) >= 11 and words[0].isdigit() and words[2] != "DU":
      entries[words[1]] = words[10]
  if residues != [mol.atoms[0].residue]:
    raise CalculationError("PREP residue name differs (case-sensitive)")
  if set(entries) != {a.name for a in mol.atoms}:
    raise CalculationError("PREPI physical atom names differ")
  validate_export_charges([entries[a.name] for a in mol.atoms], charges, mol.charge)


def _leap_check(directory, tools, residue, prefix):
  script = directory / "validate.leap"
  script.write_text(
    f"source leaprc.gaff2\nloadAmberParams {prefix}.frcmod\n"
    f"M = loadMol2 {prefix}.mol2\ncheck M\nsaveAmberParm M mol2.prmtop mol2.inpcrd\n"
    f"loadAmberPrep {prefix}.prep\ncheck {residue}\n"
    f"saveAmberParm {residue} prepi.prmtop prepi.inpcrd\nquit\n"
  )
  text = _run([tools["tleap"], "-f", script.name], directory, "tleap.log")
  errors = re.findall(r"Errors\s*=\s*(\d+)", text)
  if (
    not errors
    or any(int(n) for n in errors)
    or "FATAL" in text.upper()
    or not all((directory / f).is_file() for f in ("mol2.prmtop", "prepi.prmtop"))
  ):
    raise CalculationError(f"LEaP validation failed; inspect {directory / 'tleap.log'}")


def export_parameters(
  result: dict,
  output: Path,
  amber_bin: Path | None = None,
  resume=False,
  resname: str | None = None,
  publish=True,
):
  resname = resolve_resname(resname, output, resume, publish)
  molecule = Molecule.from_dict(result["molecule"])
  if resname is not None:
    molecule = replace(
      molecule,
      atoms=tuple(replace(atom, residue=resname) for atom in molecule.atoms),
    )
  prefix = resname if resname is not None else "ligand"
  mol2, prep, frcmod = (f"{prefix}.{ext}" for ext in ("mol2", "prep", "frcmod"))
  raw = np.asarray(result["charges"], dtype=float)
  if raw.shape != (len(molecule.atoms),):
    raise InputError("Invalid RESP charges for export")
  groups = final_equivalence_groups(molecule, result.get("constraints"))
  charges, rounding = round_charges(raw, molecule.charge, groups)
  rounding["atoms"] = [a.name for a in molecule.atoms]
  residue = molecule.atoms[0].residue
  _validate_resname(residue)
  tools = amber_tools(amber_bin)
  settings = {
    "charge_hash": digest({"charges": raw.tolist()}),
    "charge_export_policy": POLICY,
    "equivalence_groups": groups,
    "prep_extension": ".prep",
    "force_field": "gaff2",
    "tools": tools,
    "resname": resname,
  }
  with Stage(output, "export", molecule.to_dict(), settings, resume, publish=publish) as stage:
    if stage.cached is not None:
      return stage.cached
    directory = stage.directory
    write_json(directory / "resp-source.json", result)
    write_json(directory / "charge-rounding.json", rounding)
    _write_input(directory / "input.mol2", molecule, charges)
    np.savetxt(directory / "charges.txt", charges, fmt="%.6f")
    common = [
      "-at",
      "gaff2",
      "-an",
      "n",
      "-du",
      "n",
      "-eq",
      "0",
      "-pf",
      "n",
      "-rn",
      residue,
      "-nc",
      str(molecule.charge),
      "-m",
      str(molecule.multiplicity),
    ]
    _run(
      [
        tools["antechamber"],
        "-i",
        "input.mol2",
        "-fi",
        "mol2",
        "-o",
        mol2,
        "-fo",
        "mol2",
        "-c",
        "rc",
        "-cf",
        "charges.txt",
        "-j",
        "1",
        "-seq",
        "n",
        *common,
      ],
      directory,
      "typing.log",
    )
    atoms = validate_mol2(directory / mol2, molecule, charges)
    _run(
      [
        tools["antechamber"],
        "-i",
        mol2,
        "-fi",
        "mol2",
        "-o",
        prep,
        "-fo",
        "prepi",
        "-j",
        "0",
        *common,
      ],
      directory,
      "prepi.log",
    )
    _validate_prepi(directory / prep, molecule, charges)
    _run(
      [
        tools["antechamber"],
        "-i",
        prep,
        "-fi",
        "prepi",
        "-o",
        "roundtrip.mol2",
        "-fo",
        "mol2",
        "-j",
        "0",
        *common,
      ],
      directory,
      "roundtrip.log",
    )
    validate_mol2(
      directory / "roundtrip.mol2",
      molecule,
      charges,
      ordered=False,
      check_coordinates=False,
    )
    _run(
      [
        tools["parmchk2"],
        "-i",
        mol2,
        "-f",
        "mol2",
        "-o",
        frcmod,
        "-s",
        "gaff2",
      ],
      directory,
      "parmchk2.log",
    )
    if "ATTN" in (directory / frcmod).read_text():
      raise CalculationError(f"Unresolved GAFF2 parameters (ATTN); inspect {frcmod}")
    _leap_check(directory, tools, residue, prefix)
    write_json(directory / "atom-map.json", {"atoms": atoms})
    names = (
      mol2,
      prep,
      frcmod,
      "atom-map.json",
      "charge-rounding.json",
    )
    return stage.complete(
      {
        "molecule": molecule.to_dict(),
        "charges": charges.tolist(),
        "resp_charges": raw.tolist(),
        "charge_rounding": rounding,
        "files": {n: file_hash(directory / n) for n in names},
      },
      deliverables=(mol2, prep, frcmod),
      binaries=amber_versions(tools),
    )
