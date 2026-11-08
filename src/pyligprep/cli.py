"""CLI entry points; quantum modules are imported only when needed."""

import argparse
import json
import sys
from dataclasses import fields, replace
from pathlib import Path

from pyligprep import __version__
from pyligprep.errors import InputError, PreparationError

VERSION_LINE = f"pyligprep {__version__}"


class _Parser(argparse.ArgumentParser):
  """Parser for the command and every subcommand: `-V/--version` and a version line atop `--help`.

  Subparsers are created with the parent's class, so a subcommand added later gets both automatically.
  """

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.add_argument("-V", "--version", action="version", version=VERSION_LINE, help="show the version and exit")

  def format_help(self):
    return f"{VERSION_LINE}\n\n{super().format_help()}"


def _chemical_arguments(parser):
  parser.add_argument("--charge", type=int)
  parser.add_argument("--multiplicity", type=int)
  parser.add_argument("--smiles")
  parser.add_argument("--template", type=Path)


def _quantum_arguments(parser):
  parser.add_argument("--device", choices=("auto", "gpu", "cpu"))
  parser.add_argument(
    "--threads",
    type=int,
    help=(
      "CPU threads: automatically use Slurm CPUs per task (single task only), including on resume; "
      "otherwise keep the saved value or default to 8. An explicit value must fit the Slurm/CPU-affinity limit"
    ),
  )
  parser.add_argument("--memory-mb", type=int)
  parser.add_argument(
    "--scf-maxcycle",
    dest="scf_max_cycle",
    type=int,
    help="SCF cycle limit (default 512; DFT Newton SCF is capped at 100)",
  )


def _opt_arguments(parser):
  parser.add_argument("--optimizer", type=str.lower, choices=("berny", "geometric"))
  parser.add_argument("--method")
  parser.add_argument("--basis")
  parser.add_argument(
    "--opt-maxcycles",
    dest="opt_max_cycles",
    type=int,
    help="Geometry optimization cycle limit (default 200; Gaussian Opt=MaxCycles)",
  )
  parser.add_argument(
    "--opt-maxstep",
    dest="opt_max_step",
    type=int,
    metavar="N",
    help=(
      "Initial geometry trust radius on the Gaussian MaxStep scale (positive integer): "
      "0.01 N Bohr/radian for Berny; the length converted to angstrom for geomeTRIC "
      "(different coordinate metric; tmax raised if needed). "
      "Omit to keep the saved value on resume or the existing backend default"
    ),
  )
  parser.add_argument("--scf-type", choices=("auto", "rks", "uks"))
  parser.add_argument(
    "--opt-overlap-threshold",
    type=float,
    metavar="VALUE",
    help=(
      "AO overlap eigenvalue cutoff for optimization only (finite positive number, e.g. 1e-8). "
      "Omit to keep the saved value on resume or the backend default; does not affect RESP HF"
    ),
  )
  parser.add_argument("--constraints", type=Path, help="geomeTRIC constraints file")


def _resp_arguments(parser):
  parser.add_argument("--density", type=float, help="ESP points/angstrom^2 (default 1)")
  parser.add_argument("--qwt1", type=float)
  parser.add_argument("--qwt2", type=float)
  parser.add_argument("--max-iterations", type=int)
  parser.add_argument("--tolerance", type=float)
  parser.add_argument("--equivalence-depth", type=int)
  parser.add_argument("--resp-constraints", type=Path, help="JSON with stage1/stage2 ivary arrays")


def make_parser():
  parser = _Parser(prog="pyligprep", description="Ligand optimization, RESP and GAFF2 preparation")
  commands = parser.add_subparsers(dest="command", required=True)
  hadd = commands.add_parser("hadd", help="Add hydrogens for user review; never starts QM")
  hadd.add_argument("-i", "--input", type=Path, required=True)
  hadd.add_argument("-o", "--output", type=Path, required=True)
  hadd.add_argument("--charge", type=int)
  hadd.add_argument("--smiles")
  hadd.add_argument("--template", type=Path)
  doctor = commands.add_parser("doctor", help="Report versions, GPU and AmberTools availability")
  doctor.add_argument("--amber-bin", type=Path)
  check = commands.add_parser("check", help="Validate an explicit-H PDB without QM")
  check.add_argument("-i", "--input", type=Path, required=True)
  _chemical_arguments(check)
  for name in ("opt", "resp", "export", "run"):
    sub = commands.add_parser(name)
    source = sub.add_mutually_exclusive_group(required=True)
    if name != "export":
      source.add_argument("-i", "--input", type=Path)
    if name in ("resp", "export"):
      source.add_argument("--from", dest="source", type=Path)
    source.add_argument(
      "--resume",
      type=Path,
      help="Resume an output directory from the working files kept in its tmp/",
    )
    sub.add_argument("-o", "--output", type=Path)
    if name != "export":
      _chemical_arguments(sub)
      _quantum_arguments(sub)
    if name in ("opt", "run"):
      _opt_arguments(sub)
    if name in ("resp", "run"):
      _resp_arguments(sub)
    if name == "run":
      sub.add_argument(
        "--keep-tmp",
        action="store_true",
        help="Keep tmp/ (working files for --resume) after success; by default it is removed",
      )
    if name in ("export", "run"):
      sub.add_argument("--amber-bin", type=Path)
      sub.add_argument(
        "-r",
        "--resname",
        help="Case-sensitive residue name and output prefix (1-3 ASCII alphanumeric characters, starting with a letter)",
      )
  return parser


def _options(cls, args, saved, group):
  values = {**saved.get(group, {})}
  unknown = values.keys() - {field.name for field in fields(cls)}
  if unknown:
    raise InputError(
      f"Unsupported saved {group} settings: {', '.join(sorted(unknown))}; "
      "start a new calculation in a new output directory with -o"
    )
  for field in fields(cls):
    value = getattr(args, field.name, None)
    if field.name == "constraints":
      if group == "resp":
        from pyligprep.files import read_json

        path = getattr(args, "resp_constraints", None)
        value = read_json(path) if path else None
      else:
        value = value.read_text() if value else None
    if value is not None:
      values[field.name] = value
  return cls(**values)


def _quantum_options(args, saved):
  from pyligprep.quantum import QuantumOptions
  from pyligprep.resources import cpu_affinity, in_slurm_job, resolve_threads

  quantum = _options(QuantumOptions, args, saved, "quantum")
  # A saved thread count belongs to the previous allocation, not the current job.
  requested = args.threads if in_slurm_job() else quantum.threads
  quantum = replace(quantum, threads=resolve_threads(requested))
  quantum.validate()
  if in_slurm_job():
    affinity = cpu_affinity()
    allowed = ",".join(map(str, affinity)) if affinity is not None else "unavailable"
    print(f"CPU threads: {quantum.threads} (Slurm; allowed logical CPU IDs: {allowed})")
  return quantum


def _molecule(args):
  from pyligprep.artifacts import completed_result, load_state
  from pyligprep.chemistry import load_molecule
  from pyligprep.errors import InputError
  from pyligprep.model import Molecule

  if getattr(args, "resume", None):
    state = load_state(args.resume)
    if state.get("stage") != args.command:
      raise InputError("Resume directory belongs to a different subcommand")
    mol = Molecule.from_dict(state["molecule"])
    settings = state["settings"]
  elif getattr(args, "source", None):
    result = completed_result(args.source, "opt")
    mol = Molecule.from_dict(result["molecule"])
    settings = {}
  else:
    if args.charge is None or args.multiplicity is None:
      raise InputError("--charge and --multiplicity are mandatory with --input")
    mol = load_molecule(args.input, args.charge, args.multiplicity, args.smiles, args.template)
    settings = {}
  for key in ("charge", "multiplicity"):
    supplied = getattr(args, key, None)
    if supplied is not None and supplied != getattr(mol, key):
      raise InputError(f"--{key} differs from saved molecule")
  return mol, settings


def _doctor(args):
  from pyligprep.amber import amber_tools
  from pyligprep.artifacts import versions
  from pyligprep.dispersion import dispersion_backend
  from pyligprep.quantum import gpu_probe

  available, message = gpu_probe()
  report = {
    "versions": versions(),
    "gpu": {"available": available, "detail": message},
    "dispersion": {"d3_backend": dispersion_backend()},
  }
  try:
    report["amber"] = amber_tools(args.amber_bin)
  except PreparationError as exc:
    report["amber"] = {"error": str(exc)}
  print(json.dumps(report, indent=2))


def dispatch(args):
  from pyligprep.errors import InputError

  if args.command == "doctor":
    return _doctor(args)
  if args.command == "hadd":
    from pyligprep.chemistry import add_hydrogens

    return add_hydrogens(args.input, args.output, args.smiles, args.template, args.charge)
  if args.command == "check":
    mol, _ = _molecule(args)
    print(f"Valid: {len(mol.atoms)} atoms, charge={mol.charge}, multiplicity={mol.multiplicity}")
    return mol
  output = args.resume or args.output
  if output is None or (args.resume and args.output and args.output.resolve() != args.resume.resolve()):
    raise InputError("Specify -o for a new calculation; --resume already identifies its output")
  if args.command == "export":
    return _export(args, output)
  mol, saved = _molecule(args)
  quantum = _quantum_options(args, saved)
  if args.command in ("opt", "run"):
    from pyligprep.optimization import OptimizationOptions, optimize

    optimization = _options(OptimizationOptions, args, saved, "optimization")
    if args.command == "opt":
      return optimize(mol, output, optimization, quantum, bool(args.resume))
  from pyligprep.charges import RespOptions, calculate_charges

  resp = _options(RespOptions, args, saved, "resp")
  if args.command == "resp":
    return calculate_charges(mol, output, resp, quantum, bool(args.resume))
  from pyligprep.pipeline import run_pipeline

  amber_bin = args.amber_bin or (Path(saved["amber_bin"]) if saved.get("amber_bin") else None)
  return run_pipeline(
    mol,
    output,
    optimization,
    resp,
    quantum,
    amber_bin,
    bool(args.resume),
    resname=args.resname if args.resname is not None else saved.get("resname"),
    keep_tmp=args.keep_tmp,
  )


def _export(args, output):
  from pyligprep.amber import export_parameters
  from pyligprep.artifacts import TMP, completed_result, load_state
  from pyligprep.errors import InputError
  from pyligprep.files import read_json

  if args.resume:
    state = load_state(args.resume, "export")
    source = args.resume / TMP / "resp-source.json"
    if not source.is_file():
      raise InputError(f"{source} is missing (export stopped before saving its RESP input); rerun export --from <resp>")
    result = read_json(source)
    if args.amber_bin is None:
      args.amber_bin = Path(state["settings"]["tools"]["antechamber"]).parent
  else:
    result = completed_result(args.source, "resp")
  return export_parameters(result, output, args.amber_bin, bool(args.resume), resname=args.resname)


def main(argv=None) -> int:
  parser = make_parser()
  args = parser.parse_args(argv)
  try:
    dispatch(args)
    if args.command in ("opt", "resp", "export", "run"):
      print(f"Completed {args.command}: {(args.resume or args.output).resolve()}")
  except (
    PreparationError,
    FileExistsError,
    FileNotFoundError,
    PermissionError,
    ValueError,
  ) as exc:
    print(f"pyligprep: {exc}", file=sys.stderr)
    return 2
  return 0
