"""Shared orchestration used by both the CLI and Python callers."""

import shutil
import sys
from dataclasses import asdict
from pathlib import Path

from pyligprep.amber import amber_tools, export_parameters, resolve_resname
from pyligprep.artifacts import STATE, Stage
from pyligprep.charges import RespOptions, calculate_charges
from pyligprep.errors import CalculationError
from pyligprep.files import file_hash
from pyligprep.model import Molecule
from pyligprep.optimization import OptimizationOptions, optimize
from pyligprep.quantum import QuantumOptions


def run_pipeline(
  molecule: Molecule,
  output: Path,
  optimization=OptimizationOptions(),
  resp=RespOptions(),
  quantum=QuantumOptions(),
  amber_bin=None,
  resume=False,
  resname: str | None = None,
  keep_tmp=False,
):
  """Run opt -> resp -> export with child stages in `<output>/tmp`, publishing only the deliverables.

  `tmp/` is removed after success unless `keep_tmp`; after a failure it is always kept for `--resume`.
  """
  resname = resolve_resname(resname, output, resume)
  amber_tools(amber_bin)
  settings = {
    "optimization": asdict(optimization),
    "resp": asdict(resp),
    "quantum": asdict(quantum),
    "amber_bin": str(amber_bin) if amber_bin else None,
    "resname": resname,
  }
  # Always revisit child stages: users may have independently rerun a child
  # after a successful run. Each child owns its own validated completion cache.
  with Stage(output, "run", molecule.to_dict(), settings, resume, reuse_complete=False) as stage:
    opt_dir, resp_dir, export_dir = (stage.directory / name for name in ("opt", "resp", "export"))
    geometry = optimize(molecule, opt_dir, optimization, quantum, resume=_started(opt_dir), publish=False)
    charges = calculate_charges(
      Molecule.from_dict(geometry["molecule"]),
      resp_dir,
      resp,
      quantum,
      resume=_started(resp_dir),
      publish=False,
    )
    parameters = export_parameters(
      charges, export_dir, amber_bin, resume=_started(export_dir), resname=resname, publish=False
    )
    files = _collect_deliverables(parameters, export_dir, stage.directory)
    result = stage.complete(
      {
        "molecule": geometry["molecule"],
        "charges": parameters["charges"],
        "files": files,
      },
      deliverables=tuple(files),
      binaries=parameters.get("binaries"),
    )
  if not keep_tmp:
    try:
      shutil.rmtree(stage.directory)
    except OSError as exc:
      # The deliverables are already published; a leftover tmp/ is not a failure of the run.
      print(f"pyligprep: warning: could not remove {stage.directory}: {exc}", file=sys.stderr)
  return result


def _started(directory: Path) -> bool:
  return (directory / STATE).exists()


def _collect_deliverables(exported: dict, export_dir: Path, directory: Path) -> dict[str, str]:
  """Copy MOL2/PREP/FRCMOD from the export stage into the run's working directory, checking their hashes."""
  files = {}
  for name in exported["deliverables"]:
    shutil.copyfile(export_dir / name, directory / name)
    files[name] = file_hash(directory / name)
    if files[name] != exported["files"][name]:
      raise CalculationError(f"Export artifact changed while copying: {export_dir / name}")
  return files
