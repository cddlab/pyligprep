"""Ligand optimization and charge preparation."""

import tomllib
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path


def _project_version() -> str:
  """Return `project.version` of pyproject.toml, the single source of the version number.

  The build backend copies it into the installed package metadata, which is read first. A source tree that was
  never installed (e.g. PYTHONPATH=src) falls back to the pyproject.toml next to `src/`.
  """
  try:
    return version("pyligprep")
  except PackageNotFoundError:
    pass
  try:
    with (Path(__file__).resolve().parents[2] / "pyproject.toml").open("rb") as handle:
      project = tomllib.load(handle)["project"]
    if project["name"] == "pyligprep":
      return str(project["version"])
  except (OSError, KeyError, tomllib.TOMLDecodeError):
    pass
  return "0+unknown"


__version__ = _project_version()
