import tomllib
from pathlib import Path

import pytest

import pyligprep
from pyligprep import cli

PYPROJECT = Path(__file__).resolve().parents[1] / "pyproject.toml"
# The command itself plus every subcommand registered in cli.make_parser().
COMMANDS = ([], ["hadd"], ["doctor"], ["check"], ["opt"], ["resp"], ["export"], ["run"])


def _project_version() -> str:
  with PYPROJECT.open("rb") as handle:
    return tomllib.load(handle)["project"]["version"]


def _run(argv, capsys):
  with pytest.raises(SystemExit) as exit_info:
    cli.main(argv)
  assert exit_info.value.code == 0
  return capsys.readouterr().out


def test_version_comes_from_pyproject():
  """Run `uv sync` after bumping project.version so the installed metadata follows."""
  assert pyligprep.__version__ == _project_version()


@pytest.mark.parametrize("command", COMMANDS, ids=lambda c: " ".join(c) or "pyligprep")
def test_help_starts_with_version_line(command, capsys):
  lines = _run([*command, "--help"], capsys).splitlines()
  assert lines[0] == f"pyligprep {_project_version()}"
  assert lines[2].startswith("usage: pyligprep")


@pytest.mark.parametrize("command", COMMANDS, ids=lambda c: " ".join(c) or "pyligprep")
@pytest.mark.parametrize("flag", ["-V", "--version"])
def test_version_flag_prints_only_the_version(command, flag, capsys):
  assert _run([*command, flag], capsys) == f"pyligprep {_project_version()}\n"


def test_version_flag_skips_required_arguments(capsys):
  """`run -V` must not demand -i/--resume."""
  assert _run(["run", "-V"], capsys).startswith("pyligprep ")
