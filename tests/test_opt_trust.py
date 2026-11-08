from dataclasses import asdict
from unittest.mock import patch

import pytest

from pyligprep import cli
from pyligprep.artifacts import Stage, completed_result
from pyligprep.chemistry import load_molecule
from pyligprep.errors import InputError
from pyligprep.optimization import OptimizationOptions, optimize, optimizer_parameters
from pyligprep.quantum import QuantumOptions


def test_omitting_opt_max_step_preserves_backend_defaults(tmp_path):
  from geometric.params import OptParams

  _, berny = optimizer_parameters(OptimizationOptions(), tmp_path)
  assert "trust" not in berny
  _, geometric = optimizer_parameters(OptimizationOptions(optimizer="geometric"), tmp_path)
  assert geometric["trust"] == 0.05
  assert geometric["tmax"] == 0.10
  assert OptParams(**{**geometric, "hessian": None}).trust == 0.05


@pytest.mark.parametrize("value", [1, 10, 30, 100])
def test_opt_max_step_reaches_berny_in_gaussian_units(value, tmp_path):
  from berny import Berny, geomlib

  _, parameters = optimizer_parameters(OptimizationOptions(opt_max_step=value), tmp_path)
  geometry = geomlib.Geometry(["O", "H", "H"], [[0, 0, 0], [0, -0.757, 0.587], [0, 0.757, 0.587]])
  optimizer = Berny(geometry, **parameters)
  assert optimizer.trust == pytest.approx(value * 0.01)
  # Convergence thresholds do not control the initial step size.
  assert parameters["stepmax"] == 6e-5
  assert parameters["steprms"] == 4e-5


@pytest.mark.parametrize("value", [1, 10, 30, 100])
def test_opt_max_step_reaches_geometric_without_clipping(value, tmp_path):
  from geometric.params import OptParams
  from pyscf import lib

  _, parameters = optimizer_parameters(OptimizationOptions(optimizer="geometric", opt_max_step=value), tmp_path)
  expected = value * 0.01 * lib.param.BOHR
  actual = OptParams(**{**parameters, "hessian": None})
  assert actual.trust == pytest.approx(expected)
  assert actual.tmax == pytest.approx(max(0.10, expected))
  assert parameters["convergence_set"] == "GAU_TIGHT"


@pytest.mark.parametrize("value", [0, -1, 0.1, float("nan"), float("inf"), True, "10"])
def test_opt_max_step_requires_a_positive_integer_in_python(value):
  with pytest.raises(InputError, match="opt-maxstep must be a positive integer"):
    OptimizationOptions(opt_max_step=value)


@pytest.mark.parametrize(
  "command,target", [("opt", "pyligprep.optimization.optimize"), ("run", "pyligprep.pipeline.run_pipeline")]
)
def test_opt_max_step_cli_dispatch(command, target, ethanol_pdb, tmp_path):
  with patch(target) as call:
    assert (
      cli.main(
        [
          command,
          "-i",
          str(ethanol_pdb),
          "-o",
          str(tmp_path / command),
          "--charge",
          "0",
          "--multiplicity",
          "1",
          "--opt-maxstep",
          "10",
        ]
      )
      == 0
    )
  assert call.call_args.args[2].opt_max_step == 10


@pytest.mark.parametrize("command", ["opt", "run"])
@pytest.mark.parametrize("value", ["0", "-1"])
def test_invalid_opt_max_step_stops_before_calculation(command, value, ethanol_pdb, tmp_path, capsys):
  output = tmp_path / command
  with patch("pyligprep.quantum.gpu_probe", side_effect=AssertionError("GPU must not be initialized")):
    assert (
      cli.main(
        [
          command,
          "-i",
          str(ethanol_pdb),
          "-o",
          str(output),
          "--charge",
          "0",
          "--multiplicity",
          "1",
          "--opt-maxstep",
          value,
        ]
      )
      == 2
    )
  assert "opt-maxstep must be a positive integer" in capsys.readouterr().err
  assert not output.exists()


@pytest.mark.parametrize("value", ["0.1", "nan", "inf"])
def test_fractional_and_nonfinite_cli_values_are_rejected(value):
  with pytest.raises(SystemExit) as error:
    cli.make_parser().parse_args(["opt", "--resume", "unused", "--opt-maxstep", value])
  assert error.value.code == 2


@pytest.mark.parametrize(
  "command,target", [("opt", "pyligprep.optimization.optimize"), ("run", "pyligprep.pipeline.run_pipeline")]
)
@pytest.mark.parametrize(
  "saved_value,override,expected", [(None, None, None), (None, 10, 10), (10, None, 10), (10, 5, 5)]
)
def test_opt_max_step_resume_preserves_overrides_and_omitted_defaults(
  command, target, saved_value, override, expected, ethanol_pdb, tmp_path
):
  molecule = load_molecule(ethanol_pdb, 0, 1)
  options = OptimizationOptions().to_dict()
  options.pop("opt_max_step", None)
  if saved_value is not None:
    options["opt_max_step"] = saved_value
  settings = {"optimization": options, "quantum": asdict(QuantumOptions())}
  output = tmp_path / command
  with Stage(output, command, molecule.to_dict(), settings) as stage:
    stage.complete({"molecule": molecule.to_dict()})
  args = [command, "--resume", str(output)]
  if override is not None:
    args.extend(["--opt-maxstep", str(override)])
  with patch(target) as call:
    assert cli.main(args) == 0
  assert call.call_args.args[2].opt_max_step == expected


def test_changed_opt_max_step_invalidates_completed_optimization_cache(tmp_path):
  output = tmp_path / "opt"
  settings = {"optimization": asdict(OptimizationOptions(opt_max_step=10))}
  with Stage(output, "opt", {"input": 1}, settings) as stage:
    stage.complete({"converged": True})
  with Stage(output, "opt", {"input": 1}, settings, resume=True) as stage:
    assert stage.cached is not None
  changed = {"optimization": asdict(OptimizationOptions(opt_max_step=5))}
  with Stage(output, "opt", {"input": 1}, changed, resume=True) as stage:
    assert stage.cached is None
    assert stage.prior_settings == settings


def test_omitting_opt_max_step_reuses_completed_optimization_with_default_radius(ethanol_pdb, tmp_path):
  molecule = load_molecule(ethanol_pdb, 0, 1)
  options = OptimizationOptions().to_dict()
  options.pop("opt_max_step", None)
  quantum = QuantumOptions(device="cpu")
  settings = {"optimization": options, "quantum": asdict(quantum)}
  output = tmp_path / "opt"
  result = {"molecule": molecule.to_dict(), "converged": True}
  with Stage(output, "opt", molecule.to_dict(), settings) as stage:
    stage.complete(result)
  expected = completed_result(output, "opt")
  with patch("pyligprep.optimization.choose_device", side_effect=AssertionError("Cache must avoid recomputation")):
    assert optimize(molecule, output, quantum=quantum, resume=True) == expected
