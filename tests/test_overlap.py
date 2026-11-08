from dataclasses import asdict
from unittest.mock import patch

import numpy as np
import pytest

from pyligprep import cli
from pyligprep.artifacts import Stage, completed_result
from pyligprep.chemistry import load_molecule
from pyligprep.errors import CalculationError, InputError
from pyligprep.optimization import OptimizationOptions, optimize
from pyligprep.quantum import QuantumOptions, make_method


@pytest.mark.parametrize("value", [0, -1e-8, float("nan"), float("inf"), True, "1e-8"])
def test_invalid_threshold_is_rejected_before_calculation(value):
  with pytest.raises(InputError, match="opt-overlap-threshold"):
    OptimizationOptions(opt_overlap_threshold=value)


@pytest.mark.parametrize(
  "command,target", [("opt", "pyligprep.optimization.optimize"), ("run", "pyligprep.pipeline.run_pipeline")]
)
def test_cli_passes_optimization_threshold_and_new_scf_name(command, target, ethanol_pdb, tmp_path):
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
          "--opt-overlap-threshold",
          "1e-8",
          "--scf-maxcycle",
          "321",
        ]
      )
      == 0
    )
  assert call.call_args.args[2].opt_overlap_threshold == 1e-8
  quantum = call.call_args.args[3 if command == "opt" else 4]
  assert quantum.scf_max_cycle == 321
  assert "opt_overlap_threshold" not in asdict(quantum)


@pytest.mark.parametrize("value", ["0", "-0.1", "nan", "inf"])
def test_invalid_cli_threshold_does_not_create_output(value, ethanol_pdb, tmp_path):
  output = tmp_path / "opt"
  with patch("pyligprep.optimization.choose_device", side_effect=AssertionError("No calculation expected")):
    assert (
      cli.main(
        [
          "opt",
          "-i",
          str(ethanol_pdb),
          "-o",
          str(output),
          "--charge",
          "0",
          "--multiplicity",
          "1",
          f"--opt-overlap-threshold={value}",
        ]
      )
      == 2
    )
  assert not output.exists()


def test_resp_rejects_optimization_only_option():
  with pytest.raises(SystemExit):
    cli.make_parser().parse_args(["resp", "--resume", "unused", "--opt-overlap-threshold", "1e-8"])


@pytest.mark.parametrize(
  "command,target", [("opt", "pyligprep.optimization.optimize"), ("run", "pyligprep.pipeline.run_pipeline")]
)
@pytest.mark.parametrize(
  "saved,override,expected", [(None, None, None), (None, 1e-8, 1e-8), (1e-8, None, 1e-8), (1e-8, 1e-7, 1e-7)]
)
def test_resume_inherits_or_overrides_threshold(command, target, saved, override, expected, ethanol_pdb, tmp_path):
  molecule = load_molecule(ethanol_pdb, 0, 1)
  settings = {
    "optimization": OptimizationOptions(opt_overlap_threshold=saved).to_dict(),
    "quantum": asdict(QuantumOptions()),
  }
  output = tmp_path / command
  with Stage(output, command, molecule.to_dict(), settings) as stage:
    stage.complete({"molecule": molecule.to_dict()})
  args = [command, "--resume", str(output)]
  if override is not None:
    args.extend(["--opt-overlap-threshold", str(override)])
  with patch(target) as call:
    assert cli.main(args) == 0
  assert call.call_args.args[2].opt_overlap_threshold == expected


def test_threshold_controls_cache_without_invalidating_omitted_default(ethanol_pdb, tmp_path):
  molecule = load_molecule(ethanol_pdb, 0, 1)
  options = OptimizationOptions()
  assert "opt_overlap_threshold" not in options.to_dict()
  quantum = QuantumOptions(device="cpu")
  settings = {"optimization": options.to_dict(), "quantum": asdict(quantum)}
  output = tmp_path / "opt"
  with Stage(output, "opt", molecule.to_dict(), settings) as stage:
    stage.complete({"molecule": molecule.to_dict(), "converged": True})
  expected = completed_result(output, "opt")
  with patch("pyligprep.optimization.choose_device", side_effect=AssertionError("Must use cache")):
    assert optimize(molecule, output, quantum=quantum, resume=True) == expected
  changed = {**settings, "optimization": OptimizationOptions(opt_overlap_threshold=1e-8).to_dict()}
  with Stage(output, "opt", molecule.to_dict(), changed, resume=True) as stage:
    assert stage.cached is None
    stage.complete({"converged": True})
  with Stage(output, "opt", molecule.to_dict(), changed, resume=True) as stage:
    assert stage.cached is not None


@pytest.fixture
def h2():
  from pyscf import gto, lib

  lib.num_threads(1)
  return gto.M(atom="H 0 0 0; H 0 0 0.74", basis="sto-3g", verbose=0)


@pytest.mark.parametrize("method,scf_type", [("HF", "rks"), ("HF", "uks"), ("PBE", "rks"), ("PBE", "uks")])
def test_threshold_is_instance_local_and_survives_scanner_reset(method, scf_type, h2, tmp_path):
  from pyscf.scf import hf

  backend_default = hf.overlap_zero_eigenvalue_threshold
  options = QuantumOptions(device="cpu")
  mf = make_method(h2, method, "cpu", tmp_path / "opt.chk", options, scf_type, opt_overlap_threshold=1e-8)
  other = make_method(h2, method, "cpu", tmp_path / "other.chk", options, scf_type, opt_overlap_threshold=1e-6)
  overlap = np.diag([5e-7, 1.0])
  for item, rank in [(mf, 2), (other, 1)]:
    scanner = item.nuc_grad_method().as_scanner().base
    scanner.reset(h2.set_geom_("H 0 0 0; H 0 0 0.75", inplace=False))
    for target in [scanner, getattr(scanner, "_scf", scanner)]:
      x = target.check_linear_dependency(overlap)
      assert x.shape == (2, rank)
      np.testing.assert_allclose(x.T @ overlap @ x, np.eye(rank), atol=1e-12)
  assert hf.overlap_zero_eigenvalue_threshold == backend_default
  resp = make_method(h2, "HF", "cpu", tmp_path / "resp.chk", options)
  assert not hasattr(resp, "opt_overlap_threshold")
  np.testing.assert_allclose(resp.check_linear_dependency(overlap), hf.check_linear_dependency(overlap))


def test_bad_overlap_or_too_few_orbitals_is_a_calculation_error(h2, tmp_path):
  mf = make_method(h2, "HF", "cpu", tmp_path / "scf.chk", QuantumOptions(), opt_overlap_threshold=1e-8)
  for overlap in [np.diag([0.0, 1e-10]), np.diag([np.nan, 1.0]), np.diag([-1.0, -2.0])]:
    with pytest.raises(CalculationError, match="overlap|orbital"):
      mf.check_linear_dependency(overlap)


def test_threshold_reaches_optimization_factory(ethanol_pdb, tmp_path):
  molecule = load_molecule(ethanol_pdb, 0, 1)
  with patch("pyligprep.optimization.make_method", side_effect=CalculationError("factory reached")) as factory:
    with pytest.raises(CalculationError, match="factory reached"):
      optimize(
        molecule, tmp_path / "opt", OptimizationOptions(opt_overlap_threshold=1e-8), QuantumOptions(device="cpu")
      )
  assert factory.call_args.kwargs["opt_overlap_threshold"] == 1e-8


@pytest.mark.parametrize("device", ["cpu", pytest.param("gpu", marks=pytest.mark.gpu)])
@pytest.mark.parametrize("scf_type", ["rks", "uks"])
def test_threshold_through_newton_and_gradient_scanner(device, scf_type, h2, tmp_path):
  if device == "gpu":
    import cupy as xp
    from gpu4pyscf.scf import hf
  else:
    import numpy as xp
    from pyscf.scf import hf

  default = hf.overlap_zero_eigenvalue_threshold
  options = QuantumOptions(device=device, threads=1)
  mf = make_method(h2, "PBE", device, tmp_path / "scf.chk", options, scf_type, opt_overlap_threshold=1e-8)
  scanner = mf.nuc_grad_method().as_scanner()
  overlap = xp.diag(xp.asarray([5e-7, 1.0]))
  for distance in [0.74, 0.75]:
    mol = h2.set_geom_(f"H 0 0 0; H 0 0 {distance}", inplace=False)
    energy, gradient = scanner(mol)
    assert scanner.converged and np.isfinite(energy)
    assert bool(xp.isfinite(xp.asarray(gradient)).all())
    x = scanner.base._scf.check_linear_dependency(overlap)
    assert isinstance(x, xp.ndarray) and x.shape == (2, 2)
    assert bool(xp.allclose(x.T @ overlap @ x, xp.eye(2), atol=1e-12))
  assert hf.overlap_zero_eigenvalue_threshold == default
  resp = make_method(h2, "HF", device, tmp_path / "resp.chk", options)
  assert not hasattr(resp, "opt_overlap_threshold")
  assert resp.check_linear_dependency(overlap).shape == (2, 1)


def test_lower_threshold_avoids_rank_change_at_backend_cutoff(h2, tmp_path):
  from pyscf.scf import hf

  mf = make_method(h2, "HF", "cpu", tmp_path / "scf.chk", QuantumOptions(), opt_overlap_threshold=1e-8)
  for value in [hf.overlap_zero_eigenvalue_threshold * 0.99, hf.overlap_zero_eigenvalue_threshold * 1.01]:
    assert mf.check_linear_dependency(np.diag([value, 1.0])).shape == (2, 2)
  # The specified cutoff still excludes zero, negative, and equal eigenvalues.
  for value in [-1e-7, 0.0, 1e-8]:
    assert mf.check_linear_dependency(np.diag([value, 1.0])).shape == (2, 1)


def test_unsupported_backend_fails_only_when_requested():
  from types import SimpleNamespace

  from pyligprep.errors import DependencyError
  from pyligprep.overlap import apply_overlap_threshold

  backend = SimpleNamespace()
  assert apply_overlap_threshold(backend, None) is backend
  with pytest.raises(DependencyError, match="check_linear_dependency"):
    apply_overlap_threshold(backend, 1e-8)
