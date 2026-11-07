"""Geometry optimization with recoverable, independently validated attempts."""

import shutil
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from pyligprep.artifacts import Stage
from pyligprep.chemistry import write_pdb
from pyligprep.errors import CalculationError, InputError
from pyligprep.files import digest, file_hash, read_json, write_json, write_xyz
from pyligprep.model import Molecule
from pyligprep.quantum import QuantumOptions, build_molecule, choose_device, make_method, run_scf


@dataclass(frozen=True)
class OptimizationOptions:
  method: str = "B3LYP-D3BJ"
  basis: str = "6-31+G(d,p)"
  optimizer: str = "berny"
  max_steps: int = 200
  scf_type: str = "auto"
  constraints: str | None = None


def optimizer_parameters(options: OptimizationOptions, directory: Path):
  if options.max_steps < 1:
    raise InputError("max-steps must be positive")
  if options.optimizer == "berny":
    if options.constraints:
      raise InputError("Constraints require --optimizer geometric")
    from pyscf.geomopt import berny_solver

    return berny_solver, {
      "gradientmax": 1.5e-5,
      "gradientrms": 1e-5,
      "stepmax": 6e-5,
      "steprms": 4e-5,
    }
  if options.optimizer == "geometric":
    from pyscf.geomopt import geometric_solver

    parameters = {
      "convergence_set": "GAU_TIGHT",
      "hessian": True,
      "reset": False,
      "trust": 0.05,
      "tmax": 0.10,
    }
    if options.constraints:
      path = directory / "constraints.txt"
      path.write_text(options.constraints, encoding="utf-8")
      parameters.update(constraints=str(path), enforce=0.1)
    return geometric_solver, parameters
  raise InputError(f"Unknown optimizer: {options.optimizer}")


def _starting_geometry(stage: Stage, molecule: Molecule, options):
  if stage.resumed and (stage.directory / "last.json").exists():
    saved = read_json(stage.directory / "last.json")
    restored = Molecule.from_dict(saved["molecule"])
    # Only coordinates are allowed to change during optimization.
    if digest(restored.with_coords(molecule.coords).to_dict()) != digest(molecule.to_dict()):
      raise InputError("Restart atom identity differs")
    return restored
  return molecule


def _restart_density(stage, mol, options):
  if not stage.resumed or stage.prior_settings is None:
    return None
  previous_settings = stage.prior_settings["optimization"]
  if any(previous_settings[key] != getattr(options, key) for key in ("method", "basis")):
    print("Method/basis changed: reusing coordinates with a fresh SCF")
    return None
  path = stage.directory / "restart.chk"
  if not path.exists():
    path = stage.directory / "scf.chk"
  if not path.exists():
    return None
  from pyligprep.checkpoint import load_restart

  target = options.scf_type
  if target == "auto":
    target = "uks" if mol.spin else "rks"
  return load_restart(mol, path, target)


def optimize(
  molecule: Molecule,
  output: Path,
  options=OptimizationOptions(),
  quantum=QuantumOptions(),
  resume=False,
  publish=True,
) -> dict:
  settings = {"optimization": asdict(options), "quantum": asdict(quantum)}
  with Stage(output, "opt", molecule.to_dict(), settings, resume, publish=publish) as stage:
    if stage.cached is not None:
      return stage.cached
    solver, parameters = optimizer_parameters(options, stage.directory)
    device = choose_device(quantum.device)
    current = _starting_geometry(stage, molecule, options)
    mol = build_molecule(current, options.basis, quantum, stage.directory / "qm.log")
    with mol.stdout:
      dm0 = _restart_density(stage, mol, options)
      current = molecule.with_coords(mol.atom_coords(unit="Angstrom"))
      write_xyz(stage.directory / "initial.xyz", current)
      if options.constraints:
        from geometric.molecule import Molecule as GeometricMolecule
        from geometric.prepare import parse_constraints

        geometry = GeometricMolecule()
        geometry.elem = [a.element for a in current.atoms]
        geometry.xyzs = [current.coords]
        if not parse_constraints(geometry, options.constraints)[0]:
          raise InputError("No valid constraints supplied")
      mf = make_method(
        mol,
        options.method,
        device,
        stage.directory / "scf.chk",
        quantum,
        options.scf_type,
      )
      run_scf(mf, dm0, device)
      last = {"energy": float(mf.e_tot), "evaluation": 0}

      def save_step(env):
        converged = bool(env["g_scanner"].converged)
        coords = env["mol"].atom_coords(unit="Angstrom")
        energy = float(env["energy"])
        if not converged or not np.isfinite(energy):
          raise CalculationError("SCF failed during optimization; last valid geometry retained")
        snapshot = molecule.with_coords(coords)
        last.update(energy=energy, evaluation=last["evaluation"] + 1)
        write_xyz(stage.directory / "trajectory.xyz", snapshot, str(last), append=True)
        _save_restart(stage.directory, snapshot, last)

      _save_restart(stage.directory, current, last)
      gradient = mf.nuc_grad_method()
      if options.method.upper() != "HF":
        gradient.grid_response = True
      converged, final_mol = solver.kernel(
        gradient.as_scanner(),
        maxsteps=options.max_steps,
        assert_convergence=True,
        callback=save_step,
        **parameters,
      )
      if not converged:
        raise CalculationError(f"Geometry did not converge in {options.max_steps} steps; resume from last.xyz")
      final = molecule.with_coords(final_mol.atom_coords(unit="Angstrom"))
    write_xyz(stage.directory / "optimized.xyz", final, "converged")
    write_pdb(stage.directory / "optimized.pdb", final)
    deliverables = ("optimized.pdb", "optimized.xyz")
    return stage.complete(
      {
        "molecule": final.to_dict(),
        "converged": True,
        "device": device,
        **last,
        "files": {name: file_hash(stage.directory / name) for name in deliverables},
      },
      deliverables=deliverables,
      discard=("scf.chk", "restart.chk"),
    )


def _save_restart(directory, molecule, info):
  checkpoint = directory / "scf.chk"
  if checkpoint.exists():
    shutil.copyfile(checkpoint, directory / "restart.chk")
  write_xyz(directory / "last.xyz", molecule, str(info))
  write_json(directory / "last.json", {"molecule": molecule.to_dict(), **info})
