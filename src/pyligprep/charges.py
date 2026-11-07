"""HF, ESP, and both RESP stages with independent substep caches."""

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from scipy.spatial.distance import cdist

from pyligprep.artifacts import Stage, versions
from pyligprep.errors import InputError
from pyligprep.esp import BOHR, RADII, SCALES, evaluate_esp, surface_points, write_esp
from pyligprep.files import digest, file_hash, read_json, write_json
from pyligprep.model import Molecule
from pyligprep.quantum import QuantumOptions, build_molecule, choose_device, make_method, run_scf
from pyligprep.resp import fit_resp, stage_constraints, validate_constraints, write_resp_input


@dataclass(frozen=True)
class RespOptions:
  density: float = 1.0
  qwt1: float = 0.0005
  qwt2: float = 0.001
  max_iterations: int = 42
  tolerance: float = 1e-6
  equivalence_depth: int = 3
  constraints: dict | None = None

  def validate(self):
    values = (self.density, self.qwt1, self.qwt2, self.tolerance)
    if (
      not np.isfinite(values).all()
      or self.density <= 0
      or min(self.qwt1, self.qwt2) < 0
      or self.tolerance <= 0
      or self.max_iterations < 1
      or not 1 <= self.equivalence_depth <= 5
    ):
      raise InputError("Invalid RESP density, weights, tolerance or iteration/depth limit")


def _esp_data(stage, molecule, options, quantum):
  electronic = {
    "molecule": molecule.to_dict(),
    "method": "HF",
    "basis": "6-31g*",
    "cart": True,
    "scf_conv_tol": 1e-10,
    "versions": versions(),
  }
  hf_key = digest(electronic)
  esp_key = digest({**electronic, "density": options.density, "radii": RADII, "scales": SCALES})
  if stage.reusable("esp", esp_key):
    with np.load(stage.directory / "esp.npz", allow_pickle=False) as data:
      return data["coords"], data["points"], data["potential"], "cached"
  stage.forget("esp")
  device = choose_device(quantum.device)
  mol = build_molecule(molecule, "6-31g*", quantum, stage.directory / "hf.log", cart=True)
  with mol.stdout:
    mf = make_method(mol, "HF", device, stage.directory / "hf.chk", quantum)
    if stage.reusable("hf", hf_key):
      from pyscf import scf

      saved, data = scf.chkfile.load_scf(str(stage.directory / "hf.chk"))
      from pyligprep.checkpoint import checkpoint_density, validate_checkpoint_molecule

      validate_checkpoint_molecule(mol, saved)
      if not np.allclose(mol.atom_coords(), saved.atom_coords(), atol=1e-10, rtol=0):
        raise InputError("Cached HF geometry differs")
      _, dm = checkpoint_density(saved, data)
      # A converged checkpoint is sufficient for ESP; no HF rerun.
      if device == "gpu":
        import cupy

        dm = cupy.asarray(dm)
      mf.make_rdm1 = lambda: dm
    else:
      stage.forget("hf")
      dm0 = None
      # A checkpoint left by an interrupted attempt on this molecule seeds the SCF.
      if stage.resumed and (stage.directory / "hf.chk").exists():
        from pyligprep.checkpoint import load_restart

        candidate = mol.copy()
        dm0 = load_restart(candidate, stage.directory / "hf.chk", "uks" if mol.spin else "rks")
        if not np.allclose(candidate.atom_coords(), mol.atom_coords(), atol=1e-10, rtol=0):
          raise InputError("HF restart geometry differs")
      run_scf(mf, dm0, device)
      stage.mark(
        "hf",
        hf_key,
        ["hf.chk"],
        energy=float(mf.e_tot),
        device=device,
      )
    coords = mol.atom_coords()
    points = surface_points(coords, molecule.numbers, options.density)
    potential = evaluate_esp(mf, points, device)
    if not np.isfinite(potential).all():
      raise InputError("Nonfinite ESP values")
    np.savez_compressed(
      stage.directory / "esp.npz",
      coords=coords,
      points=points,
      potential=potential,
    )
    write_esp(stage.directory / "esp.dat", molecule, coords, points, potential)
    stage.mark(
      "esp",
      esp_key,
      ["esp.npz", "esp.dat"],
      units="coordinates=bohr; potential=hartree/e",
      device=device,
      grid="GPU4PySCF-1.9.1-GAMESS",
    )
  return coords, points, potential, device


def calculate_charges(
  molecule: Molecule,
  output: Path,
  options=RespOptions(),
  quantum=QuantumOptions(),
  resume=False,
  publish=True,
) -> dict:
  options.validate()
  quantum.validate()
  settings = {
    "resp": asdict(options),
    "quantum": asdict(quantum),
    "esp_method": "HF/6-31G*",
    "cart": True,
    "gas_phase": True,
  }
  with Stage(output, "resp", molecule.to_dict(), settings, resume, publish=publish) as stage:
    if stage.cached is not None:
      return stage.cached
    constraints = options.constraints or stage_constraints(molecule, options.equivalence_depth)
    validate_constraints(molecule, constraints)
    write_json(stage.directory / "constraints.json", constraints)
    coords, points, potential, device = _esp_data(stage, molecule, options, quantum)
    if not np.allclose(coords * BOHR, molecule.coords, atol=1e-8, rtol=0):
      raise InputError("Cached ESP coordinates do not match input")
    design = 1.0 / cdist(points, coords)
    initial = None
    summaries = []
    for number, qwt in ((1, options.qwt1), (2, options.qwt2)):
      name = f"stage{number}"
      key = digest(
        {
          "esp": file_hash(stage.directory / "esp.npz"),
          "constraints": constraints[name],
          "qwt": qwt,
          "initial": initial,
          "tolerance": options.tolerance,
          "max_iterations": options.max_iterations,
        }
      )
      write_resp_input(stage.directory / f"{name}.in", molecule, constraints[name], qwt, number)
      if not stage.reusable(name, key):
        stage.forget(name)
        fit = fit_resp(
          design,
          potential,
          molecule.numbers,
          molecule.charge,
          constraints[name],
          initial=initial,
          qwt=qwt,
          max_iterations=options.max_iterations,
          tolerance=options.tolerance,
        )
        write_json(
          stage.directory / f"{name}-charges.json",
          {
            "charges": fit.charges.tolist(),
            "iterations": fit.iterations,
            "rms_esp_error": fit.rms_esp_error,
          },
        )
        stage.mark(name, key, [f"{name}-charges.json"])
      summary = read_json(stage.directory / f"{name}-charges.json")
      initial = summary["charges"]
      summaries.append(summary)
    write_json(
      stage.directory / "charges.json",
      {"atoms": [a.name for a in molecule.atoms], "charges": initial, "stages": summaries},
    )
    return stage.complete(
      {
        "molecule": molecule.to_dict(),
        "charges": initial,
        "device": device,
        "stages": summaries,
        "constraints": constraints,
        "files": {
          f: file_hash(stage.directory / f)
          for f in (
            "esp.npz",
            "esp.dat",
            "constraints.json",
            "stage1-charges.json",
            "stage2-charges.json",
            "charges.json",
          )
        },
      },
      deliverables=("charges.json",),
      discard=("hf.chk",),
    )
