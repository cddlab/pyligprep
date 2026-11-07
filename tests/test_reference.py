import os
import subprocess

import numpy as np
import pytest

from pyligprep.chemistry import load_molecule
from pyligprep.esp import BOHR, surface_points, write_esp
from pyligprep.resp import fit_resp, stage_constraints, write_resp_input


@pytest.mark.reference
def test_original_pyresp_two_stage_agreement(ethanol_pdb, tmp_path, amber_bin):
  """The original scripts are reference oracles only, never runtime dependencies."""
  import sys

  from scipy.spatial.distance import cdist

  mol = load_molecule(ethanol_pdb, 0, 1)
  coords = mol.coords / BOHR
  points = surface_points(coords, mol.numbers)
  design = 1 / cdist(points, coords)
  rng = np.random.default_rng(76)
  truth = rng.normal(size=len(mol.atoms))
  truth -= truth.mean()
  potential = design @ truth + rng.normal(scale=1e-4, size=len(points))
  write_esp(tmp_path / "esp.dat", mol, coords, points, potential)
  # Compare using exactly the same rounded values consumed by original PyRESP.
  lines = (tmp_path / "esp.dat").read_text().splitlines()
  coords = np.array([list(map(float, row.split()[:3])) for row in lines[1 : 1 + len(mol.atoms)]])
  esp = np.array([list(map(float, row.split())) for row in lines[1 + len(mol.atoms) :]])
  design = 1 / cdist(esp[:, 1:], coords)
  constraints = stage_constraints(mol)
  previous = None
  for stage, weight in ((1, 0.0005), (2, 0.001)):
    write_resp_input(
      tmp_path / f"stage{stage}.in",
      mol,
      constraints[f"stage{stage}"],
      weight,
      stage,
    )
    arguments = [
      sys.executable,
      str(amber_bin / "py_resp.py"),
      "-i",
      f"stage{stage}.in",
      "-o",
      f"stage{stage}.out",
      "-e",
      "esp.dat",
      "-t",
      f"stage{stage}.qout",
    ]
    if stage == 2:
      arguments += ["-q", "stage1.qout"]
    execution = subprocess.run(arguments, cwd=tmp_path, capture_output=True, text=True)
    assert execution.returncode == 0, execution.stdout + execution.stderr
    rows = (tmp_path / f"stage{stage}.qout").read_text().split("%FLAG ATOM CHRG:")[1].splitlines()[2:]
    actual = np.array([float(row.split()[-1]) for row in rows if row.strip()])
    fit = fit_resp(
      design,
      esp[:, 0],
      mol.numbers,
      mol.charge,
      constraints[f"stage{stage}"],
      initial=previous,
      qwt=weight,
    )
    np.testing.assert_allclose(fit.charges, actual, atol=6e-6, rtol=0)
    # Use the exact printed stage-one charges read by the reference program.
    previous = actual


@pytest.mark.reference
def test_original_generator_constraints(ethanol_pdb, tmp_path, amber_bin):
  mol = load_molecule(ethanol_pdb, 0, 1)
  coords = mol.coords / BOHR
  points = surface_points(coords, mol.numbers)
  write_esp(tmp_path / "esp.dat", mol, coords, points, np.zeros(len(points)))
  env = {
    **os.environ,
    "PYTHONPATH": os.pathsep.join(str(p) for p in (amber_bin.parent / "lib").glob("python*/site-packages")),
  }
  result = subprocess.run(
    [
      str(amber_bin / "pyresp_gen.py"),
      "-i",
      "esp.dat",
      "-p",
      "chg",
      "-q",
      "0",
      "-f1",
      "first.in",
      "-f2",
      "second.in",
    ],
    cwd=tmp_path,
    env=env,
    capture_output=True,
    text=True,
  )
  assert result.returncode == 0, result.stderr
  expected = stage_constraints(mol)
  for name, stage in (("first.in", "stage1"), ("second.in", "stage2")):
    rows = (tmp_path / name).read_text().split("&end")[1].strip().splitlines()
    ivary = [int(row.split()[1]) for row in rows[3 : 3 + len(mol.atoms)]]
    assert ivary == expected[stage]
