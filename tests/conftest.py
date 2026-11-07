import os
import shutil
from pathlib import Path

import pytest
from rdkit import Chem
from rdkit.Chem import rdDistGeom


@pytest.fixture
def amber_bin():
  explicit = os.environ.get("PYLIGPREP_AMBER_BIN")
  if explicit:
    return Path(explicit)
  if os.environ.get("AMBERHOME"):
    return Path(os.environ["AMBERHOME"]) / "bin"
  executable = shutil.which("antechamber")
  if executable:
    return Path(executable).resolve().parent
  pytest.skip("Load AmberTools 26 or set PYLIGPREP_AMBER_BIN")


@pytest.fixture
def ethanol_pdb(tmp_path):
  mol = Chem.AddHs(Chem.MolFromSmiles("CCO"))
  assert rdDistGeom.EmbedMolecule(mol, randomSeed=7) == 0
  path = tmp_path / "ethanol.pdb"
  Chem.MolToPDBFile(mol, str(path))
  return path


@pytest.fixture
def fake_export():
  """Stand-in for export_parameters that writes placeholder deliverables without AmberTools."""
  from pyligprep.files import file_hash

  def export(result, output, amber_bin=None, resume=False, resname=None, publish=True):
    assert not publish, "run must keep the export stage inside tmp/"
    output.mkdir(parents=True, exist_ok=True)
    prefix = resname or "ligand"
    deliverables = [f"{prefix}.{ext}" for ext in ("mol2", "prep", "frcmod")]
    for name in [*deliverables, "atom-map.json"]:
      (output / name).write_text(f"{name}\n")
    return {
      **result,
      "files": {n: file_hash(output / n) for n in [*deliverables, "atom-map.json"]},
      "deliverables": deliverables,
      "binaries": {"ambertools": "AmberToolsTest"},
    }

  return export


def pytest_collection_modifyitems(config, items):
  for item in items:
    for marker, variable in (
      ("gpu", "PYLIGPREP_TEST_GPU"),
      ("amber", "PYLIGPREP_TEST_AMBER"),
      ("reference", "PYLIGPREP_TEST_REFERENCE"),
    ):
      if marker in item.keywords and os.environ.get(variable) != "1":
        item.add_marker(pytest.mark.skip(reason=f"Set {variable}=1"))
