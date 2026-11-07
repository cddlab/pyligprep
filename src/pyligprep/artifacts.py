"""Locked, resumable stages: deliverables in the output directory, working files in tmp/."""

import fcntl
import platform
import shutil
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from pyligprep.errors import InputError
from pyligprep.files import digest, file_hash, read_json, write_json

SCHEMA = 2
TMP = "tmp"
STATE = "state.json"
LOCK = ".lock"
VERSIONS = "versions.json"


def versions() -> dict:
  result = {
    "python": platform.python_version(),
    "platform": f"{platform.system()}-{platform.machine()}",
  }
  for name in (
    "pyligprep",
    "numpy",
    "scipy",
    "pyscf",
    "pyscf-dispersion",
    "dftd3",
    "rdkit",
    "gpu4pyscf-cuda12x",
    "cupy-cuda12x",
    "cutensor-cu12",
    "pyberny",
    "geometric",
  ):
    try:
      result[name] = version(name)
    except PackageNotFoundError:
      pass
  return result


def stage_directory(output: Path, publish: bool) -> Path:
  """Working directory of a stage: `<output>/tmp` when it publishes into `output`, else `output` itself."""
  return output / TMP if publish else output


def check_resumable(output: Path) -> None:
  """Explain why an output directory cannot be resumed or reused."""
  if (output / TMP / STATE).exists():
    return
  if (output / "manifest.json").exists():
    raise InputError(f"{output} uses the old output format (manifest.json/attempts); rerun into a new output directory")
  if (output / VERSIONS).exists():
    raise InputError(
      f"{output} is complete and its tmp/ was removed; rerun into a new output directory "
      "(use run --keep-tmp to keep tmp/ for later reruns)"
    )
  raise InputError(f"No stage state to resume in {output}")


def load_state(output: Path, stage: str | None = None) -> dict:
  """Read the state of the stage whose deliverables are in `output`."""
  check_resumable(output)
  return _read_state(output / TMP, stage)


def _read_state(directory: Path, stage: str | None = None) -> dict:
  state = read_json(directory / STATE)
  if state.get("schema") != SCHEMA:
    raise InputError(f"Unsupported stage state in {directory}")
  if stage is not None and state.get("stage") != stage:
    raise InputError(f"{directory} is not a {stage} stage")
  return state


def completed_result(output: Path, stage: str) -> dict:
  """Verified result of a completed stage, given its output directory (or a run's tmp/<stage>)."""
  if (output / STATE).exists():
    return _verified_result(output, stage)
  check_resumable(output)
  return _verified_result(output / TMP, stage)


def _verified_result(directory: Path, stage: str) -> dict:
  state = _read_state(directory, stage)
  if state.get("status") != "complete":
    raise InputError(f"{directory} is not a completed {stage} stage")
  result = state["result"]
  for name, expected in result.get("files", {}).items():
    path = directory / name
    if Path(name).name != name or path.is_symlink() or not path.is_file() or file_hash(path) != expected:
      raise InputError(f"Output artifact was modified: {path}")
  return result


def _clear(directory: Path) -> None:
  for entry in directory.iterdir():
    if entry.name == LOCK:
      continue
    if entry.is_dir() and not entry.is_symlink():
      shutil.rmtree(entry)
    else:
      entry.unlink()


class Stage:
  """Hold an exclusive lock on a stage's working directory until it finishes or records a failure.

  With `publish=True` the deliverables go to `output` and everything else to `output/tmp`;
  with `publish=False` (child stages of `run`) `output` is the working directory itself.
  """

  def __init__(
    self,
    output: Path,
    name: str,
    molecule: dict,
    settings: dict,
    resume=False,
    reuse_complete=True,
    publish=True,
  ):
    output = output.resolve()
    self.public = output if publish else None
    self.directory = stage_directory(output, publish)
    self.name = name
    self.molecule = molecule
    self.settings = settings
    self.resume = resume
    self.fingerprint = digest({"molecule": molecule, "settings": settings, "versions": versions()})
    self.reuse_complete = reuse_complete
    self.resumed = False
    self.prior_settings: dict | None = None
    self.cached = None
    self.state: dict = {}
    self.lock = None

  def __enter__(self):
    if self.public is not None:
      if self.resume:
        check_resumable(self.public)
      self.public.mkdir(parents=True, exist_ok=self.resume)
    self.directory.mkdir(parents=True, exist_ok=True)
    self.lock = (self.directory / LOCK).open("a")
    try:
      try:
        fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
      except BlockingIOError as exc:
        raise InputError(f"Another process holds the stage lock: {self.directory}") from exc
      self._begin()
    except BaseException:
      self.lock.close()
      raise
    return self

  def _begin(self):
    old = _read_state(self.directory) if (self.directory / STATE).exists() else None
    if old is None and self.resume:
      raise InputError(f"No stage state to resume in {self.directory}")
    steps, published = {}, {}
    if old is not None:
      if old.get("stage") != self.name:
        raise InputError("Resume stage type differs")
      same_molecule = digest(old["molecule"]) == digest(self.molecule)
      if not same_molecule and self.name in ("opt", "run"):
        raise InputError("Resume input differs; use a new output directory")
      published = old.get("published", {})
      if self.reuse_complete and old["status"] == "complete" and old["fingerprint"] == self.fingerprint:
        self.state = old
        self.cached = _verified_result(self.directory, self.name)
        self._deliver(self.cached)
        return
      if same_molecule:
        self.resumed = True
        self.prior_settings = old["settings"]
        steps = old.get("steps", {})
      else:
        # Working files of another molecule must never seed this one (e.g. an HF checkpoint).
        _clear(self.directory)
    self.state = {
      "schema": SCHEMA,
      "stage": self.name,
      "status": "running",
      "molecule": self.molecule,
      "settings": self.settings,
      "fingerprint": self.fingerprint,
      "versions": versions(),
      "steps": steps,
      "published": published,
    }
    self._save()

  def _save(self):
    write_json(self.directory / STATE, self.state)

  def reusable(self, name: str, fingerprint: str) -> bool:
    """Whether substep `name` already completed with `fingerprint` and its files are intact."""
    step = self.state["steps"].get(name)
    if not step or step.get("fingerprint") != fingerprint:
      return False
    for filename, expected in step["files"].items():
      path = self.directory / filename
      if Path(filename).name != filename or path.is_symlink() or not path.is_file() or file_hash(path) != expected:
        return False
    print(f"Reusing completed {name}")
    return True

  def mark(self, name: str, fingerprint: str, filenames, **extra) -> None:
    self.state["steps"][name] = {
      "fingerprint": fingerprint,
      **extra,
      "files": {f: file_hash(self.directory / f) for f in filenames},
    }
    self._save()

  def forget(self, name: str) -> None:
    if self.state["steps"].pop(name, None) is not None:
      self._save()

  def complete(self, result: dict, deliverables=(), discard=(), binaries: dict | None = None) -> dict:
    """Record the result, drop `discard` files (checkpoints) and publish `deliverables`."""
    for name in discard:
      (self.directory / name).unlink(missing_ok=True)
    self.state["steps"] = {
      key: step for key, step in self.state["steps"].items() if not set(step["files"]) & set(discard)
    }
    result = {
      **result,
      "deliverables": list(deliverables),
      "binaries": binaries,
      "schema": SCHEMA,
      "stage": self.name,
      "settings": self.settings,
      "versions": self.state["versions"],
    }
    self._deliver(result)
    self.state.update(status="complete", result=result)
    self.state.pop("error", None)
    self._save()
    return result

  def _deliver(self, result: dict) -> None:
    if self.public is None:
      return
    self.publish(result.get("deliverables", ()))
    packages = {k: v for k, v in result["versions"].items() if k not in ("python", "platform")}
    report = {
      "stage": self.name,
      # Files from an earlier --resname may remain beside them; these are the current ones.
      "deliverables": list(result.get("deliverables", ())),
      "pyligprep": packages.get("pyligprep"),
      "python": result["versions"].get("python"),
      "platform": result["versions"].get("platform"),
      "python_packages": packages,
    }
    if result.get("binaries"):
      report["binaries"] = result["binaries"]
    write_json(self.public / VERSIONS, report)

  def publish(self, names) -> None:
    """Copy working files to the output directory, never clobbering edited files or symlinks."""
    if self.public is None:
      raise InputError("This stage has no output directory to publish into")
    published = self.state.setdefault("published", {})
    # Any content this stage published before may be replaced (names can alias on
    # case-insensitive filesystems); anything else was edited by the user.
    known = set(published.values())
    for name in names:
      target = self.public / name
      if target.is_symlink():
        raise InputError(f"Refusing to replace output symlink: {target}")
      if target.exists() and file_hash(target) not in known | {file_hash(self.directory / name)}:
        raise InputError(f"Refusing to overwrite modified output: {target}")
    for name in names:
      shutil.copyfile(self.directory / name, self.public / name)
      # Record each copy at once so an interrupted publish is not mistaken for a user edit later.
      published[name] = file_hash(self.public / name)
      self._save()

  def __exit__(self, exc_type, exc, traceback):
    lock = self.lock
    if lock is None:  # __enter__ never opened the lock file
      return
    try:
      if exc is not None and self.cached is None:
        self.state.update(status="failed", error=f"{exc_type.__name__}: {exc}")
        self._save()
    finally:
      fcntl.flock(lock, fcntl.LOCK_UN)
      lock.close()
