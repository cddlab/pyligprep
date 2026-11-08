"""Resolve CPU thread counts without changing the scheduler's CPU affinity."""

import os

from pyligprep.errors import InputError


def in_slurm_job() -> bool:
  return bool(os.environ.get("SLURM_JOB_ID") or os.environ.get("SLURM_JOBID"))


def cpu_affinity() -> tuple[int, ...] | None:
  get_affinity = getattr(os, "sched_getaffinity", None)
  if get_affinity is None:
    return None
  try:
    return tuple(sorted(get_affinity(0)))
  except OSError:
    return None


def _positive_env(name: str) -> int | None:
  value = os.environ.get(name)
  if value is None:
    return None
  try:
    count = int(value)
  except ValueError as exc:
    raise InputError(f"{name} must be a positive integer; got {value!r}") from exc
  if count < 1:
    raise InputError(f"{name} must be a positive integer; got {value!r}")
  return count


def _slurm_cpu_limit() -> int:
  step = os.environ.get("SLURM_STEP_ID", os.environ.get("SLURM_STEPID"))
  tasks = None
  if step not in (None, "batch", "extern", "interactive"):
    tasks = _positive_env("SLURM_STEP_NUM_TASKS")
  tasks = tasks or _positive_env("SLURM_NTASKS") or 1
  if tasks != 1:
    raise InputError("pyligprep requires a single Slurm task; use --ntasks=1 --cpus-per-task=N")
  per_task = _positive_env("SLURM_CPUS_PER_TASK")
  on_node = _positive_env("SLURM_CPUS_ON_NODE")
  # Slurm defaults to one CPU per task; never pool the CPUs of multiple tasks.
  allocated = per_task or 1
  if on_node is not None:
    allocated = min(allocated, on_node)
  affinity = cpu_affinity()
  if affinity is not None:
    if not affinity:
      raise InputError("The process CPU affinity contains no available CPUs")
    allocated = min(allocated, len(affinity))
  return allocated


def resolve_threads(requested: int | None = None) -> int:
  """Use the local Slurm allocation automatically; retain the non-Slurm default of 8."""
  if requested is not None and (type(requested) is not int or requested < 1):
    raise InputError("--threads must be a positive integer")
  if not in_slurm_job():
    return requested if requested is not None else 8
  limit = _slurm_cpu_limit()
  if requested is not None and requested > limit:
    raise InputError(f"--threads {requested} exceeds the available Slurm/CPU-affinity limit of {limit}")
  return requested if requested is not None else limit
