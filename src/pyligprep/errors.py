"""Errors reported to CLI users without hiding the original exception."""


class PreparationError(Exception):
  """Base class for expected preparation failures."""


class InputError(PreparationError):
  """Invalid or inconsistent chemical input or saved data."""


class CalculationError(PreparationError):
  """An electronic structure, optimization or fitting calculation failed."""


class DependencyError(PreparationError):
  """A requested runtime dependency is unavailable."""
