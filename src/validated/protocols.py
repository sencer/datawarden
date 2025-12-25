"""Protocols from validated."""

from typing import Protocol, runtime_checkable


@runtime_checkable
class ScalarConstraint(Protocol):
  """Protocol for objects that can constrain a scalar value."""

  def check(self, value: int) -> bool:
    """Check if the value satisfies the constraint."""
    ...

  def describe(self) -> str:
    """Return a string description of the constraint (e.g. '>= 10')."""
    ...
