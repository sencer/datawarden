"""Base classes for validators."""

from __future__ import annotations

from typing import Annotated, override

# Validated alias for Annotated - the main type hint for validated parameters
Validated = Annotated


class Validator[T]:
  """Base class for validators."""

  def validate(self, data: T) -> None:
    """Validate the data and raise an exception if invalid.

    Returns None. Does not return the data because separate validation logic
    should not modify data.
    """
    pass

  @override
  def __eq__(self, other: object) -> bool:
    """Check equality based on type and attributes."""
    if not isinstance(other, type(self)):
      return NotImplemented
    return self.__dict__ == other.__dict__

  @override
  def __hash__(self) -> int:
    """Hash based on type and attributes."""
    # Convert dict items to a sorted tuple to ensure consistent ordering
    # We use string representation of keys just to be safe, though they are usually strings
    return hash((type(self), tuple(sorted(self.__dict__.items()))))
