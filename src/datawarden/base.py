"""Base classes for validators."""

from __future__ import annotations

import copy
from enum import IntEnum
from typing import Annotated, override

# Validated alias for Annotated - the main type hint for validated parameters
Validated = Annotated


class Priority(IntEnum):
  """Execution priority for validators (lower runs earlier)."""

  STRUCTURAL = 0  # O(1) or O(columns) - Shape, existence checks
  VECTORIZED = 10  # O(N) numpy - Finite, NonNaN, Simple comparisons
  COMPLEX = 20  # O(N) or O(N log N) - Monotonicity, Index checks, Gaps
  DEFAULT = 50  # Unknown/User-defined
  SLOW = 100  # O(N) Python loops - Rows validation


class Validator[T]:
  """Base class for validators."""

  # Performance flags (using class attributes instead of properties for speed)
  is_chunkable: bool = True
  is_numeric_only: bool = False
  is_promotable: bool = False
  is_holistic: bool = False
  priority: int = Priority.DEFAULT

  def clone(self) -> Validator[T]:
    """Return a fresh copy of this validator with the same configuration.

    The state of the validator (if any) is reset in the clone.
    """
    new_v = copy.deepcopy(self)
    new_v.reset()
    return new_v

  def reset(self) -> None:
    """Reset validator state for new validation run.

    This is called before starting a validation run, especially important
    for stateful chunked validation.
    """
    pass

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


class CompoundValidator[T](Validator[T]):
  """A validator that groups multiple validators for performance optimization.

  Used during planning to fuse consecutive promotable validators into a single
  call, reducing decorator overhead and redundant data checks.
  """

  def __init__(self, validators: list[Validator[T]]) -> None:
    super().__init__()
    self.validators = validators
    # Inherit flags from group (only promotable/holistic if all are)
    self.is_promotable = all(v.is_promotable for v in validators)
    self.is_holistic = all(v.is_holistic for v in validators)
    self.is_numeric_only = all(v.is_numeric_only for v in validators)
    self.is_chunkable = all(v.is_chunkable for v in validators)
    # Priority is the lowest (first) validator in the group
    self.priority = min(v.priority for v in validators)

  @override
  def reset(self) -> None:
    for v in self.validators:
      v.reset()

  @override
  def validate(self, data: T) -> None:
    for v in self.validators:
      v.validate(data)

  @override
  def __repr__(self) -> str:
    inner = ", ".join(repr(v) for v in self.validators)
    return f"CompoundValidator([{inner}])"
