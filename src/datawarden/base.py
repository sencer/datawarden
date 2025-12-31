"""Base classes for validators."""

from __future__ import annotations

import copy
from enum import IntEnum
from typing import Annotated, Any, cast, override

import numpy as np
import pandas as pd

# Validated alias for Annotated - the main type hint for validated parameters
Validated = Annotated

# Type aliases for pandas types
PandasData = pd.Series | pd.DataFrame | pd.Index

# Type alias for vectorized validation results (mask or scalar True)
VectorizedResult = pd.Series | pd.DataFrame | bool | Any


class Priority(IntEnum):
  """Execution priority for validators (lower runs earlier)."""

  STRUCTURAL = 0  # O(1) or O(columns) - Shape, existence checks
  VECTORIZED = 10  # O(N) numpy - Finite, NotNaN, Simple comparisons
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

  def describe(self) -> str:
    """Return a descriptive string for the validator."""
    return self.__class__.__name__

  def negate(self) -> Validator[T] | None:
    """Return an optimized negated version of this validator, or None.

    If a validator knows its logical negation (e.g., Positive -> NonPositive),
    it can return that here. The Not() wrapper will use this instead of
    doing bitwise mask inversion, which is faster.

    Returns None if no optimized negation is available.
    """
    return None

  def validate_vectorized(self, data: PandasData) -> VectorizedResult:
    """Validate all values in vectorized data (Series, DataFrame, or Index).

    Returns a boolean mask (or scalar True).
    Raises NotImplementedError if vectorization is not supported.
    """
    raise NotImplementedError(
      f"Validator {self.__class__.__name__} does not support vectorization"
    )

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
    # 1. OPTIMIZED PATH (Vectorized Fusion)
    # Try to validate all at once using boolean logic.
    if isinstance(data, (pd.Series, pd.DataFrame, pd.Index)):
      try:
        # This method raises NotImplementedError if any child lacks vectorization support
        mask = self.validate_vectorized(data)

        # Handle mask checks (True scalar or array-like)
        if hasattr(mask, "all"):
          # Use values.all() or np.all() to handle DataFrame masks
          if isinstance(mask, (pd.DataFrame, pd.Series)):
            if cast("Any", mask.values).all():
              return  # Success
          elif cast("Any", mask).all():
            return  # Success
        elif mask is True:
          return

      except (NotImplementedError, AttributeError, TypeError):
        # Fallback if vectorization not supported
        pass
    elif hasattr(data, "values") and isinstance(
      getattr(data, "values", None), (np.ndarray, pd.Series, pd.DataFrame, pd.Index)
    ):
      # Handle other array-like types if necessary, or just fall through
      pass

    # 2. SLOW / FALLBACK PATH
    # Run sequentially. This happens if:
    # a) Vectorization not supported
    # b) Validation failed (mask had some False). We re-run to let individual validators
    #    report their specific errors with nice messages.
    for v in self.validators:
      v.validate(data)

  @override
  def __repr__(self) -> str:
    inner = ", ".join(repr(v) for v in self.validators)
    return f"CompoundValidator([{inner}])"

  @override
  def validate_vectorized(self, data: PandasData) -> VectorizedResult:
    """Validate all validators in the group using vectorized operations.

    Returns:
      A boolean mask where True means all validators passed.

    Raises:
      NotImplementedError: If any child validator does not support vectorization.
    """
    if not self.validators:
      # No validators = always valid? Or error?
      # Assuming mostly panda objects, returning True scalar or array?
      # Return True scalar which works in broadcasting.
      return True

    final_mask = None

    for v in self.validators:
      # We rely on dynamic protocol check.
      # If any validator doesn't support it, we must fail so the caller (IgnoringNaNs or Plan)
      # falls back to safe sequential validation.
      if not hasattr(v, "validate_vectorized"):
        raise NotImplementedError(f"Validator {v} does not support vectorization")

      mask = cast("Any", v).validate_vectorized(data)

      if final_mask is None:
        final_mask = mask
      else:
        # Combine with BITWISE AND
        final_mask &= mask

    return cast("VectorizedResult", final_mask)
