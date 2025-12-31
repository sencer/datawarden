from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast, override

import numpy as np
import pandas as pd

from datawarden.base import Priority, Validator

if TYPE_CHECKING:
  from datawarden.base import (
    PandasData,
    VectorizedResult,
  )
from datawarden.protocols import MetaValidator
from datawarden.utils import instantiate_validator, report_failures, scalar_any


class Not(Validator[pd.Series | pd.DataFrame | pd.Index], MetaValidator):
  """Inverts the validation logic of the wrapped validator.

  Example:
      ```python
      # Values must NOT be between 5 and 10 (i.e., < 5 or > 10)
      Not(Between(5, 10))

      # Values must NOT be NaN
      Not(IsNaN) # Equivalent to NotNaN
      ```
  """

  priority = Priority.VECTORIZED

  # Set when using smart negation (negated validator directly)
  _using_smart_negation: bool = False

  def __init__(
    self, validator: Validator[Any] | object, ignore_nan: bool = False
  ) -> None:
    super().__init__()

    # Use _check_syntax=False to allow both class and instance patterns
    instantiated = instantiate_validator(validator, _check_syntax=False)
    # If instantiate_validator returns None (non-Validator type), use original
    wrapped = instantiated if instantiated is not None else validator

    # Smart negation: if the wrapped validator has a negate() method,
    # use that instead of wrapping and inverting.
    # We skip this if we are already a subclass of Not (to avoid recursion)
    # or if the result is of the same type as self.
    negated = None
    if type(self) is Not and hasattr(wrapped, "negate"):
      negated = cast("Any", wrapped).negate()

    if negated is not None:
      self.wrapped = negated  # pyright: ignore[reportAssignmentType]
      self._using_smart_negation = True
    else:
      self.wrapped = wrapped  # pyright: ignore[reportAssignmentType]
      self._using_smart_negation = False

    self.is_vectorized = hasattr(self.wrapped, "validate_vectorized")
    self.ignore_nan = ignore_nan

    # Inherit fusion flags from wrapped validator
    self.is_promotable = getattr(self.wrapped, "is_promotable", False)
    self.is_numeric_only = getattr(self.wrapped, "is_numeric_only", False)
    self.is_holistic = getattr(self.wrapped, "is_holistic", False)

  @override
  def describe(self) -> str:
    if hasattr(self.wrapped, "describe"):
      desc = cast("Any", self.wrapped).describe()
    else:
      desc = str(self.wrapped)

    # If using smart negation, the wrapped validator IS already the negated form
    # so just return its description directly
    if self._using_smart_negation:
      return desc

    if desc == "NaN":
      return "not contain NaN"
    if desc == "negative":
      return "non-negative"
    if desc == "positive":
      return "non-positive"
    if desc == "empty":
      return "non-empty"

    return f"not {desc}"

  @override
  def transform(self) -> list[Validator[Any]]:
    """Return self as the only validator (identity transform)."""
    return [self]

  @override
  def validate_vectorized(self, data: PandasData) -> VectorizedResult:
    if not hasattr(self.wrapped, "validate_vectorized"):
      raise NotImplementedError(
        f"Wrapped validator {self.wrapped} does not support vectorization"
      )

    mask = cast("Any", self.wrapped).validate_vectorized(data)

    # If using smart negation, the wrapped validator IS already the negated form
    # so no inversion needed - just return its mask directly
    if self._using_smart_negation:
      return mask

    # Otherwise, invert the mask
    # If mask is a pandas object or numpy array, use ~
    if isinstance(mask, (pd.DataFrame, pd.Series, pd.Index, np.ndarray)):
      return np.logical_not(mask)

    # Scalar boolean fallback
    return not mask

  @override
  def validate(self, data: pd.Series | pd.DataFrame | pd.Index) -> None:
    # If using smart negation, just delegate to the negated validator directly
    # It handles its own NaN checks, error messages, etc.
    if self._using_smart_negation:
      cast("Any", self.wrapped).validate(data)
      return

    # Check for NaN values first (unless ignore_nan is True)
    if not self.ignore_nan:
      desc = self.describe()
      # Optimization: If we are validating "not contain NaN", the NaN guard
      # is redundant and produces a worse error message than the validator itself.
      if desc != "not contain NaN":
        if hasattr(data, "isna"):
          nan_mask = data.isna()
        elif hasattr(data, "values"):
          nan_mask = pd.isna(data.values)
        else:
          nan_mask = pd.isna(data)

        if scalar_any(nan_mask):
          report_failures(
            data,
            nan_mask,
            f"Cannot validate {desc} with NaN values (use IgnoringNaNs wrapper to skip NaN values)",
          )

    # Try vectorized path first
    if self.is_vectorized:
      try:
        mask = self.validate_vectorized(data)

        # Handle scalar boolean mask (e.g. from Empty validator)
        if isinstance(mask, (bool, np.bool_)):
          if not mask:
            desc = self.describe()
            msg = (
              f"Data must {desc}" if desc.startswith("not ") else f"Data must be {desc}"
            )
            raise ValueError(msg)
          return

        # Check for failures (False in mask)
        if isinstance(mask, (pd.DataFrame, pd.Series)):
          is_all_valid = cast("Any", mask.values).all()
        elif isinstance(mask, np.ndarray):
          is_all_valid = mask.all()
        else:
          is_all_valid = bool(mask)

        if is_all_valid:
          return

        # Report failures
        desc = self.describe()
        msg = f"Data must {desc}" if desc.startswith("not ") else f"Data must be {desc}"
        report_failures(data, np.logical_not(mask), msg)
        return

      except (NotImplementedError, AttributeError):
        pass

    # Fallback to sequential (logic assertion)
    # We run wrapped.validate(item) and expect it to fail.
    # If it passes, WE raise Error.
    try:
      cast("Any", self.wrapped).validate(data)
    except ValueError:
      # Wrapped failed -> Not passes
      return

    # Wrapped passed -> Not fails
    desc = self.describe()
    msg = f"Data must {desc}" if desc.startswith("not ") else f"Data must be {desc}"
    raise ValueError(msg)
