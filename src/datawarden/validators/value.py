"""Value validators for Series and DataFrame data."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast, override

if TYPE_CHECKING:
  from collections.abc import Callable

import numpy as np
import pandas as pd

from datawarden.base import (
  Priority,
  Validator,
)

if TYPE_CHECKING:
  from datawarden.base import (
    PandasData,
    VectorizedResult,
  )
from datawarden.protocols import MetaValidator, ScalarConstraint
from datawarden.utils import (
  NUMERIC_KINDS,
  instantiate_validator,
  report_failures,
  scalar_any,
)
from datawarden.validators.columns import HasColumn, HasColumns
from datawarden.validators.comparison import Ge, Le
from datawarden.validators.logic import Not


def _require_pandas(data: object, validator_name: str) -> PandasData:
  """Ensure data is a pandas type, raise TypeError if not."""
  if not isinstance(data, (pd.Series, pd.DataFrame, pd.Index)):
    raise TypeError(
      f"{validator_name} requires pandas Series, DataFrame, or Index, got {type(data).__name__}"
    )
  return data


def _get_numeric_df_values(
  data: pd.DataFrame,
) -> tuple[np.ndarray, pd.DataFrame] | tuple[None, None]:
  """Get numeric values from DataFrame with fast path for all-numeric data.

  Returns:
    Tuple of (values array, error_data DataFrame) or (None, None) if no numeric cols.
  """
  # Fast path: if all columns are numeric, use df.values directly (avoids copy)
  if all(data[c].dtype.kind in NUMERIC_KINDS for c in data.columns):
    return data.values, data
  # Slow path: select only numeric columns
  numeric_data = data.select_dtypes(include=[np.number])
  if numeric_data.empty:
    return None, None
  return numeric_data.values, numeric_data


class IgnoringNaNs(Validator[pd.Series | pd.DataFrame | pd.Index], MetaValidator):
  """Wrapper validator that ignores NaN values during validation.

  Use this to apply validators only to non-NaN values, allowing NaN to pass through.

  Can be used in two ways:

  1. Explicit wrapping - wrap specific validators:
     data: Validated[pd.Series, IgnoringNaNs(Ge(0)), Lt(10)]
     # Only Ge(0) ignores NaNs, Lt(10) still rejects NaNs

  2. Marker mode - apply to all validators:
     data: Validated[pd.Series, Ge(0), Lt(10), IgnoringNaNs()]
     # Equivalent to: IgnoringNaNs(Ge(0)), IgnoringNaNs(Lt(10))

  The wrapper correctly preserves data types, supporting pd.Series, pd.DataFrame,
  and pd.Index (including nested index validators like Index(Datetime)).

  Example:
    ```python
    # Allow NaN values but enforce that non-NaN values are >= 0
    data: Validated[pd.Series, IgnoringNaNs(Ge(0))]

    # All validators ignore NaNs
    data: Validated[pd.Series, Ge(0), Lt(100), IgnoringNaNs()]
    ```
  """

  def __init__(
    self,
    wrapped: Validator[Any] | type[Validator[Any]] | None = None,
    *,
    _check_syntax: bool = True,
  ) -> None:
    super().__init__()
    self.wrapped = instantiate_validator(wrapped, _check_syntax=_check_syntax)

    # Initialize flags from wrapped validator
    if self.wrapped:
      self.is_numeric_only = self.wrapped.is_numeric_only
      self.is_promotable = self.wrapped.is_promotable
      self.is_holistic = self.wrapped.is_holistic
      self.priority = self.wrapped.priority
    else:
      # Marker mode defaults
      self.priority = Priority.VECTORIZED

  def __repr__(self) -> str:
    if self.wrapped is None:
      return "IgnoringNaNs()"
    return f"IgnoringNaNs({self.wrapped!r})"

  def is_marker(self) -> bool:
    """Check if this is a marker (no wrapped validator)."""
    return self.wrapped is None

  @override
  def reset(self) -> None:
    if self.wrapped:
      self.wrapped.reset()

  def transform(self) -> list[Validator[Any]]:
    """Unwrap nested HasColumn/HasColumns validators.

    If this validator wraps a HasColumn/HasColumns, we need to push the
    IgnoringNaNs logic *inside* the column validator so it applies to the
    column's data, not the DataFrame itself.
    """
    if self.wrapped is not None:
      with_ignore_nan = getattr(self.wrapped, "with_ignore_nan", None)
      if with_ignore_nan is not None:
        return [with_ignore_nan()]

    if isinstance(self.wrapped, (HasColumn, HasColumns)):
      inner = self.wrapped
      new_validators: list[Validator[Any]] = []

      for sub in inner.validators:
        if isinstance(sub, IgnoringNaNs):
          new_validators.append(sub)
        else:
          new_validators.append(IgnoringNaNs(sub, _check_syntax=False))

      if isinstance(inner, HasColumn):
        return [HasColumn(inner.column, *new_validators)]
      return [HasColumns(inner.columns, *new_validators)]

    return [self]

  @override
  def validate(self, data: pd.Series | pd.DataFrame | pd.Index) -> None:
    """Validate only non-NaN values."""
    # If no wrapped validator, this is just a marker - no validation
    if self.wrapped is None:
      return

    if isinstance(data, pd.Series):
      # If the wrapped validator has a vectorized validation method, use it.
      # This is an optimization for validators that can handle masks directly.
      try:
        validity_mask = cast("Any", self.wrapped).validate_vectorized(data)
        nan_mask = pd.isna(data)

        # Combine masks: A value is invalid if it's NOT NaN AND the wrapped validator says it's invalid.
        # (True = invalid)
        combined_invalid_mask = np.logical_not(nan_mask) & np.logical_not(validity_mask)

        if combined_invalid_mask.any():
          # Re-run on the non-NaN subset to let the wrapped validator report its specific error message
          # The original line `self.wrapped.validate(data[~nan_mask])` is the correct way to delegate
          # error reporting to the wrapped validator, which will raise an exception with its specific message.
          # The user's proposed change `report_failures(data, np.logical_not(mask), msg)` is not directly applicable
          # here as `mask` and `msg` are undefined, and `np.logical_not(mask)` would invert the wrong mask.
          # The `combined_invalid_mask` already represents the failures.
          # To use report_failures directly, we would need a generic message or to extract one from the wrapped validator.
          # For now, we revert to the original, more robust delegation.
          self.wrapped.validate(data[np.logical_not(nan_mask)])
        return

      except (NotImplementedError, AttributeError):
        # Fallback to filtering out NaNs and validating the rest
        # (This handles validators that haven't implemented vectorized validation)
        mask_not_nan = np.logical_not(pd.isna(data))
        if mask_not_nan.any():
          self.wrapped.validate(data[mask_not_nan])
        return

    elif isinstance(data, pd.DataFrame):
      # Holistic validators that require the whole DataFrame should not be run column-wise.
      # If they reached here, it means they didn't implement with_ignore_nan to handle
      # NaNs surgically, so we fall back to the safe (but aggressive) dropna().
      if self.wrapped.is_holistic:
        self.wrapped.validate(data.dropna())
        return

      # Apply column-wise for DataFrames
      for col in data.columns:
        mask = np.logical_not(pd.isna(data[col]))
        if mask.any():
          self.wrapped.validate(data[col][mask])

    elif isinstance(data, pd.Index):
      # For Index, filter and validate
      mask = np.logical_not(pd.isna(data))
      if mask.any():
        # Preserve Index type when filtering
        filtered = data[mask]
        self.wrapped.validate(filtered)

  @override
  def validate_vectorized(self, data: PandasData) -> VectorizedResult:
    """Validate only non-NaN values."""
    if self.wrapped is None:
      return np.ones(len(data), dtype=bool)

    if hasattr(self.wrapped, "validate_vectorized"):
      # Only attempted if wrapped validator implements the protocol explicitly
      validity_mask = cast("Any", self.wrapped).validate_vectorized(data)
    else:
      raise NotImplementedError(
        f"Wrapped validator {self.wrapped} in IgnoringNaNs does not support vectorization"
      )

    # 2. Get NaN mask (True = NaN)
    nan_mask = pd.isna(data)

    # Combine: valid if (wrapped says valid) OR (is NaN)
    return validity_mask | nan_mask


class AllowNaN(Validator[pd.Series | pd.DataFrame | pd.Index]):
  """Marker validator to explicitly allow NaN values.

  Used to override a global NonNaN constraint locally.

  Example:
    ```python
    @validate
    def process(df: Validated[pd.DataFrame, NonNaN, HasColumn("optional", AllowNaN)]):
        ...
    ```
  """

  @override
  def validate(self, data: pd.Series | pd.DataFrame | pd.Index) -> None:
    pass


class AllowInf(Validator[pd.Series | pd.DataFrame | pd.Index]):
  """Marker validator to explicitly allow Infinite values.

  Used to override a global Finite/StrictFinite constraint locally.

  Example:
    ```python
    @validate
    def process(df: Validated[pd.DataFrame, Finite, HasColumn("slope", AllowInf)]):
        ...
    ```
  """

  @override
  def validate(self, data: pd.Series | pd.DataFrame | pd.Index) -> None:
    pass


class Finite(Validator[pd.Series | pd.DataFrame | pd.Index]):
  """Validator for non-infinite values.

  Rejects infinite values (+Inf, -Inf), allows NaN.
  Use StrictFinite if you need to reject both Inf and NaN.

  Mixed-type Behavior:
    - **DataFrames**: Automatically selects only numeric columns to validate.
      Non-numeric columns (strings, objects, etc.) are ignored. If no local
      overrides exist, this validator is "promoted" to run once on the entire
      DataFrame for maximum performance.
    - **Series/Index**: Strictly requires numeric dtype; raises TypeError if applied
      to string or object data.

  Example:
    ```python
    Validated[pd.Series, Finite]           # No Inf, allows NaN
    Validated[pd.Series, StrictFinite]     # No Inf, no NaN
    ```
  """

  is_numeric_only = True
  is_promotable = True
  priority = Priority.VECTORIZED

  def with_ignore_nan(self) -> Finite:
    """Finite already ignores NaN values."""
    return self

  @override
  def validate(self, data: pd.Series | pd.DataFrame | pd.Index) -> None:
    if isinstance(data, pd.Series):
      if data.dtype.kind not in NUMERIC_KINDS:
        raise TypeError(f"Finite requires numeric data, got {data.dtype}")
      if len(data) > 0 and np.any(mask := np.isinf(data.values)):
        report_failures(data, mask, "Data must be finite (contains Inf)")
    elif isinstance(data, pd.DataFrame):
      # Fast path: if all columns are numeric, use df.values directly (avoids copy)
      if all(data[c].dtype.kind in NUMERIC_KINDS for c in data.columns):
        if len(data) > 0 and np.any(mask := np.isinf(data.values)):
          report_failures(data, mask, "Data must be finite (contains Inf)")
      else:
        # Slow path: select only numeric columns
        numeric_data = data.select_dtypes(include=[np.number])
        if not numeric_data.empty and np.any(mask := np.isinf(numeric_data.values)):
          report_failures(numeric_data, mask, "Data must be finite (contains Inf)")
    elif isinstance(data, pd.Index):
      if data.dtype.kind not in NUMERIC_KINDS:
        raise TypeError(f"Finite requires numeric data, got {data.dtype}")
      if len(data) > 0 and np.any(mask := np.isinf(data.values)):
        report_failures(data, mask, "Data must be finite (contains Inf)")
    else:
      raise TypeError(
        f"Finite requires pandas Series, DataFrame, or Index, got {type(data).__name__}"
      )

  @override
  def validate_vectorized(self, data: PandasData) -> VectorizedResult:
    """Return boolean validity mask."""
    if isinstance(data, pd.DataFrame):
      mask = pd.DataFrame(True, index=data.index, columns=data.columns)
      for col in data.columns:
        if data[col].dtype.kind in NUMERIC_KINDS:
          mask[col] = self.validate_vectorized(data[col])
      return mask

    if data.dtype.kind not in NUMERIC_KINDS:
      return np.ones(len(data), dtype=bool)

    # np.isfinite returns False for NaN, but Finite ALLOWS NaN.
    # np.isinf(nan) is False. So VALID if NOT isinf.
    return np.logical_not(np.isinf(getattr(data, "values", data)))


class StrictFinite(Validator[pd.Series | pd.DataFrame | pd.Index]):
  """Validator for strictly finite values (no Inf, no NaN).

  Checks for both NaN and infinite values in a single atomic operation.
  Uses np.isfinite() which is ~37% faster than separate isna()+isinf() checks.

  Mixed-type Behavior:
    - **DataFrames**: Automatically selects only numeric columns to validate.
      Non-numeric columns (strings, objects, etc.) are ignored. If no local
      overrides exist, this validator is "promoted" to run once on the entire
      DataFrame for maximum performance.
    - **Series/Index**: Strictly requires numeric dtype; raises TypeError if applied
      to string or object data.

  Use Finite alone if you want to allow NaN values.
  """

  is_numeric_only = True
  is_promotable = True
  priority = Priority.VECTORIZED

  def with_ignore_nan(self) -> Finite:
    """Ignore NaN for StrictFinite means allowing NaNs, which is Finite."""
    return Finite()

  @override
  def validate(self, data: pd.Series | pd.DataFrame | pd.Index) -> None:
    if isinstance(data, pd.Series):
      if data.dtype.kind not in NUMERIC_KINDS:
        raise TypeError(f"StrictFinite requires numeric data, got {data.dtype}")
      if len(data) > 0 and not np.all(mask_finite := np.isfinite(data.values)):
        report_failures(
          data, np.logical_not(mask_finite), "Data must be finite (contains NaN or Inf)"
        )
    elif isinstance(data, pd.DataFrame):
      # Fast path: if all columns are numeric, use df.values directly (avoids copy)
      if all(data[c].dtype.kind in NUMERIC_KINDS for c in data.columns):
        if len(data) > 0 and not np.all(mask_finite := np.isfinite(data.values)):
          report_failures(
            data,
            np.logical_not(mask_finite),
            "Data must be finite (contains NaN or Inf)",
          )
      else:
        # Slow path: select only numeric columns
        numeric_data = data.select_dtypes(include=[np.number])
        if not numeric_data.empty and not np.all(
          mask_finite := np.isfinite(numeric_data.values)
        ):
          report_failures(
            numeric_data,
            np.logical_not(mask_finite),
            "Data must be finite (contains NaN or Inf)",
          )
    elif isinstance(data, pd.Index):
      if data.dtype.kind not in NUMERIC_KINDS:
        raise TypeError(f"StrictFinite requires numeric data, got {data.dtype}")
      if len(data) > 0 and not np.all(mask_finite := np.isfinite(data.values)):
        report_failures(
          data, np.logical_not(mask_finite), "Data must be finite (contains NaN or Inf)"
        )

  @override
  def validate_vectorized(self, data: PandasData) -> VectorizedResult:
    """Return boolean validity mask."""
    if isinstance(data, pd.DataFrame):
      mask = pd.DataFrame(True, index=data.index, columns=data.columns)
      for col in data.columns:
        if data[col].dtype.kind in NUMERIC_KINDS:
          mask[col] = self.validate_vectorized(data[col])
      return mask

    if data.dtype.kind not in NUMERIC_KINDS:
      return np.ones(len(data), dtype=bool)

    return np.isfinite(getattr(data, "values", data))


class Empty(Validator[pd.Series | pd.DataFrame | pd.Index]):
  """Validator that ensures data is empty (no rows).

  Priority: 0 (Structural).
  """

  is_holistic = True
  priority = 0

  @override
  def validate(self, data: pd.Series | pd.DataFrame | pd.Index) -> None:
    _require_pandas(data, "Empty")
    if not data.empty:
      raise ValueError("Data must be empty")

  @override
  def negate(self) -> NotEmpty:
    """Return NotEmpty as the logical negation of Empty."""
    return NotEmpty()


class NotEmpty(Validator[pd.Series | pd.DataFrame | pd.Index]):
  """Validator for non-empty data (at least one row).

  Checks that the data has at least one row. For DataFrames, this is an
  atomic O(1) check.

  Priority: 0 (Structural).
  """

  is_holistic = True
  priority = 0

  @override
  def validate(self, data: pd.Series | pd.DataFrame | pd.Index) -> None:
    _require_pandas(data, "NotEmpty")
    if data.empty:
      raise ValueError("Data must be non-empty")

  @override
  def negate(self) -> Empty:
    """Return Empty as the logical negation of NotEmpty."""
    return Empty()


class Positive(Validator[pd.Series | pd.DataFrame | pd.Index]):
  """Validator for positive values (> 0).

  Rejects NaN values by default. Use IgnoringNaNs(Positive()) to allow NaN.

  Mixed-type Behavior:
    - **DataFrames**: Automatically selects only numeric columns to validate.
    - **Series/Index**: Strictly requires numeric dtype; raises TypeError otherwise.
  """

  is_numeric_only = True
  is_promotable = True
  priority = Priority.VECTORIZED

  def __init__(self, ignore_nan: bool = False) -> None:
    super().__init__()
    self.ignore_nan = ignore_nan

  @override
  def validate(self, data: pd.Series | pd.DataFrame | pd.Index) -> None:
    if isinstance(data, pd.Series):
      if data.dtype.kind not in NUMERIC_KINDS:
        raise TypeError(f"Positive requires numeric data, got {data.dtype}")
      vals = data.values
      error_data = data
    elif isinstance(data, pd.DataFrame):
      result = _get_numeric_df_values(data)
      if result[0] is None:
        return
      vals, error_data = result
    elif isinstance(data, pd.Index):
      vals = data
      error_data = data
    else:
      return

    if not self.ignore_nan and scalar_any(mask_nan := pd.isna(vals)):
      report_failures(
        error_data,
        mask_nan,
        "Cannot validate positive with NaN values (use IgnoringNaNs wrapper to skip NaN values)",
      )
    if scalar_any(vals <= 0):
      mask = vals <= 0
      report_failures(error_data, mask, "Data must be positive")

  @override
  def validate_vectorized(self, data: PandasData) -> VectorizedResult:
    """Return boolean validity mask."""
    if isinstance(data, pd.DataFrame):
      mask = pd.DataFrame(True, index=data.index, columns=data.columns)
      for col in data.columns:
        if data[col].dtype.kind in NUMERIC_KINDS:
          mask[col] = self.validate_vectorized(data[col])
      return mask

    if data.dtype.kind not in NUMERIC_KINDS:
      return np.ones(len(data), dtype=bool)

    vals = getattr(data, "values", data)
    valid_mask = vals > 0
    if self.ignore_nan:
      valid_mask |= pd.isna(vals)
    return valid_mask

  @override
  def negate(self) -> Le:
    """Logical negation of Positive (x > 0 -> x <= 0)."""
    return Le(0, ignore_nan=self.ignore_nan)


class Negative(Validator[pd.Series | pd.DataFrame | pd.Index]):
  """Validator for negative values (< 0).

  Rejects NaN values by default. Use IgnoringNaNs(Negative()) to allow NaN.

  Mixed-type Behavior:
    - **DataFrames**: Automatically selects only numeric columns to validate.
    - **Series/Index**: Strictly requires numeric dtype; raises TypeError otherwise.
  """

  is_numeric_only = True
  is_promotable = True
  priority = Priority.VECTORIZED

  def __init__(self, ignore_nan: bool = False) -> None:
    super().__init__()
    self.ignore_nan = ignore_nan

  @override
  def validate(self, data: pd.Series | pd.DataFrame | pd.Index) -> None:
    if isinstance(data, pd.Series):
      if data.dtype.kind not in NUMERIC_KINDS:
        raise TypeError(f"Negative requires numeric data, got {data.dtype}")
      vals = data.values
      error_data = data
    elif isinstance(data, pd.DataFrame):
      result = _get_numeric_df_values(data)
      if result[0] is None:
        return
      vals, error_data = result
    elif isinstance(data, pd.Index):
      vals = data
      error_data = data
    else:
      return

    if not self.ignore_nan and scalar_any(mask_nan := pd.isna(vals)):
      report_failures(
        error_data,
        mask_nan,
        "Cannot validate negative with NaN values (use IgnoringNaNs wrapper to skip NaN values)",
      )
    if scalar_any(vals >= 0):
      mask = vals >= 0
      report_failures(error_data, mask, "Data must be negative")

  @override
  def validate_vectorized(self, data: PandasData) -> VectorizedResult:
    """Return boolean validity mask."""
    if isinstance(data, pd.DataFrame):
      mask = pd.DataFrame(True, index=data.index, columns=data.columns)
      for col in data.columns:
        if data[col].dtype.kind in NUMERIC_KINDS:
          mask[col] = self.validate_vectorized(data[col])
      return mask

    if data.dtype.kind not in NUMERIC_KINDS:
      return np.ones(len(data), dtype=bool)

    vals = getattr(data, "values", data)
    valid_mask = vals < 0
    if self.ignore_nan:
      valid_mask |= pd.isna(vals)
    return valid_mask

  @override
  def negate(self) -> Ge:
    """Logical negation of Negative (x < 0 -> x >= 0)."""
    return Ge(0, ignore_nan=self.ignore_nan)


class Between(Validator[pd.Series | pd.DataFrame | pd.Index]):
  """Validator for values within a range [lower, upper].

  Convenience validator equivalent to Ge(lower), Le(upper).
  Rejects NaN values by default. Use IgnoringNaNs(Between(...)) to allow NaN.

  Mixed-type Behavior:
    - **DataFrames**: Automatically selects only numeric columns to validate.
    - **Series/Index**: Strictly requires numeric dtype; raises TypeError otherwise.

  Args:
    lower: Minimum allowed value (inclusive).
    upper: Maximum allowed value (inclusive).
    inclusive: Tuple of (lower_inclusive, upper_inclusive). Default (True, True).

  Example:
    ```python
    # Values must be in [0, 100]
    data: Validated[pd.Series, Between(0, 100)]

    # Values must be in (0, 1] (exclusive lower, inclusive upper)
    data: Validated[pd.Series, Between(0, 1, inclusive=(False, True))]
    ```
  """

  is_numeric_only = True
  is_promotable = True
  priority = Priority.VECTORIZED

  def __init__(
    self,
    lower: float | int,
    upper: float | int,
    inclusive: tuple[bool, bool] = (True, True),
    ignore_nan: bool = False,
  ) -> None:
    super().__init__()
    self.lower = lower
    self.upper = upper
    self.lower_inclusive, self.upper_inclusive = inclusive
    self.ignore_nan = ignore_nan

  @override
  def __repr__(self) -> str:
    args = [f"{self.lower!r}", f"{self.upper!r}"]
    if not (self.lower_inclusive and self.upper_inclusive):
      args.append(f"inclusive={self.lower_inclusive, self.upper_inclusive!r}")
    if self.ignore_nan:
      args.append("ignore_nan=True")
    return f"Between({', '.join(args)})"

  @override
  def validate(self, data: pd.Series | pd.DataFrame | pd.Index) -> None:
    if isinstance(data, pd.Series):
      if data.dtype.kind not in NUMERIC_KINDS:
        raise TypeError(f"Between requires numeric data, got {data.dtype}")
      vals = data.values
      error_data = data
    elif isinstance(data, pd.DataFrame):
      result = _get_numeric_df_values(data)
      if result[0] is None:
        return
      vals, error_data = result
    elif isinstance(data, pd.Index):
      vals = data
      error_data = data
    else:
      return

    # Check for NaN values first
    if not self.ignore_nan and scalar_any(mask_nan := pd.isna(vals)):
      report_failures(
        error_data,
        mask_nan,
        "Cannot validate range with NaN values (use IgnoringNaNs wrapper to skip NaN values)",
      )
    # Check lower bound
    if self.lower_inclusive:
      if scalar_any(vals < self.lower):
        mask = vals < self.lower
        report_failures(error_data, mask, f"Data must be >= {self.lower}")
    elif scalar_any(vals <= self.lower):
      mask = vals <= self.lower
      report_failures(error_data, mask, f"Data must be > {self.lower}")

    # Check upper bound
    if self.upper_inclusive:
      if scalar_any(vals > self.upper):
        mask = vals > self.upper
        report_failures(error_data, mask, f"Data must be <= {self.upper}")
    elif scalar_any(vals >= self.upper):
      mask = vals >= self.upper
      report_failures(error_data, mask, f"Data must be < {self.upper}")

  @override
  def validate_vectorized(self, data: PandasData) -> VectorizedResult:
    """Return boolean validity mask."""
    if isinstance(data, pd.DataFrame):
      # Column-wise check
      mask = pd.DataFrame(True, index=data.index, columns=data.columns)
      for col in data.columns:
        if data[col].dtype.kind in NUMERIC_KINDS:
          mask[col] = self.validate_vectorized(data[col])
      return mask

    if data.dtype.kind not in NUMERIC_KINDS:
      return np.ones(len(data), dtype=bool)

    vals = getattr(data, "values", data)
    # Lower Check
    lower_mask = (vals >= self.lower) if self.lower_inclusive else (vals > self.lower)
    # Upper Check
    upper_mask = (vals <= self.upper) if self.upper_inclusive else (vals < self.upper)

    valid_mask = lower_mask & upper_mask
    if self.ignore_nan:
      valid_mask |= pd.isna(vals)
    return valid_mask

  @override
  def negate(self) -> Outside:
    """Return Outside as the logical negation of Between."""
    return Outside(
      self.lower,
      self.upper,
      inclusive=(self.lower_inclusive, self.upper_inclusive),
      ignore_nan=self.ignore_nan,
    )


class Outside(Validator[pd.Series | pd.DataFrame | pd.Index]):
  """Validator for values outside a range (lower, upper).

  Logical negation of Between.
  Rejects NaN values by default. Use IgnoringNaNs(Outside(...)) to allow NaN.

  Args:
    lower: Minimum of the excluded range.
    upper: Maximum of the excluded range.
    inclusive: Tuple of (lower_inclusive, upper_inclusive) for the EXCLUDED range.
               Default (True, True) means [lower, upper] is excluded.
               If (True, True), validity is (vals < lower) | (vals > upper).
  """

  is_numeric_only = True
  is_promotable = True
  priority = Priority.VECTORIZED

  def __init__(
    self,
    lower: float | int,
    upper: float | int,
    inclusive: tuple[bool, bool] = (True, True),
    ignore_nan: bool = False,
  ) -> None:
    super().__init__()
    self.lower = lower
    self.upper = upper
    self.lower_exclusive, self.upper_exclusive = inclusive
    self.ignore_nan = ignore_nan

  @override
  def __repr__(self) -> str:
    args = [f"{self.lower!r}", f"{self.upper!r}"]
    if not (self.lower_exclusive and self.upper_exclusive):
      args.append(f"inclusive={self.lower_exclusive, self.upper_exclusive!r}")
    if self.ignore_nan:
      args.append("ignore_nan=True")
    return f"Outside({', '.join(args)})"

  @override
  def validate(self, data: pd.Series | pd.DataFrame | pd.Index) -> None:
    # Values must be outside the range
    # The vectorized method returns a mask where True means "valid" (outside the range)
    # So we need to report failures where the mask is False (~mask)
    if isinstance(data, (pd.Index, pd.Series, pd.DataFrame)):
      mask = self.validate_vectorized(data)
      if not np.all(mask):
        error_data = data
        if isinstance(data, pd.DataFrame):
          # Narrow to numeric columns for failure reporting
          _, error_data = _get_numeric_df_values(data)
          if error_data is None:
            return

        report_failures(
          error_data,
          np.logical_not(mask),
          f"Data must be outside [{self.lower}, {self.upper}]",
        )

  @override
  def validate_vectorized(self, data: PandasData) -> VectorizedResult:
    """Return boolean validity mask."""
    if isinstance(data, pd.DataFrame):
      mask = pd.DataFrame(True, index=data.index, columns=data.columns)
      for col in data.columns:
        if data[col].dtype.kind in NUMERIC_KINDS:
          mask[col] = self.validate_vectorized(data[col])
      return mask

    if data.dtype.kind not in NUMERIC_KINDS:
      return np.ones(len(data), dtype=bool)

    vals = getattr(data, "values", data)
    # Invalid if (vals >= lower) AND (vals <= upper) -> depends on inclusive
    # Valid if (vals < lower) OR (vals > upper) -> depends on exclusive
    lower_fail = (vals >= self.lower) if self.lower_exclusive else (vals > self.lower)
    upper_fail = (vals <= self.upper) if self.upper_exclusive else (vals < self.upper)

    invalid_mask = lower_fail & upper_fail
    valid_mask = np.logical_not(invalid_mask)

    if self.ignore_nan:
      valid_mask |= pd.isna(vals)
    return valid_mask

  @override
  def negate(self) -> Between:
    """Return Between as the logical negation of Outside."""
    return Between(
      self.lower,
      self.upper,
      inclusive=(self.lower_exclusive, self.upper_exclusive),
      ignore_nan=self.ignore_nan,
    )


# =============================================================================
# Lambda-based Validators
# =============================================================================


class Is(Validator[pd.Series | pd.DataFrame | pd.Index]):
  """Element-wise predicate validator using a lambda function.

  Validates that all values satisfy the given predicate.

  Example:
    ```python
    # Check all values are in range [0, 100]
    data: Validated[pd.Series, Is(lambda x: (x >= 0) & (x <= 100))]

    # Check column values satisfy custom condition
    data: Validated[pd.DataFrame, HasColumn("root", Is(lambda x: x**2 < 2))]

    # With descriptive name for error messages
    data: Validated[pd.Series, Is(lambda x: x > 0, name="values must be positive")]
    ```
  """

  def __init__(
    self,
    predicate: Callable[[Any], Any],
    name: str | None = None,
  ) -> None:
    """Initialize Is validator.

    Args:
      predicate: Function that takes array-like and returns boolean array.
                 Should be vectorized (work on entire Series/array at once).
      name: Optional description for error messages.
    """
    super().__init__()
    self.predicate = predicate
    self.name = name
    self.is_holistic = True

  @override
  def describe(self) -> str:
    """Return a descriptive string for the predicate."""
    return self.name or "satisfied predicate"

  @override
  def validate_vectorized(self, data: PandasData) -> VectorizedResult:
    """Return boolean validity mask."""
    # We assume the predicate is vectorized
    return self.predicate(data)

  @override
  def validate(self, data: pd.Series | pd.DataFrame | pd.Index) -> None:
    """Validate that all values satisfy the predicate."""
    if isinstance(data, pd.Series):
      result = self.predicate(data)
      if not result.all():
        msg = self.name or "Values failed predicate check"
        # Count failures
        n_failed = np.logical_not(result).sum()
        raise ValueError(f"{msg} ({n_failed} values failed)")

    elif isinstance(data, pd.DataFrame):
      result = self.predicate(data)
      if not np.all(result):
        msg = self.name or "DataFrame failed predicate check"
        # For DataFrames, we can try to be more specific if it returned a mask
        if isinstance(result, (pd.DataFrame, pd.Series)):
          if isinstance(result, pd.DataFrame):
            n_failed = int(np.logical_not(result).sum().sum())
            failed_cols = [col for col in result.columns if not result[col].all()]
            if len(failed_cols) == 1:
              msg = self.name or f"Column '{failed_cols[0]}' failed predicate check"
            else:
              msg = self.name or f"Columns {failed_cols} failed predicate check"
          else:
            n_failed = np.logical_not(result).sum()

          raise ValueError(f"{msg} ({n_failed} values failed)")
        raise ValueError(msg)

    elif isinstance(data, pd.Index):
      series = pd.Series(data)
      result = self.predicate(series)
      if not result.all():
        msg = self.name or "Index values failed predicate check"
        n_failed = np.logical_not(result).sum()
        raise ValueError(f"{msg} ({n_failed} values failed)")


class Rows(Validator[pd.DataFrame]):
  """Row-wise predicate validator for DataFrames.

  Validates that all rows satisfy the given predicate.

  Example:
    ```python
    # Check each row sums to less than 100
    data: Validated[pd.DataFrame, Rows(lambda row: row.sum() < 100)]
    ```

  Performance Note:
    This validator uses DataFrame.apply(axis=1) internally, which iterates
    row-by-row in Python. For large DataFrames (>10k rows), it is significantly
    slower than vectorized operations.

  Optimization Example:
    ```python
    # ❌ SLOW: Row-wise (Python loop)
    # Rows(lambda row: row["a"] + row["b"] == row["c"])
    ```

    Note: The vectorized version using Is() is much faster as it avoids
    Python-level row iteration.

  Priority: 100 (Slow) - Validated last, only if all other validators pass.
  """

  is_holistic = True
  priority = Priority.SLOW

  def __init__(
    self,
    predicate: Callable[[pd.Series], bool],
    name: str | None = None,
  ) -> None:
    """Initialize Rows validator.

    Args:
      predicate: Function that takes a row (as Series) and returns bool.
      name: Optional description for error messages.
    """
    super().__init__()
    self.predicate = predicate
    self.name = name

  def __repr__(self) -> str:
    name_str = f", name={self.name!r}" if self.name else ""
    return f"Rows(<predicate>{name_str})"

  @override
  def validate(self, data: pd.DataFrame) -> None:
    """Validate that all rows satisfy the predicate."""
    if not isinstance(data, pd.DataFrame):
      raise TypeError("Rows validator requires a pandas DataFrame")

    # Apply predicate to each row
    results = data.apply(self.predicate, axis=1)

    if not results.all():
      msg = self.name or "Rows failed predicate check"
      n_failed = np.logical_not(results).sum()
      failed_indices = data.index[np.logical_not(results)].tolist()[:5]  # Show first 5
      raise ValueError(
        f"{msg} ({n_failed} rows failed, e.g. at indices: {failed_indices})"
      )

  @override
  def validate_vectorized(self, data: PandasData) -> VectorizedResult:
    """Return boolean validity mask."""
    if not isinstance(data, pd.DataFrame):
      raise TypeError("Rows validator requires a pandas DataFrame")
    # We assume the predicate is vectorized (operates on the whole object)
    return self.predicate(cast("Any", data))


class OneOf(Validator[pd.Series | pd.DataFrame | pd.Index]):
  """Validator for categorical values - ensures all values are in allowed set.

  Performance Note:
    Uses vectorized `isin()` operations for Series validation, making it
    extremely efficient for large datasets compared to manual set iteration.

  Example:
    ```python
    # OneOf("a", "b", "c")

    # Can be used with Index wrapper for index validation:
    # Index(OneOf("x", "y", "z"))

    # Can be used with HasColumn for column-specific validation:
    # HasColumn("category", OneOf("a", "b", "c"))
    ```
  """

  is_promotable = True
  priority = Priority.VECTORIZED

  def __init__(self, *allowed: object) -> None:
    super().__init__()
    self.allowed = set(allowed)

  def __repr__(self) -> str:
    args = ", ".join(repr(a) for a in sorted(self.allowed, key=str))
    return f"OneOf({args})"

  @override
  def validate(self, data: pd.Series | pd.DataFrame | pd.Index) -> None:
    if isinstance(data, (pd.Index, pd.Series)):
      mask = np.logical_not(data.isin(self.allowed)) & pd.notna(data)
      if mask.any():
        # Use set subtraction for Index if it's faster, but isin covers both
        invalid = set(pd.unique(data[mask]))
        report_failures(
          data, mask, f"Values must be one of {self.allowed}, got invalid: {invalid}"
        )

  @override
  def validate_vectorized(self, data: PandasData) -> VectorizedResult:
    """Return boolean validity mask."""
    if isinstance(data, (pd.Index, pd.Series)):
      return data.isin(self.allowed) | pd.isna(data)
    if isinstance(data, pd.DataFrame):
      # data.isin results in False for NaN, but we allow NaN
      return data.isin(self.allowed) | pd.isna(data)
    return np.ones(len(data), dtype=bool)


# =============================================================================
# Shape Validators
# =============================================================================

# typing.Any is used for 'any dimension' in Shape constraints
# pyright: reportMissingSuperCall=false, reportImplicitOverride=false
# pyright: reportIncompatibleMethodOverride=false


class _ExactDim:
  """Exact dimension constraint."""

  def __init__(self, n: int) -> None:
    self.n = n

  def check(self, value: object) -> bool:
    return value == self.n

  def describe(self) -> str:
    return f"== {self.n}"


class _AnyDimConstraint:
  """Any dimension constraint."""

  def check(self, value: object) -> bool:  # noqa: ARG002
    return True

  def describe(self) -> str:
    return "any"


class IsNaN(Validator[Any]):
  """Validator and ScalarConstraint for checking if a value is NaN."""

  def check(self, value: object) -> bool:
    return pd.isna(cast("Any", value))

  def describe(self) -> str:
    return "NaN"

  @override
  def validate_vectorized(self, data: PandasData) -> VectorizedResult:
    """Return boolean validity mask."""
    return pd.isna(getattr(data, "values", data))

  @override
  def negate(self) -> NotNaN:
    """Return NotNaN as the logical negation of IsNaN."""
    return NotNaN()


class NotNaN(Not):
  """Validator that ensures values are not NaN (missing).

  Equivalent to Not(IsNaN).
  """

  def __init__(self) -> None:
    super().__init__(IsNaN())

  @override
  def negate(self) -> IsNaN:
    """Return IsNaN as the logical negation of NotNaN."""
    return IsNaN()


def _parse_dim_constraint(item: object) -> ScalarConstraint:
  """Parse a dimension constraint from Shape() arguments.

  Accepts:
  - int: exact dimension match
  - Any (from typing): any dimension allowed
  - ScalarConstraint: any object satisfying the protocol (Ge, Le, etc.)
  """
  # Check for typing.Any (use identity check)
  if item is Any:
    return _AnyDimConstraint()

  if isinstance(item, int):
    return _ExactDim(item)

  # Check protocol compliance
  if isinstance(item, ScalarConstraint):
    return item

  raise TypeError(f"Invalid shape constraint: {item}")


class Shape(Validator[pd.Series | pd.DataFrame | pd.Index]):
  """Validator for DataFrame/Series dimensions.

  Supports exact values, constraints, or Any for flexible validation:
    ```python
    - Shape(10, 5) - Exactly 10 rows, 5 columns
    - Shape(Ge(10), Any) - At least 10 rows, any columns
    - Shape(Any, Le(5)) - Any rows, at most 5 columns
    - Shape(Gt(0), Lt(100)) - More than 0 rows, less than 100 columns
    - Shape(100) - For Series: exactly 100 rows
    ```

  For Series, only the first dimension (rows) is checked.

  Priority: 0 (Structural) - Validated first before any content checks.
  """

  is_chunkable = False
  is_holistic = True
  priority = Priority.STRUCTURAL

  def __init__(
    self,
    rows: object,
    cols: object | None = None,
  ) -> None:
    super().__init__()
    self.rows = _parse_dim_constraint(rows)
    self.cols = _parse_dim_constraint(cols) if cols is not None else None

  @override
  def validate(self, data: pd.Series | pd.DataFrame | pd.Index) -> None:
    if isinstance(data, (pd.Series, pd.Index)):
      n_rows = len(data)
      if not self.rows.check(n_rows):
        type_name = "Index" if isinstance(data, pd.Index) else "Series"
        raise ValueError(
          f"{type_name} must have {self.rows.describe()} rows, got {n_rows}"
        )
    else:
      # isinstance(data, pd.DataFrame) implied
      n_rows = len(data)
      n_cols = len(data.columns)

      if not self.rows.check(n_rows):
        raise ValueError(
          f"DataFrame must have {self.rows.describe()} rows, got {n_rows}"
        )
      if self.cols is not None and not self.cols.check(n_cols):
        raise ValueError(
          f"DataFrame must have {self.cols.describe()} columns, got {n_cols}"
        )


# =============================================================================
# Legacy Aliases (Backward Compatibility)
# =============================================================================


class NonNegative(Ge):
  """Legacy alias for Ge(0)."""

  def __init__(self, *, ignore_nan: bool = False) -> None:
    super().__init__(0, ignore_nan=ignore_nan)
