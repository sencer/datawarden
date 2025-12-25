"""Value validators for Series and DataFrame data."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, override

if TYPE_CHECKING:
  from collections.abc import Callable

import numpy as np
import pandas as pd

from datawarden.base import Validator
from datawarden.protocols import ScalarConstraint
from datawarden.utils import instantiate_validator

# Type alias for pandas types
PandasData = pd.Series | pd.DataFrame | pd.Index


def _require_pandas(data: object, validator_name: str) -> PandasData:
  """Ensure data is a pandas type, raise TypeError if not."""
  if not isinstance(data, (pd.Series, pd.DataFrame, pd.Index)):
    raise TypeError(
      f"{validator_name} requires pandas Series, DataFrame, or Index, got {type(data).__name__}"
    )
  return data


class IgnoringNaNs(Validator[pd.Series | pd.DataFrame | pd.Index]):
  """Wrapper validator that ignores NaN values during validation.

  Use this to apply validators only to non-NaN values, allowing NaN to pass through.

  Can be used in two ways:

  1. Explicit wrapping - wrap specific validators:
     data: Validated[pd.Series, IgnoringNaNs(Ge(0)), Lt(10)]
     # Only Ge(0) ignores NaNs, Lt(10) still rejects NaNs

  2. Marker mode - apply to all validators:
     data: Validated[pd.Series, Ge(0), Lt(10), IgnoringNaNs()]
     # Equivalent to: IgnoringNaNs(Ge(0)), IgnoringNaNs(Lt(10))

  Example:
    # Allow NaN values but enforce that non-NaN values are >= 0
    data: Validated[pd.Series, IgnoringNaNs(Ge(0))]

    # All validators ignore NaNs
    data: Validated[pd.Series, Ge(0), Lt(100), IgnoringNaNs()]
  """

  def __init__(
    self,
    wrapped: Validator[Any] | type[Validator[Any]] | None = None,  # pyright: ignore[reportExplicitAny]
    *,
    _check_syntax: bool = True,
  ) -> None:
    super().__init__()
    self.wrapped = instantiate_validator(wrapped, _check_syntax=_check_syntax)

  def __repr__(self) -> str:
    if self.wrapped is None:
      return "IgnoringNaNs()"
    return f"IgnoringNaNs({self.wrapped!r})"

  def is_marker(self) -> bool:
    """Check if this is a marker (no wrapped validator)."""
    return self.wrapped is None

  @override
  def validate(self, data: pd.Series | pd.DataFrame | pd.Index) -> None:
    """Validate only non-NaN values."""
    # If no wrapped validator, this is just a marker - no validation
    if self.wrapped is None:
      return

    if isinstance(data, pd.Series):
      # Filter out NaNs, validate the rest
      mask = ~pd.isna(data)
      if not mask.any():
        return  # All NaN, nothing to validate

      # Validate non-NaN subset
      filtered = data[mask]
      self.wrapped.validate(filtered)

    elif isinstance(data, pd.DataFrame):
      # Apply column-wise for DataFrames
      for col in data.columns:
        mask = ~pd.isna(data[col])
        if mask.any():  # pyright: ignore[reportGeneralTypeIssues]
          self.wrapped.validate(data[col][mask])

    elif isinstance(data, pd.Index):
      # For Index, filter and validate
      mask = ~pd.isna(data)
      if mask.any():  # pyright: ignore[reportGeneralTypeIssues]
        # Create temporary series for validation
        filtered = pd.Series(data[mask], dtype=data.dtype)
        self.wrapped.validate(filtered)


class Finite(Validator[pd.Series | pd.DataFrame | pd.Index]):
  """Validator for non-infinite values.

  Rejects infinite values (+Inf, -Inf), allows NaN.
  Use with NonNaN if you need to reject both Inf and NaN.

  Example:
    Validated[pd.Series, Finite]           # No Inf, allows NaN
    Validated[pd.Series, Finite, NonNaN]   # No Inf, no NaN
  """

  @override
  def validate(self, data: pd.Series | pd.DataFrame | pd.Index) -> None:
    _require_pandas(data, "Finite")
    numeric_data = (
      data.select_dtypes(include=[np.number])  # type: ignore[arg-type]
      if isinstance(data, pd.DataFrame)
      else data
    )
    # Check if data is numeric before checking for inf
    if (
      (isinstance(data, pd.DataFrame) or pd.api.types.is_numeric_dtype(data))
      and len(numeric_data) > 0  # type: ignore[arg-type]
      and np.any(np.isinf(numeric_data.values))  # type: ignore[arg-type]
    ):
      raise ValueError("Data must be finite (contains Inf)")


class StrictFinite(Validator[pd.Series | pd.DataFrame | pd.Index]):
  """Validator for strictly finite values (no Inf, no NaN).

  Checks for both NaN and infinite values. Always rejects NaN.
  Equivalent to combining Finite and NonNaN.

  Use Finite alone if you want to allow NaN values.
  """

  @override
  def validate(self, data: pd.Series | pd.DataFrame | pd.Index) -> None:
    _require_pandas(data, "StrictFinite")
    # Check for NaN (axis=None returns scalar bool, pandas-stubs typing issue)
    if data.isna().any(axis=None):  # pyright: ignore[reportArgumentType,reportGeneralTypeIssues]
      raise ValueError("Data must be finite (contains NaN)")

    # Check for infinite values (only for numeric data)
    numeric_data = (
      data.select_dtypes(include=[np.number])  # type: ignore[arg-type]
      if isinstance(data, pd.DataFrame)
      else data
    )
    if (
      (isinstance(data, pd.DataFrame) or pd.api.types.is_numeric_dtype(data))
      and len(numeric_data) > 0  # type: ignore[arg-type]
      and np.any(np.isinf(numeric_data.values))  # type: ignore[arg-type]
    ):
      raise ValueError("Data must be finite (contains Inf)")


class NonEmpty(Validator[pd.Series | pd.DataFrame | pd.Index]):
  """Validator for non-empty data."""

  @override
  def validate(self, data: pd.Series | pd.DataFrame | pd.Index) -> None:
    _require_pandas(data, "NonEmpty")
    if data.empty:
      raise ValueError("Data must not be empty")


class NonNaN(Validator[pd.Series | pd.DataFrame | pd.Index]):
  """Validator for non-NaN values.

  Uses pd.isna() for compatibility with all dtypes including object columns.
  """

  @override
  def validate(self, data: pd.Series | pd.DataFrame | pd.Index) -> None:
    _require_pandas(data, "NonNaN")
    # axis=None returns scalar bool, pandas-stubs typing issue
    if data.isna().any(axis=None):  # pyright: ignore[reportArgumentType,reportGeneralTypeIssues]
      raise ValueError("Data must not contain NaN values")


class NonNegative(Validator[pd.Series | pd.DataFrame | pd.Index]):
  """Validator for non-negative values (>= 0).

  Rejects NaN values by default. Use IgnoringNaNs(NonNegative()) to allow NaN.
  """

  @override
  def validate(self, data: pd.Series | pd.DataFrame | pd.Index) -> None:
    _require_pandas(data, "NonNegative")
    # Check for NaN values first (consistent with comparison validators)
    if pd.isna(data.values).any():
      raise ValueError(
        "Cannot validate non-negative with NaN values (use IgnoringNaNs wrapper to skip NaN values)"
      )
    if np.any(data.values < 0):  # pyright: ignore[reportOperatorIssue,reportUnknownArgumentType]
      raise ValueError("Data must be non-negative")


class Positive(Validator[pd.Series | pd.DataFrame | pd.Index]):
  """Validator for positive values (> 0).

  Rejects NaN values by default. Use IgnoringNaNs(Positive()) to allow NaN.
  """

  @override
  def validate(self, data: pd.Series | pd.DataFrame | pd.Index) -> None:
    _require_pandas(data, "Positive")
    # Check for NaN values first (consistent with comparison validators)
    if pd.isna(data.values).any():
      raise ValueError(
        "Cannot validate positive with NaN values (use IgnoringNaNs wrapper to skip NaN values)"
      )
    if np.any(data.values <= 0):  # pyright: ignore[reportOperatorIssue,reportUnknownArgumentType]
      raise ValueError("Data must be positive")


class Between(Validator[pd.Series | pd.DataFrame | pd.Index]):
  """Validator for values within a range [lower, upper].

  Convenience validator equivalent to Ge(lower), Le(upper).
  Rejects NaN values by default. Use IgnoringNaNs(Between(...)) to allow NaN.

  Args:
    lower: Minimum allowed value (inclusive).
    upper: Maximum allowed value (inclusive).
    inclusive: Tuple of (lower_inclusive, upper_inclusive). Default (True, True).

  Example:
    # Values must be in [0, 100]
    data: Validated[pd.Series, Between(0, 100)]

    # Values must be in (0, 1] (exclusive lower, inclusive upper)
    data: Validated[pd.Series, Between(0, 1, inclusive=(False, True))]
  """

  def __init__(
    self,
    lower: float | int,
    upper: float | int,
    inclusive: tuple[bool, bool] = (True, True),
  ) -> None:
    super().__init__()
    self.lower = lower
    self.upper = upper
    self.lower_inclusive, self.upper_inclusive = inclusive

  @override
  def __repr__(self) -> str:
    if self.lower_inclusive and self.upper_inclusive:
      return f"Between({self.lower!r}, {self.upper!r})"
    return f"Between({self.lower!r}, {self.upper!r}, inclusive={self.lower_inclusive, self.upper_inclusive!r})"

  @override
  def validate(self, data: pd.Series | pd.DataFrame | pd.Index) -> None:
    _require_pandas(data, "Between")
    # Check for NaN values first
    if pd.isna(data.values).any():
      raise ValueError(
        "Cannot validate range with NaN values (use IgnoringNaNs wrapper to skip NaN values)"
      )
    # Check lower bound
    if self.lower_inclusive:
      if np.any(data.values < self.lower):  # pyright: ignore[reportOperatorIssue,reportUnknownArgumentType]
        raise ValueError(f"Data must be >= {self.lower}")
    elif np.any(data.values <= self.lower):  # pyright: ignore[reportOperatorIssue,reportUnknownArgumentType]
      raise ValueError(f"Data must be > {self.lower}")
    # Check upper bound
    if self.upper_inclusive:
      if np.any(data.values > self.upper):  # pyright: ignore[reportOperatorIssue,reportUnknownArgumentType]
        raise ValueError(f"Data must be <= {self.upper}")
    elif np.any(data.values >= self.upper):  # pyright: ignore[reportOperatorIssue,reportUnknownArgumentType]
      raise ValueError(f"Data must be < {self.upper}")


# =============================================================================
# Lambda-based Validators
# =============================================================================


class Is(Validator[pd.Series | pd.DataFrame | pd.Index]):
  """Element-wise predicate validator using a lambda function.

  Validates that all values satisfy the given predicate.

  Example:
    # Check all values are in range [0, 100]
    data: Validated[pd.Series, Is(lambda x: (x >= 0) & (x <= 100))]

    # Check column values satisfy custom condition
    data: Validated[pd.DataFrame, HasColumn("root", Is(lambda x: x**2 < 2))]

    # With descriptive name for error messages
    data: Validated[pd.Series, Is(lambda x: x > 0, name="values must be positive")]
  """

  def __init__(
    self,
    predicate: Callable[[pd.Series], pd.Series],
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

  def __repr__(self) -> str:
    name_str = f", name={self.name!r}" if self.name else ""
    return f"Is(<predicate>{name_str})"

  @override
  def validate(self, data: pd.Series | pd.DataFrame | pd.Index) -> None:
    """Validate that all values satisfy the predicate."""
    if isinstance(data, pd.Series):
      result = self.predicate(data)
      if not result.all():
        msg = self.name or "Values failed predicate check"
        # Count failures
        n_failed = (~result).sum()
        raise ValueError(f"{msg} ({n_failed} values failed)")

    elif isinstance(data, pd.DataFrame):
      for col in data.columns:
        result = self.predicate(data[col])  # pyright: ignore[reportArgumentType]
        if not result.all():
          msg = self.name or f"Column '{col}' failed predicate check"
          n_failed = (~result).sum()
          raise ValueError(f"{msg} ({n_failed} values failed)")

    elif isinstance(data, pd.Index):
      series = pd.Series(data)
      result = self.predicate(series)
      if not result.all():
        msg = self.name or "Index values failed predicate check"
        n_failed = (~result).sum()
        raise ValueError(f"{msg} ({n_failed} values failed)")


class Rows(Validator[pd.DataFrame]):
  """Row-wise predicate validator for DataFrames.

  Validates that all rows satisfy the given predicate.

  Example:
    # Check each row sums to less than 100
    data: Validated[pd.DataFrame, Rows(lambda row: row.sum() < 100)]

    # Check high >= low for each row
    data: Validated[pd.DataFrame, Rows(lambda row: row["high"] >= row["low"])]

    # With descriptive name
    data: Validated[pd.DataFrame, Rows(
      lambda row: row["close"] <= row["high"],
      name="close must not exceed high"
    )]

  Performance Note:
    This validator uses DataFrame.apply(axis=1) internally, which iterates
    row-by-row in Python. For large DataFrames (>10k rows), consider using
    column-based validators like Ge("high", "low") or vectorized Is() instead.
  """

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

    if not results.all():  # pyright: ignore[reportGeneralTypeIssues]
      msg = self.name or "Rows failed predicate check"
      n_failed = (~results).sum()
      failed_indices = data.index[~results].tolist()[:5]  # Show first 5
      raise ValueError(
        f"{msg} ({n_failed} rows failed, e.g. at indices: {failed_indices})"
      )


class OneOf(Validator[pd.Series | pd.Index]):
  """Validator for categorical values - ensures all values are in allowed set.

  Example:
  - OneOf("a", "b", "c")

  Can be used with Index wrapper for index validation:
  - Index(OneOf("x", "y", "z"))

  Can be used with HasColumn for column-specific validation:
  - HasColumn("category", OneOf("a", "b", "c"))
  """

  def __init__(self, *allowed: object) -> None:
    super().__init__()
    self.allowed = set(allowed)

  def __repr__(self) -> str:
    args = ", ".join(repr(a) for a in sorted(self.allowed, key=str))
    return f"OneOf({args})"

  @override
  def validate(self, data: pd.Series | pd.Index) -> None:
    if isinstance(data, pd.Index):
      invalid = set(data) - self.allowed
      if invalid:
        raise ValueError(
          f"Values must be one of {self.allowed}, got invalid: {invalid}"
        )
    else:
      # isinstance(data, pd.Series) implied
      invalid = set(data.dropna().unique()) - self.allowed
      if invalid:
        raise ValueError(
          f"Values must be one of {self.allowed}, got invalid: {invalid}"
        )


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

  def check(self, value: int) -> bool:
    return value == self.n

  def describe(self) -> str:
    return f"== {self.n}"


class _AnyDimConstraint:
  """Any dimension constraint."""

  def check(self, value: int) -> bool:  # noqa: ARG002
    return True

  def describe(self) -> str:
    return "any"


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
  - Shape(10, 5) - Exactly 10 rows, 5 columns
  - Shape(Ge(10), Any) - At least 10 rows, any columns
  - Shape(Any, Le(5)) - Any rows, at most 5 columns
  - Shape(Gt(0), Lt(100)) - More than 0 rows, less than 100 columns
  - Shape(100) - For Series: exactly 100 rows

  For Series, only the first dimension (rows) is checked.
  """

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
