"""Value validators for Series and DataFrame data."""

from __future__ import annotations

from typing import Any, override

import numpy as np
import pandas as pd

from validated.base import Validator
from validated.protocols import ScalarConstraint


class Finite(Validator[pd.Series | pd.DataFrame | pd.Index]):
  """Validator for non-infinite values.

  Rejects infinite values (+Inf, -Inf), allows NaN.
  Works correctly with the Nullable marker.

  Example:
    Validated[pd.Series, Finite]           # No Inf, no NaN (NonNaN auto-applied)
    Validated[pd.Series, Finite, Nullable] # No Inf, allows NaN
  """

  @override
  def validate(
    self, data: pd.Series | pd.DataFrame | pd.Index
  ) -> pd.Series | pd.DataFrame | pd.Index:
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
    return data


class StrictFinite(Validator[pd.Series | pd.DataFrame | pd.Index]):
  """Validator for strictly finite values (no Inf, no NaN).

  Checks for both NaN and infinite values. Always rejects NaN
  regardless of the Nullable marker.

  Use Finite instead if you want to allow NaN with the Nullable marker.
  """

  @override
  def validate(
    self, data: pd.Series | pd.DataFrame | pd.Index
  ) -> pd.Series | pd.DataFrame | pd.Index:
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
    return data


class NonEmpty(Validator[pd.Series | pd.DataFrame | pd.Index]):
  """Validator for non-empty data."""

  @override
  def validate(
    self, data: pd.Series | pd.DataFrame | pd.Index
  ) -> pd.Series | pd.DataFrame | pd.Index:
    if isinstance(data, (pd.Series, pd.DataFrame, pd.Index)) and data.empty:  # pyright: ignore[reportUnnecessaryIsInstance]
      raise ValueError("Data must not be empty")
    return data


class NonNaN(Validator[pd.Series | pd.DataFrame | pd.Index]):
  """Validator for non-NaN values.

  Uses pd.isna() for compatibility with all dtypes including object columns.
  """

  @override
  def validate(
    self, data: pd.Series | pd.DataFrame | pd.Index
  ) -> pd.Series | pd.DataFrame | pd.Index:
    # axis=None returns scalar bool, pandas-stubs typing issue
    if isinstance(data, (pd.Series, pd.DataFrame, pd.Index)) and data.isna().any(  # pyright: ignore[reportUnnecessaryIsInstance,reportGeneralTypeIssues]
      axis=None  # pyright: ignore[reportArgumentType,reportGeneralTypeIssues]
    ):
      raise ValueError("Data must not contain NaN values")
    return data


class NonNegative(Validator[pd.Series | pd.DataFrame | pd.Index]):
  """Validator for non-negative values (>= 0)."""

  @override
  def validate(
    self, data: pd.Series | pd.DataFrame | pd.Index
  ) -> pd.Series | pd.DataFrame | pd.Index:
    if isinstance(data, (pd.Series, pd.DataFrame, pd.Index)) and np.any(  # pyright: ignore[reportUnnecessaryIsInstance]
      data.values < 0  # pyright: ignore[reportOperatorIssue,reportUnknownArgumentType]
    ):
      raise ValueError("Data must be non-negative")
    return data


class Positive(Validator[pd.Series | pd.DataFrame | pd.Index]):
  """Validator for positive values (> 0)."""

  @override
  def validate(
    self, data: pd.Series | pd.DataFrame | pd.Index
  ) -> pd.Series | pd.DataFrame | pd.Index:
    if isinstance(data, (pd.Series, pd.DataFrame, pd.Index)) and np.any(  # pyright: ignore[reportUnnecessaryIsInstance]
      data.values <= 0  # pyright: ignore[reportOperatorIssue,reportUnknownArgumentType]
    ):
      raise ValueError("Data must be positive")
    return data


class OneOf(Validator[pd.Series | pd.Index]):
  """Validator for categorical values - ensures all values are in allowed set.

  Supports multiple syntax forms:
  - OneOf[Literal["a", "b", "c"]]
  - OneOf["a", "b", "c"]
  - OneOf[Literal["a"], Literal["b"], Literal["c"]]

  Can be used with Index[] wrapper for index validation:
  - Index[OneOf["x", "y", "z"]]

  Can be used with HasColumn for column-specific validation:
  - HasColumn["category", OneOf["a", "b", "c"]]
  """

  def __init__(self, *allowed: object) -> None:
    super().__init__()
    self.allowed = set(allowed)

  @override
  def validate(self, data: pd.Series | pd.Index) -> pd.Series | pd.Index:
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
    return data


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
  """Parse a dimension constraint from Shape[] arguments.

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
  - Shape[10, 5] - Exactly 10 rows, 5 columns
  - Shape[Ge[10], Any] - At least 10 rows, any columns
  - Shape[Any, Le[5]] - Any rows, at most 5 columns
  - Shape[Gt[0], Lt[100]] - More than 0 rows, less than 100 columns
  - Shape[100] - For Series: exactly 100 rows

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
  def validate(
    self, data: pd.Series | pd.DataFrame | pd.Index
  ) -> pd.Series | pd.DataFrame | pd.Index:
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
    return data
