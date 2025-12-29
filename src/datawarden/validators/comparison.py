"""Comparison validators for Series and DataFrame data."""

from __future__ import annotations

import operator
from typing import TYPE_CHECKING, Any, cast, override

import numpy as np
import pandas as pd

if TYPE_CHECKING:
  from collections.abc import Callable
from datawarden.base import Validator
from datawarden.utils import report_failures, scalar_any


class _ComparisonValidator(Validator[pd.Series | pd.DataFrame | pd.Index]):
  """Base class for comparison validators (Ge, Le, Gt, Lt).

  Eliminates code duplication by providing common validation logic.
  """

  # Subclasses must define these
  op_symbol: str = ""  # pyright: ignore[reportUninitializedInstanceVariable]
  op_func: Callable[[float, float], bool]  # pyright: ignore[reportUninitializedInstanceVariable]
  opposite_op_func: Callable[[float, float], bool]  # pyright: ignore[reportUninitializedInstanceVariable]

  def __init__(
    self,
    *targets: str | float | int | None,
    ignore_nan: bool = False,
  ) -> None:
    super().__init__()
    self.targets = targets
    self.ignore_nan = ignore_nan

  @override
  def __repr__(self) -> str:
    args = [repr(t) for t in self.targets]
    if self.ignore_nan:
      args.append("ignore_nan=True")
    args_str = ", ".join(args)
    return f"{self.__class__.__name__}({args_str})"

  def check(self, value: int) -> bool:
    """Check constraint for a scalar integer (used by Shape validator)."""
    if len(self.targets) == 1 and isinstance(self.targets[0], (int, float)):
      return bool((self.op_func)(value, self.targets[0]))  # type: ignore
    return False

  def describe(self) -> str:
    """Describe the constraint."""
    if len(self.targets) == 1:
      return f"{self.op_symbol} {self.targets[0]}"
    return f"{self.__class__.__name__}({self.targets})"

  def with_ignore_nan(self) -> _ComparisonValidator:
    """Return a copy of this validator that ignores NaN values."""
    return self.__class__(*self.targets, ignore_nan=True)

  @override
  def validate(self, data: pd.Series | pd.DataFrame | pd.Index) -> None:
    # Check for NaN values before comparison
    if not self.ignore_nan and np.any(mask_nan := pd.isna(data.values)):  # pyright: ignore
      report_failures(
        data,
        mask_nan,  # pyright: ignore
        f"Cannot perform {self.op_symbol} comparison with NaN values (use IgnoringNaNs wrapper to skip NaN values)",
      )

    if len(self.targets) == 1:
      # Unary comparison: data op target
      target = self.targets[0]
      if isinstance(data, (pd.Series, pd.DataFrame, pd.Index)):
        # Use cast to Any to avoid stub mismatch with operator functions
        mask = (self.opposite_op_func)(data.values, target)  # type: ignore
        if scalar_any(cast("Any", mask)):
          report_failures(
            data, cast("Any", mask), f"Data must be {self.op_symbol} {target}"
          )
    else:
      # Column comparison: col1 op col2 op col3 ...
      if not isinstance(data, pd.DataFrame):
        raise TypeError("Column comparison requires a pandas DataFrame")

      for i in range(len(self.targets) - 1):
        col1 = self.targets[i]
        col2 = self.targets[i + 1]

        if not isinstance(col1, str) or not isinstance(col2, str):
          raise TypeError("Column comparison requires string column names")

        # Check columns exist - fail explicitly rather than silently passing
        missing = [c for c in (col1, col2) if c not in data.columns]
        if missing:
          raise ValueError(f"Missing columns for comparison: {missing}")

        col1_vals = data[col1].values
        col2_vals = data[col2].values
        mask = (self.opposite_op_func)(col1_vals, col2_vals)  # type: ignore
        if scalar_any(cast("Any", mask)):
          report_failures(
            data, cast("Any", mask), f"{col1} must be {self.op_symbol} {col2}"
          )


class Ge(_ComparisonValidator):
  """Validator that data >= target (unary) or col1 >= col2 >= ... (n-ary).

  Two modes of operation:

  **Unary mode** - Compare all values against a scalar:
    Ge(5) means all values must be >= 5

  **N-ary mode** - Compare DataFrame columns pairwise:
    Ge("high", "low") means high >= low for all rows
    Ge("a", "b", "c") means a >= b >= c for all rows

  Examples:
    # Series: all values >= 0
    data: Validated[pd.Series, Ge(0)]

    # DataFrame: ensure high >= low
    data: Validated[pd.DataFrame, Ge("high", "low")]

  Note:
    Rejects NaN values by default. Use IgnoringNaNs(Ge(...)) to skip NaN.
  """

  op_symbol = ">="
  op_func = operator.ge
  opposite_op_func = operator.lt  # Check for violations (< instead of >=)


class Le(_ComparisonValidator):
  """Validator that data <= target (unary) or col1 <= col2 <= ... (n-ary).

  Two modes of operation:

  **Unary mode** - Compare all values against a scalar:
    Le(100) means all values must be <= 100

  **N-ary mode** - Compare DataFrame columns pairwise:
    Le("low", "high") means low <= high for all rows

  Examples:
    # Series: all values <= 100
    data: Validated[pd.Series, Le(100)]

    # DataFrame: ensure low <= mid <= high
    data: Validated[pd.DataFrame, Le("low", "mid", "high")]

  Note:
    Rejects NaN values by default. Use IgnoringNaNs(Le(...)) to skip NaN.
  """

  op_symbol = "<="
  op_func = operator.le
  opposite_op_func = operator.gt  # Check for violations (> instead of <=)


class Gt(_ComparisonValidator):
  """Validator that data > target (unary) or col1 > col2 > ... (n-ary).

  Two modes of operation:

  **Unary mode** - Compare all values against a scalar:
    Gt(0) means all values must be > 0 (strictly positive)

  **N-ary mode** - Compare DataFrame columns pairwise:
    Gt("high", "low") means high > low for all rows (strict inequality)

  Examples:
    # Series: all values > 0
    data: Validated[pd.Series, Gt(0)]

    # DataFrame columns strictly ordered
    data: Validated[pd.DataFrame, Gt("high", "low")]

  Note:
    Rejects NaN values by default. Use IgnoringNaNs(Gt(...)) to skip NaN.
  """

  op_symbol = ">"
  op_func = operator.gt
  opposite_op_func = operator.le  # Check for violations (<= instead of >)


class Lt(_ComparisonValidator):
  """Validator that data < target (unary) or col1 < col2 < ... (n-ary).

  Two modes of operation:

  **Unary mode** - Compare all values against a scalar:
    Lt(100) means all values must be < 100 (strictly less)

  **N-ary mode** - Compare DataFrame columns pairwise:
    Lt("low", "high") means low < high for all rows (strict inequality)

  Examples:
    # Series: all values < 100
    data: Validated[pd.Series, Lt(100)]

    # DataFrame columns strictly ordered
    data: Validated[pd.DataFrame, Lt("low", "high")]

  Note:
    Rejects NaN values by default. Use IgnoringNaNs(Lt(...)) to skip NaN.
  """

  op_symbol = "<"
  op_func = operator.lt
  opposite_op_func = operator.ge  # Check for violations (>= instead of <)
