"""Comparison validators for Series and DataFrame data."""

from __future__ import annotations

from typing import override

import numpy as np
import pandas as pd

from validated.base import Validator


class Ge(Validator[pd.Series | pd.DataFrame | pd.Index]):
  """Validator that data >= target (unary) or col1 >= col2 >= ... (n-ary)."""

  def __init__(self, *targets: str | float | int | None) -> None:
    super().__init__()
    self.targets = targets

  def check(self, value: int) -> bool:
    """Check constraint for a scalar integer (used by Shape validator)."""
    if len(self.targets) == 1 and isinstance(self.targets[0], (int, float)):
      return value >= self.targets[0]
    return False

  def describe(self) -> str:
    """Describe the constraint."""
    if len(self.targets) == 1:
      return f">= {self.targets[0]}"
    return f"Ge({self.targets})"

  @override
  def validate(
    self, data: pd.Series | pd.DataFrame | pd.Index
  ) -> pd.Series | pd.DataFrame | pd.Index:
    if len(self.targets) == 1:
      # Unary comparison
      target = self.targets[0]
      if isinstance(data, (pd.Series, pd.DataFrame, pd.Index)) and np.any(
        data.values < target
      ):
        raise ValueError(f"Data must be >= {target}")
    else:
      # Column comparison
      if not isinstance(data, pd.DataFrame):
        raise TypeError("Column comparison requires a pandas DataFrame")

      for i in range(len(self.targets) - 1):
        col1 = self.targets[i]
        col2 = self.targets[i + 1]

        if not isinstance(col1, str) or not isinstance(col2, str):
          raise TypeError("Column comparison requires string column names")

        if (
          col1 in data.columns
          and col2 in data.columns
          and np.any(data[col1].values < data[col2].values)
        ):
          raise ValueError(f"{col1} must be >= {col2}")

    return data


class Le(Validator[pd.Series | pd.DataFrame | pd.Index]):
  """Validator that data <= target (unary) or col1 <= col2 <= ... (n-ary)."""

  def __init__(self, *targets: str | float | int | None) -> None:
    super().__init__()
    self.targets = targets

  def check(self, value: int) -> bool:
    """Check constraint for a scalar integer (used by Shape validator)."""
    if len(self.targets) == 1 and isinstance(self.targets[0], (int, float)):
      return value <= self.targets[0]
    return False

  def describe(self) -> str:
    """Describe the constraint."""
    if len(self.targets) == 1:
      return f"<= {self.targets[0]}"
    return f"Le({self.targets})"

  @override
  def validate(
    self, data: pd.Series | pd.DataFrame | pd.Index
  ) -> pd.Series | pd.DataFrame | pd.Index:
    if len(self.targets) == 1:
      # Unary comparison
      target = self.targets[0]
      if isinstance(data, (pd.Series, pd.DataFrame, pd.Index)) and np.any(
        data.values > target
      ):
        raise ValueError(f"Data must be <= {target}")
    else:
      # Column comparison
      if not isinstance(data, pd.DataFrame):
        raise TypeError("Column comparison requires a pandas DataFrame")

      for i in range(len(self.targets) - 1):
        col1 = self.targets[i]
        col2 = self.targets[i + 1]

        if not isinstance(col1, str) or not isinstance(col2, str):
          raise TypeError("Column comparison requires string column names")

        if (
          col1 in data.columns
          and col2 in data.columns
          and np.any(data[col1].values > data[col2].values)
        ):
          raise ValueError(f"{col1} must be <= {col2}")

    return data


class Gt(Validator[pd.Series | pd.DataFrame | pd.Index]):
  """Validator that data > target (unary) or col1 > col2 > ... (n-ary)."""

  def __init__(self, *targets: str | float | int | None) -> None:
    super().__init__()
    self.targets = targets

  def check(self, value: int) -> bool:
    """Check constraint for a scalar integer (used by Shape validator)."""
    if len(self.targets) == 1 and isinstance(self.targets[0], (int, float)):
      return value > self.targets[0]
    return False

  def describe(self) -> str:
    """Describe the constraint."""
    if len(self.targets) == 1:
      return f"> {self.targets[0]}"
    return f"Gt({self.targets})"

  @override
  def validate(
    self, data: pd.Series | pd.DataFrame | pd.Index
  ) -> pd.Series | pd.DataFrame | pd.Index:
    if len(self.targets) == 1:
      # Unary comparison
      target = self.targets[0]
      if isinstance(data, (pd.Series, pd.DataFrame, pd.Index)) and np.any(
        data.values <= target
      ):
        raise ValueError(f"Data must be > {target}")
    else:
      # Column comparison
      if not isinstance(data, pd.DataFrame):
        raise TypeError("Column comparison requires a pandas DataFrame")

      for i in range(len(self.targets) - 1):
        col1 = self.targets[i]
        col2 = self.targets[i + 1]

        if not isinstance(col1, str) or not isinstance(col2, str):
          raise TypeError("Column comparison requires string column names")

        if (
          col1 in data.columns
          and col2 in data.columns
          and np.any(data[col1].values <= data[col2].values)
        ):
          raise ValueError(f"{col1} must be > {col2}")

    return data


class Lt(Validator[pd.Series | pd.DataFrame | pd.Index]):
  """Validator that data < target (unary) or col1 < col2 < ... (n-ary)."""

  def __init__(self, *targets: str | float | int | None) -> None:
    super().__init__()
    self.targets = targets

  def check(self, value: int) -> bool:
    """Check constraint for a scalar integer (used by Shape validator)."""
    if len(self.targets) == 1 and isinstance(self.targets[0], (int, float)):
      return value < self.targets[0]
    return False

  def describe(self) -> str:
    """Describe the constraint."""
    if len(self.targets) == 1:
      return f"< {self.targets[0]}"
    return f"Lt({self.targets})"

  @override
  def validate(
    self, data: pd.Series | pd.DataFrame | pd.Index
  ) -> pd.Series | pd.DataFrame | pd.Index:
    if len(self.targets) == 1:
      # Unary comparison
      target = self.targets[0]
      if isinstance(data, (pd.Series, pd.DataFrame, pd.Index)) and np.any(
        data.values >= target
      ):
        raise ValueError(f"Data must be < {target}")
    else:
      # Column comparison
      if not isinstance(data, pd.DataFrame):
        raise TypeError("Column comparison requires a pandas DataFrame")

      for i in range(len(self.targets) - 1):
        col1 = self.targets[i]
        col2 = self.targets[i + 1]

        if not isinstance(col1, str) or not isinstance(col2, str):
          raise TypeError("Column comparison requires string column names")

        if (
          col1 in data.columns
          and col2 in data.columns
          and np.any(data[col1].values >= data[col2].values)
        ):
          raise ValueError(f"{col1} must be < {col2}")

    return data
