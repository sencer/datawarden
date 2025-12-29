"""Column validators for DataFrames."""

from __future__ import annotations

from typing import Any, override

import numpy as np
import pandas as pd

from datawarden.base import Validator
from datawarden.utils import instantiate_validator


class IsDtype(Validator[pd.Series | pd.DataFrame]):
  """Validator for specific dtype."""

  def __init__(self, dtype: str | type | np.dtype) -> None:
    super().__init__()
    self.dtype = np.dtype(dtype)

  @override
  def validate(self, data: pd.Series | pd.DataFrame) -> None:
    if isinstance(data, pd.Series):
      if data.dtype != self.dtype:
        raise ValueError(f"Data must be of type {self.dtype}, got {data.dtype}")
    elif isinstance(data, pd.DataFrame):
      # Vectorized dtype check: compare all column dtypes at once
      mismatches = data.dtypes != self.dtype
      if mismatches.any():
        bad_cols = mismatches[mismatches].index.tolist()
        bad_dtypes = data.dtypes[mismatches].tolist()
        msg = f"Columns with wrong dtype (expected {self.dtype}): {dict(zip(bad_cols, bad_dtypes, strict=True))}"
        raise ValueError(msg)


class HasColumns(Validator[pd.DataFrame]):
  """Validator for presence of specific columns in DataFrame.

  Can also apply validators to the specified columns:
  HasColumns(["a", "b"], Finite, Positive)
  """

  def __init__(
    self,
    columns: list[str],
    *validators: Validator[Any],
  ) -> None:
    super().__init__()
    self.columns = columns

    instantiated: list[Validator[Any]] = []
    for v_item in validators:
      v = instantiate_validator(v_item)
      if v:
        instantiated.append(v)
    self.validators = tuple(instantiated)

  @property
  @override
  def is_chunkable(self) -> bool:
    """Only chunkable if ALL inner validators are chunkable."""
    return all(getattr(v, "is_chunkable", True) for v in self.validators)

  @override
  def validate(self, data: pd.DataFrame) -> None:
    if not isinstance(data, pd.DataFrame):
      raise TypeError("HasColumns validator requires a pandas DataFrame")

    missing = [col for col in self.columns if col not in data.columns]
    if missing:
      raise ValueError(f"Missing columns: {missing}")

    # Apply validators to columns
    final_validators = list(self.validators)

    if final_validators:
      for col in self.columns:
        column_data = data[col]
        for v in final_validators:
          v.validate(column_data)


class HasColumn(Validator[pd.DataFrame]):
  """Wrapper to apply validators to specific DataFrame columns.

  Example:
  HasColumn("my_col", Positive)
  """

  def __init__(
    self,
    column: str,
    *validators: Validator[Any] | type[Validator[Any]],
  ) -> None:
    super().__init__()
    self.column = column

    # Instantiate validators at construction time (unify with HasColumns)
    instantiated: list[Validator[Any]] = []
    for v_item in validators:
      v = instantiate_validator(v_item)
      if v:
        instantiated.append(v)
    self.validators = tuple(instantiated)

  @property
  @override
  def is_chunkable(self) -> bool:
    """Only chunkable if ALL inner validators are chunkable."""
    return all(getattr(v, "is_chunkable", True) for v in self.validators)

  @override
  def validate(self, data: pd.DataFrame) -> None:
    if not isinstance(data, pd.DataFrame):
      raise TypeError(f"HasColumn requires pandas DataFrame, got {type(data).__name__}")
    if self.column not in data.columns:
      raise ValueError(f"Column '{self.column}' not found in DataFrame")

    # Extract the column as a Series
    column_data = data[self.column]

    # Apply each validator
    for validator in self.validators:
      validator.validate(column_data)
