"""Column validators for DataFrames."""

from __future__ import annotations

import typing
from typing import TYPE_CHECKING, Any, override

import numpy as np
import pandas as pd

from validated.base import Validator
from validated.utils import apply_default_validators, instantiate_validator

if TYPE_CHECKING:
  from validated.base import ValidatorMarker


class IsDtype(Validator[pd.Series | pd.DataFrame]):
  """Validator for specific dtype."""

  def __init__(self, dtype: str | type | np.dtype) -> None:
    super().__init__()
    self.dtype = np.dtype(dtype)

  @override
  def validate(self, data: pd.Series | pd.DataFrame) -> pd.Series | pd.DataFrame:
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
    return data


class HasColumns(Validator[pd.DataFrame]):
  """Validator for presence of specific columns in DataFrame.

  Can also apply validators to the specified columns:
  HasColumns["a", "b", Finite, Positive]
  """

  def __init__(
    self,
    columns: list[str],
    *validators: Validator[Any] | ValidatorMarker,
  ) -> None:
    super().__init__()
    self.columns = columns

    instantiated: list[Validator[Any] | ValidatorMarker] = []  # pyright: ignore[reportExplicitAny]
    for v_item in validators:
      v = instantiate_validator(v_item)
      if v:
        instantiated.append(v)
    self.validators = tuple(instantiated)

  @override
  def validate(self, data: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(data, pd.DataFrame):
      raise TypeError("HasColumns validator requires a pandas DataFrame")

    missing = [col for col in self.columns if col not in data.columns]
    if missing:
      raise ValueError(f"Missing columns: {missing}")

    # Determine validators to apply (filters markers and adds defaults)
    final_validators = apply_default_validators(self.validators or [])

    if final_validators:
      for col in self.columns:
        column_data = data[col]  # type: ignore[union-attr]
        for v in final_validators:
          column_data = v.validate(column_data)  # type: ignore[assignment]

    return data


class HasColumn(Validator[pd.DataFrame]):
  """Wrapper to apply validators to specific DataFrame columns.

  Supports templating:
  T = TypeVar("T")
  CustomVal = HasColumn[T, Positive]
  CustomVal["my_col"]  # Creates HasColumn("my_col", Positive)
  """

  def __init__(
    self,
    column: str | typing.TypeVar,
    *validators: Validator[Any]  # type: ignore[misc]
    | ValidatorMarker
    | type[Validator[Any]]  # type: ignore[misc]
    | type[ValidatorMarker],
  ) -> None:
    super().__init__()
    self.column = column
    self.validators = validators

  def __getitem__(self, item: str) -> HasColumn:
    """Support for templating: CustomVal["col"]."""
    return HasColumn(item, *self.validators)

  @override
  def validate(self, data: pd.DataFrame) -> pd.DataFrame:
    if isinstance(data, pd.DataFrame):
      if self.column not in data.columns:
        raise ValueError(f"Column '{self.column}' not found in DataFrame")

      # Extract the column as a Series
      column_data = data[self.column]

      # Instantiate validators from types if necessary
      instantiated_validators: list[Validator[Any] | ValidatorMarker] = []  # pyright: ignore[reportExplicitAny]
      for validator_item in self.validators:
        v = instantiate_validator(validator_item)
        if v:
          instantiated_validators.append(v)

      # Determine validators to apply (filters markers and adds defaults)
      final_validators = apply_default_validators(instantiated_validators)

      # Apply each validator
      for validator in final_validators:
        column_data = validator.validate(column_data)

    return data
