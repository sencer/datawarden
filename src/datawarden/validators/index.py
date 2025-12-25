"""Index and monotonicity validators."""

from __future__ import annotations

from typing import Any, override

import pandas as pd

from datawarden.base import Validator
from datawarden.utils import instantiate_validator


class Datetime(Validator[pd.Series | pd.Index]):
  """Validator for datetime data.

  Behavior depends on input type:
  - pd.Index: Validates that the Index is a DatetimeIndex
  - pd.Series: Validates that the Series *values* are datetime64 dtype

  For validating DataFrame/Series index, use with Index() wrapper:
    Index(Datetime) - validates the index is a DatetimeIndex

  Example:
    # Validate Series contains datetime values
    data: Validated[pd.Series, Datetime]

    # Validate DataFrame has datetime index
    data: Validated[pd.DataFrame, Index(Datetime)]
  """

  @override
  def validate(self, data: pd.Series | pd.Index) -> None:
    if isinstance(data, pd.Index) and not isinstance(data, pd.DatetimeIndex):
      # Direct Index input - check it's a DatetimeIndex
      raise ValueError("Index must be DatetimeIndex")
    if isinstance(data, pd.Series) and not pd.api.types.is_datetime64_any_dtype(
      data.dtype
    ):
      # Series input - check VALUES are datetime dtype (not the index)
      raise ValueError(f"Series values must be datetime64 dtype, got {data.dtype}")


class Unique(Validator[pd.Series | pd.Index]):
  """Validator for unique values.

  Use with Index(Unique) to apply to Series/DataFrame index.
  Can be applied directly to pd.Index or pd.Series values.
  """

  @override
  def validate(self, data: pd.Series | pd.Index) -> None:
    if isinstance(data, pd.Index) and not data.is_unique:
      raise ValueError("Values must be unique")
    if isinstance(data, pd.Series) and not data.is_unique:
      raise ValueError("Values must be unique")


class MonoUp(Validator[pd.Series | pd.Index]):
  """Validator for monotonically increasing values.

  Use with Index(MonoUp) to apply to Series/DataFrame index.
  Can be applied directly to pd.Index or pd.Series values.
  """

  @override
  def validate(self, data: pd.Series | pd.Index) -> None:
    if isinstance(data, pd.Index) and not data.is_monotonic_increasing:
      raise ValueError("Values must be monotonically increasing")
    if isinstance(data, pd.Series) and not data.is_monotonic_increasing:
      raise ValueError("Values must be monotonically increasing")


class MonoDown(Validator[pd.Series | pd.Index]):
  """Validator for monotonically decreasing values.

  Use with Index(MonoDown) to apply to Series/DataFrame index.
  Can be applied directly to pd.Index or pd.Series values.
  """

  @override
  def validate(self, data: pd.Series | pd.Index) -> None:
    if isinstance(data, pd.Index) and not data.is_monotonic_decreasing:
      raise ValueError("Values must be monotonically decreasing")
    if isinstance(data, pd.Series) and not data.is_monotonic_decreasing:
      raise ValueError("Values must be monotonically decreasing")


class Index(Validator[pd.Series | pd.DataFrame | pd.Index]):
  """Validator for index properties.

  Can be used to apply validators to the index:
  - Index(Datetime) - Check index is DatetimeIndex
  - Index(MonoUp) - Check index is monotonically increasing
  - Index(Datetime, MonoUp) - Check both
  """

  def __init__(
    self,
    *validators: Validator[Any] | type[Validator[Any]],  # type: ignore[misc]
  ) -> None:
    super().__init__()
    instantiated: list[Validator[Any]] = []  # pyright: ignore[reportExplicitAny]
    for v_item in validators:
      v = instantiate_validator(v_item)
      if v:
        instantiated.append(v)
    self.validators = tuple(instantiated)

  @override
  def validate(self, data: pd.Series | pd.DataFrame | pd.Index) -> None:
    if not isinstance(data, (pd.Series, pd.DataFrame, pd.Index)):
      raise TypeError(
        f"Index requires pandas Series, DataFrame, or Index, got {type(data).__name__}"
      )
    target_index = data

    if isinstance(data, (pd.Series, pd.DataFrame)):
      target_index = data.index

    # Apply each validator to the index
    for validator in self.validators:
      validator.validate(target_index)
