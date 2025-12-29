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

  @property
  @override
  def is_chunkable(self) -> bool:
    return False

  @override
  def validate(self, data: pd.Series | pd.Index) -> None:
    if isinstance(data, pd.Index) and not data.is_unique:
      raise ValueError("Values must be unique")
    if isinstance(data, pd.Series) and not data.is_unique:
      raise ValueError("Values must be unique")


class MonoUp(Validator[pd.Series | pd.Index]):
  """Validator for monotonically increasing values.

  Supports Chunked Validation (Stateful):
  When processing data in chunks, this validator maintains state (last value)
  to ensure monotonicity is preserved across chunk boundaries.

  Use with Index(MonoUp) to apply to Series/DataFrame index.
  Can be applied directly to pd.Index or pd.Series values.
  """

  def __init__(self) -> None:
    super().__init__()
    self._last_val: Any = None

  @override
  def reset(self) -> None:
    self._last_val = None

  @override
  def validate(self, data: pd.Series | pd.Index) -> None:
    if len(data) == 0:
      return

    # 1. Check if first element is >= last element of previous chunk
    if self._last_val is not None:
      # Use index[0] for Index, or iloc[0] for Series
      first_val = data[0] if isinstance(data, pd.Index) else data.iloc[0]
      if first_val < self._last_val:
        raise ValueError(
          f"Monotonicity broken: first value of chunk ({first_val}) is less than last value of previous chunk ({self._last_val})"
        )

    # 2. Check monotonicity within chunk
    if isinstance(data, pd.Index) and not data.is_monotonic_increasing:
      raise ValueError("Values must be monotonically increasing")
    if isinstance(data, pd.Series) and not data.is_monotonic_increasing:
      raise ValueError("Values must be monotonically increasing")

    # 3. Store last value for next chunk
    self._last_val = data[-1] if isinstance(data, pd.Index) else data.iloc[-1]


class MonoDown(Validator[pd.Series | pd.Index]):
  """Validator for monotonically decreasing values.

  Supports Chunked Validation (Stateful):
  When processing data in chunks, this validator maintains state (last value)
  to ensure monotonicity is preserved across chunk boundaries.

  Use with Index(MonoDown) to apply to Series/DataFrame index.
  Can be applied directly to pd.Index or pd.Series values.
  """

  def __init__(self) -> None:
    super().__init__()
    self._last_val: Any = None

  @override
  def reset(self) -> None:
    self._last_val = None

  @override
  def validate(self, data: pd.Series | pd.Index) -> None:
    if len(data) == 0:
      return

    # 1. Check if first element is <= last element of previous chunk
    if self._last_val is not None:
      first_val = data[0] if isinstance(data, pd.Index) else data.iloc[0]
      if first_val > self._last_val:
        raise ValueError(
          f"Monotonicity broken: first value of chunk ({first_val}) is greater than last value of previous chunk ({self._last_val})"
        )

    # 2. Check monotonicity within chunk
    if isinstance(data, pd.Index) and not data.is_monotonic_decreasing:
      raise ValueError("Values must be monotonically decreasing")
    if isinstance(data, pd.Series) and not data.is_monotonic_decreasing:
      raise ValueError("Values must be monotonically decreasing")

    # 3. Store last value for next chunk
    self._last_val = data[-1] if isinstance(data, pd.Index) else data.iloc[-1]


class Index(Validator[pd.Series | pd.DataFrame | pd.Index]):
  """Validator for index properties.

  Chunking Support:
  - This wrapper is chunkable ONLY if all internal validators are chunkable.
  - E.g., Index(MonoUp) IS chunkable.
  - E.g., Index(Unique) IS NOT chunkable.

  Can be used to apply validators to the index:
  - Index(Datetime) - Check index is DatetimeIndex
  - Index(MonoUp) - Check index is monotonically increasing
  - Index(Datetime, MonoUp) - Check both
  """

  def __init__(
    self,
    *validators: Validator[Any] | type[Validator[Any]],
  ) -> None:
    super().__init__()
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
  def reset(self) -> None:
    """Reset all inner validators."""
    for v in self.validators:
      v.reset()

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
