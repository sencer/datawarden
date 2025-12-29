"""Time gap and numeric difference validators."""

from __future__ import annotations

from typing import override

import numpy as np
import pandas as pd

from datawarden.base import Validator


def _get_datetime_array(
  data: pd.Series | pd.Index,
) -> np.ndarray | None:
  """Extract datetime values as int64 nanoseconds array.

  Returns None if data is not datetime type.
  """
  if isinstance(data, pd.DatetimeIndex):
    return np.asarray(data.view(np.int64))
  if isinstance(data, pd.Series) and pd.api.types.is_datetime64_any_dtype(data.dtype):
    return np.asarray(data.values.view(np.int64))  # type: ignore[union-attr]
  return None


class NoTimeGaps(Validator[pd.Series | pd.Index]):
  """Validator for no time gaps in datetime data.

  Supports Chunked Validation (Stateful):
  Maintains the last timestamp of the previous chunk to ensure no gaps exist
  across chunk boundaries.

  Works on:
  - pd.Series with datetime64 dtype (validates values)
  - pd.DatetimeIndex (validates index, use with Index(NoTimeGaps(...)))

  Example:
    # Validate datetime column values
    data: Validated[pd.Series, NoTimeGaps("1min")]

    # Validate index via Index wrapper
    data: Validated[pd.DataFrame, Index(NoTimeGaps("1min"))]
  """

  is_chunkable = True

  def __init__(self, freq: str) -> None:
    super().__init__()
    self.freq = freq
    # Pre-compute expected nanoseconds for performance
    self._expected_ns: int = pd.Timedelta(pd.tseries.frequencies.to_offset(freq)).value  # type: ignore[arg-type]
    self._last_val: int | None = None

  @override
  def reset(self) -> None:
    self._last_val = None

  @override
  def __repr__(self) -> str:
    return f"NoTimeGaps({self.freq!r})"

  @override
  def validate(self, data: pd.Series | pd.Index) -> None:
    dt_array = _get_datetime_array(data)

    if dt_array is None:
      raise ValueError(
        "NoTimeGaps requires datetime data (datetime64 Series or DatetimeIndex)"
      )

    if len(dt_array) == 0:
      return

    # Check gap from previous chunk
    if self._last_val is not None:
      actual_gap = int(dt_array[0]) - self._last_val
      if actual_gap != self._expected_ns:
        raise ValueError(f"Time gaps detected with frequency '{self.freq}'")

    if len(dt_array) > 1:
      # Fully vectorized: numpy diff on underlying int64 nanoseconds
      actual_diffs_ns = np.diff(dt_array)

      if not np.all(actual_diffs_ns == self._expected_ns):
        raise ValueError(f"Time gaps detected with frequency '{self.freq}'")

    # Store last value
    self._last_val = int(dt_array[-1])


class MaxGap(Validator[pd.Series | pd.Index]):
  """Validator for maximum allowed time gap in datetime data.

  Supports Chunked Validation (Stateful):
  Maintains the last timestamp of the previous chunk to ensure gaps across
  chunk boundaries do not exceed the limit.

  Unlike NoTimeGaps which requires exact frequency, MaxGap allows gaps
  up to (and including) the specified duration. Useful for data with
  occasional missing values.

  Works on:
  - pd.Series with datetime64 dtype (validates values)
  - pd.DatetimeIndex (validates index, use with Index(MaxGap(...)))

  Example:
    # Allow up to 2-minute gaps in 1-minute data (tolerates 1 missing row)
    data: Validated[pd.Series, MaxGap("2min")]

    # Validate index via Index wrapper
    data: Validated[pd.DataFrame, Index(MaxGap("5min"))]
  """

  is_chunkable = True

  def __init__(self, max_gap: str) -> None:
    super().__init__()
    self.max_gap = max_gap
    # Pre-compute max gap nanoseconds for performance
    self._max_gap_ns: int = pd.Timedelta(
      pd.tseries.frequencies.to_offset(max_gap)
    ).value  # type: ignore[arg-type]
    self._last_val: int | None = None

  @override
  def reset(self) -> None:
    self._last_val = None

  @override
  def __repr__(self) -> str:
    return f"MaxGap({self.max_gap!r})"

  @override
  def validate(self, data: pd.Series | pd.Index) -> None:
    dt_array = _get_datetime_array(data)

    if dt_array is None:
      raise ValueError(
        "MaxGap requires datetime data (datetime64 Series or DatetimeIndex)"
      )

    if len(dt_array) == 0:
      return

    # Check gap from previous chunk
    if self._last_val is not None:
      actual_gap = int(dt_array[0]) - self._last_val
      if actual_gap > self._max_gap_ns:
        max_found = pd.Timedelta(nanoseconds=int(actual_gap))
        raise ValueError(
          f"Time gap exceeds maximum '{self.max_gap}' (found gap of {max_found})"
        )

    if len(dt_array) > 1:
      # Fully vectorized: numpy diff on underlying int64 nanoseconds
      actual_diffs_ns = np.diff(dt_array)

      if np.any(actual_diffs_ns > self._max_gap_ns):
        max_found = pd.Timedelta(nanoseconds=int(np.max(actual_diffs_ns)))
        raise ValueError(
          f"Time gap exceeds maximum '{self.max_gap}' (found gap of {max_found})"
        )

    # Store last value
    self._last_val = int(dt_array[-1])


class MaxDiff(Validator[pd.Series]):
  """Validator for maximum allowed difference between consecutive numeric values.

  Supports Chunked Validation (Stateful):
  Maintains the last value of the previous chunk to ensure jumps across
  chunk boundaries do not exceed the limit.

  Useful for validating that numeric data doesn't have large jumps.

  Example:
    # Price changes must be at most 10
    data: Validated[pd.Series, MaxDiff(10.0)]

    # Validate specific column
    data: Validated[pd.DataFrame, HasColumn("price", MaxDiff(5.0))]
  """

  is_chunkable = True

  def __init__(self, max_diff: float | int) -> None:
    super().__init__()
    self.max_diff = max_diff
    self._last_val: float | int | None = None

  @override
  def reset(self) -> None:
    self._last_val = None

  @override
  def __repr__(self) -> str:
    return f"MaxDiff({self.max_diff!r})"

  @override
  def validate(self, data: pd.Series) -> None:
    if not isinstance(data, pd.Series):
      raise TypeError("MaxDiff requires a pandas Series")

    if not pd.api.types.is_numeric_dtype(data.dtype):
      raise ValueError("MaxDiff requires numeric data")

    if len(data) == 0:
      return

    # Check diff from previous chunk
    if self._last_val is not None:
      actual_diff = float(np.abs(data.iloc[0] - self._last_val))
      if actual_diff > self.max_diff:
        raise ValueError(
          f"Difference exceeds maximum {self.max_diff} (found diff of {actual_diff})"
        )

    if len(data) > 1:
      # Use absolute difference for numeric data
      diffs = np.abs(np.diff(data.values))  # type: ignore[arg-type]

      if np.any(diffs > self.max_diff):
        max_found = float(np.max(diffs))
        raise ValueError(
          f"Difference exceeds maximum {self.max_diff} (found diff of {max_found})"
        )

    # Store last value
    self._last_val = data.iloc[-1]
