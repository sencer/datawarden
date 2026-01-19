"""Tests for sequence validators: NoTimeGaps, MaxGap, MaxDiff."""

import numpy as np
import pandas as pd
import pytest

from datawarden import Validated, validate
from datawarden.exceptions import ValidationError
from datawarden.validators import (
  Column,
  Index,
  IsNaN,
  MaxDiff,
  MaxGap,
  NoTimeGaps,
)
from datawarden.validators.base import Or


class TestNoTimeGaps:
  """Tests for NoTimeGaps validator."""

  def test_no_time_gaps_datetime_values(self):
    """Test NoTimeGaps with datetime Series values."""

    @validate
    def process(data: Validated[pd.Series, NoTimeGaps("1D")]) -> pd.Series:
      return data

    # Valid datetime values
    timestamps = pd.date_range("2024-01-01", periods=3, freq="1D")
    data = pd.Series(timestamps)
    result = process(data)
    assert result.equals(data)

    # Invalid (missing day)
    timestamps = pd.to_datetime(["2024-01-01", "2024-01-03"])
    data = pd.Series(timestamps)
    with pytest.raises(ValidationError):
      process(data)

  def test_empty_datetime_series(self):
    """Test NoTimeGaps with empty datetime Series."""

    @validate
    def process(data: Validated[pd.Series, NoTimeGaps("1D")]) -> pd.Series:
      return data

    data = pd.Series([], dtype="datetime64[ns]")
    result = process(data)
    assert result.equals(data)

  def test_notimegaps_with_series_index_via_wrapper(self):
    """Test NoTimeGaps on Series index via Index wrapper."""

    @validate
    def process(data: Validated[pd.Series, Index(NoTimeGaps("1D"))]) -> pd.Series:
      return data

    index = pd.date_range("2023-01-01", periods=5, freq="D")
    data = pd.Series([1, 2, 3, 4, 5], index=index)
    result = process(data)
    assert result.equals(data)

  def test_notimegaps_detects_gaps_via_index_wrapper(self):
    """Test NoTimeGaps detects gaps via Index wrapper."""

    @validate
    def process(data: Validated[pd.Series, Index(NoTimeGaps("1D"))]) -> pd.Series:
      return data

    index = pd.DatetimeIndex(["2023-01-01", "2023-01-02", "2023-01-04"])
    data = pd.Series([1, 2, 3], index=index)
    with pytest.raises(ValidationError):
      process(data)


class TestMaxGap:
  """Test MaxGap validator."""

  def test_maxgap_with_valid_datetime_series(self):
    """Test MaxGap passes when gaps are within limit."""

    @validate
    def process(data: Validated[pd.Series, MaxGap("2min")]) -> pd.Series:
      return data

    timestamps = pd.date_range("2023-01-01", periods=5, freq="1min")
    data = pd.Series(timestamps)
    result = process(data)
    assert result.equals(data)

  def test_maxgap_with_gaps_within_tolerance(self):
    """Test MaxGap passes with gaps within tolerance."""

    @validate
    def process(data: Validated[pd.Series, MaxGap("2min")]) -> pd.Series:
      return data

    # 1-minute data with one 2-minute gap (missing row)
    timestamps = pd.DatetimeIndex([
      "2023-01-01 09:00",
      "2023-01-01 09:01",
      "2023-01-01 09:02",
      "2023-01-01 09:04",  # Skipped 09:03
      "2023-01-01 09:05",
    ])
    data = pd.Series(timestamps)
    result = process(data)
    assert result.equals(data)

  def test_maxgap_fails_when_gap_exceeds_limit(self):
    """Test MaxGap fails when gap exceeds allowed limit."""

    @validate
    def process(data: Validated[pd.Series, MaxGap("2min")]) -> pd.Series:
      return data

    timestamps = pd.DatetimeIndex([
      "2023-01-01 09:00",
      "2023-01-01 09:01",
      "2023-01-01 09:05",  # 4-minute gap!
    ])
    data = pd.Series(timestamps)
    with pytest.raises(ValidationError):
      process(data)

  def test_maxgap_with_index_wrapper(self):
    """Test MaxGap on DataFrame index via Index wrapper."""

    @validate
    def process(df: Validated[pd.DataFrame, Index(MaxGap("3D"))]) -> pd.DataFrame:
      return df

    index = pd.DatetimeIndex([
      "2023-01-01",
      "2023-01-02",
      "2023-01-04",  # 2-day gap
    ])
    df = pd.DataFrame({"a": [1, 2, 3]}, index=index)
    result = process(df)
    assert result.equals(df)

  def test_maxgap_empty_series(self):
    """Test MaxGap with empty datetime Series."""

    @validate
    def process(data: Validated[pd.Series, MaxGap("1D")]) -> pd.Series:
      return data

    data = pd.Series([], dtype="datetime64[ns]")
    result = process(data)
    assert result.equals(data)

  def test_maxgap_single_value(self):
    """Test MaxGap with single value passes."""

    @validate
    def process(data: Validated[pd.Series, MaxGap("1D")]) -> pd.Series:
      return data

    data = pd.Series([pd.Timestamp("2023-01-01")])
    result = process(data)
    assert result.equals(data)


class TestMaxDiff:
  """Test MaxDiff validator for numeric gap validation."""

  def test_maxdiff_with_valid_series(self):
    """Test MaxDiff passes when differences are within limit."""

    @validate
    def process(data: Validated[pd.Series, MaxDiff(5)]) -> pd.Series:
      return data

    data = pd.Series([10, 12, 14, 15, 17])
    result = process(data)
    assert result.equals(data)

  def test_maxdiff_fails_when_diff_exceeds_limit(self):
    """Test MaxDiff fails when difference exceeds limit."""

    @validate
    def process(data: Validated[pd.Series, MaxDiff(5)]) -> pd.Series:
      return data

    data = pd.Series([10, 12, 20])  # 8-point jump!
    with pytest.raises(ValidationError):
      process(data)

  def test_maxdiff_with_negative_changes(self):
    """Test MaxDiff handles negative changes (uses abs diff)."""

    @validate
    def process(data: Validated[pd.Series, MaxDiff(3)]) -> pd.Series:
      return data

    data = pd.Series([20, 18, 15, 14])  # All diffs <= 3
    result = process(data)
    assert result.equals(data)

  def test_maxdiff_fails_negative_large_jump(self):
    """Test MaxDiff fails on large negative jump."""

    @validate
    def process(data: Validated[pd.Series, MaxDiff(5)]) -> pd.Series:
      return data

    data = pd.Series([20, 10])  # -10 jump exceeds 5
    with pytest.raises(ValidationError):
      process(data)

  def test_maxdiff_with_float_limit(self):
    """Test MaxDiff with float limit."""

    @validate
    def process(data: Validated[pd.Series, MaxDiff(0.5)]) -> pd.Series:
      return data

    data = pd.Series([1.0, 1.2, 1.5, 1.6])
    result = process(data)
    assert result.equals(data)

  def test_maxdiff_with_column(self):
    """Test MaxDiff with Column wrapper."""

    @validate
    def process(
      df: Validated[pd.DataFrame, Column("price", MaxDiff(5))],
    ) -> pd.DataFrame:
      return df

    df = pd.DataFrame({"price": [100, 102, 105, 104]})
    result = process(df)
    assert result.equals(df)

  def test_maxdiff_empty_series(self):
    """Test MaxDiff with empty Series."""

    @validate
    def process(data: Validated[pd.Series, MaxDiff(5)]) -> pd.Series:
      return data

    data = pd.Series([], dtype="float64")
    result = process(data)
    assert result.equals(data)

  def test_maxdiff_single_value(self):
    """Test MaxDiff with single value passes."""

    @validate
    def process(data: Validated[pd.Series, MaxDiff(5)]) -> pd.Series:
      return data

    data = pd.Series([42])
    result = process(data)
    assert result.equals(data)

  def test_maxdiff_with_isnan_logic(self):
    """Test MaxDiff with | IsNaN pattern."""

    @validate
    def process(data: Validated[pd.Series, Or(MaxDiff(5), IsNaN)]) -> pd.Series:
      return data

    # Should allow NaNs
    data = pd.Series([10.0, 12.0, np.nan, 14.0, 15.0])
    # Note: currently this fails at index 3 because 14.0 - nan is nan
    # If the user says it's implemented, maybe they expect it to work differently
    # or they fixed MaxDiff. Let's see.
    try:
      result = process(data)
      assert result.equals(data)
    except ValidationError:
      pytest.fail("MaxDiff | IsNaN should have passed")
