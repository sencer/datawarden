"""Tests for gap validators: NoTimeGaps, MaxGap, MaxDiff."""
# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false
# pyright: reportCallIssue=false, reportAttributeAccessIssue=false

import numpy as np
import pandas as pd
import pytest

from datawarden import (
  HasColumn,
  IgnoringNaNs,
  Index,
  MaxDiff,
  MaxGap,
  NoTimeGaps,
)


class TestNoTimeGaps:
  """Tests for NoTimeGaps validator."""

  def test_no_time_gaps_datetime_values(self):
    """Test NoTimeGaps with datetime Series values."""
    validator = NoTimeGaps(freq="1D")

    # Valid datetime values
    timestamps = pd.date_range("2024-01-01", periods=3, freq="1D")
    data = pd.Series(timestamps)
    assert validator.validate(data) is None

    # Invalid (missing day)
    timestamps = pd.to_datetime(["2024-01-01", "2024-01-03"])
    data = pd.Series(timestamps)
    with pytest.raises(ValueError, match="Time gaps detected"):
      validator.validate(data)

  def test_no_time_gaps_datetimeindex(self):
    """Test NoTimeGaps with DatetimeIndex directly."""
    validator = NoTimeGaps(freq="1D")

    # Valid DatetimeIndex
    index = pd.date_range("2024-01-01", periods=3, freq="1D")
    assert validator.validate(index) is None

    # Invalid (missing day)
    index = pd.DatetimeIndex(["2024-01-01", "2024-01-03"])
    with pytest.raises(ValueError, match="Time gaps detected"):
      validator.validate(index)

  def test_empty_datetime_series(self):
    validator = NoTimeGaps(freq="1D")
    data = pd.Series([], dtype="datetime64[ns]")
    assert validator.validate(data) is None

  def test_notimegaps_with_series_index_via_wrapper(self):
    """Test NoTimeGaps on Series index via Index wrapper."""
    index = pd.date_range("2023-01-01", periods=5, freq="D")
    data = pd.Series([1, 2, 3, 4, 5], index=index)
    validator = Index(NoTimeGaps("D"))
    validator = Index(NoTimeGaps("D"))
    assert validator.validate(data) is None

  def test_notimegaps_detects_gaps_via_index_wrapper(self):
    """Test NoTimeGaps detects gaps via Index wrapper."""
    index = pd.DatetimeIndex(["2023-01-01", "2023-01-02", "2023-01-04"])
    data = pd.Series([1, 2, 3], index=index)
    validator = Index(NoTimeGaps("D"))
    with pytest.raises(ValueError, match="Time gaps detected"):
      validator.validate(data)

  def test_notimegaps_with_non_datetime_series(self):
    """Test NoTimeGaps requires datetime data."""
    data = pd.Series([1, 2, 3])  # numeric, not datetime
    validator = NoTimeGaps("D")
    with pytest.raises(ValueError, match="NoTimeGaps requires datetime data"):
      validator.validate(data)

  def test_notimegaps_with_dataframe_index_via_wrapper(self):
    """Test NoTimeGaps with DataFrame index via Index wrapper."""
    index = pd.date_range("2023-01-01", periods=5, freq="D")
    df = pd.DataFrame({"a": [1, 2, 3, 4, 5]}, index=index)
    validator = Index(NoTimeGaps("D"))
    validator = Index(NoTimeGaps("D"))
    assert validator.validate(df) is None

  def test_notimegaps_wrong_number_of_entries(self):
    """Test NoTimeGaps when datetime values have wrong number of entries."""
    timestamps = pd.DatetimeIndex(["2023-01-01", "2023-01-03", "2023-01-05"])
    data = pd.Series(timestamps)
    validator = NoTimeGaps("D")
    with pytest.raises(ValueError, match="Time gaps detected"):
      validator.validate(data)

  def test_notimegaps_correct_length_wrong_dates(self):
    """Test NoTimeGaps with duplicates (correct length but wrong timestamps)."""
    timestamps = pd.DatetimeIndex([
      "2023-01-01",
      "2023-01-01",  # Duplicate!
      "2023-01-03",
    ])
    data = pd.Series(timestamps)
    validator = NoTimeGaps("D")
    with pytest.raises(ValueError, match="Time gaps detected"):
      validator.validate(data)


class TestMaxGap:
  """Test MaxGap validator."""

  def test_maxgap_with_valid_datetime_series(self):
    """Test MaxGap passes when gaps are within limit."""
    timestamps = pd.date_range("2023-01-01", periods=5, freq="1min")
    data = pd.Series(timestamps)
    validator = MaxGap("2min")  # Allow up to 2 minute gaps
    validator = MaxGap("2min")  # Allow up to 2 minute gaps
    assert validator.validate(data) is None

  def test_maxgap_with_gaps_within_tolerance(self):
    """Test MaxGap passes with gaps within tolerance."""
    # 1-minute data with one 2-minute gap (missing row)
    timestamps = pd.DatetimeIndex([
      "2023-01-01 09:00",
      "2023-01-01 09:01",
      "2023-01-01 09:02",
      "2023-01-01 09:04",  # Skipped 09:03
      "2023-01-01 09:05",
    ])
    data = pd.Series(timestamps)
    validator = MaxGap("2min")  # 2-minute gaps allowed
    validator = MaxGap("2min")  # 2-minute gaps allowed
    assert validator.validate(data) is None

  def test_maxgap_fails_when_gap_exceeds_limit(self):
    """Test MaxGap fails when gap exceeds allowed limit."""
    timestamps = pd.DatetimeIndex([
      "2023-01-01 09:00",
      "2023-01-01 09:01",
      "2023-01-01 09:05",  # 4-minute gap!
    ])
    data = pd.Series(timestamps)
    validator = MaxGap("2min")
    with pytest.raises(ValueError, match="Time gap exceeds maximum"):
      validator.validate(data)

  def test_maxgap_with_datetimeindex(self):
    """Test MaxGap with DatetimeIndex directly."""
    index = pd.date_range("2023-01-01", periods=5, freq="D")
    validator = MaxGap("2D")
    validator = MaxGap("2D")
    assert validator.validate(index) is None

  def test_maxgap_with_index_wrapper(self):
    """Test MaxGap on DataFrame index via Index wrapper."""
    index = pd.DatetimeIndex([
      "2023-01-01",
      "2023-01-02",
      "2023-01-04",  # 2-day gap
    ])
    df = pd.DataFrame({"a": [1, 2, 3]}, index=index)
    validator = Index(MaxGap("3D"))  # Allow up to 3-day gaps
    validator = Index(MaxGap("3D"))  # Allow up to 3-day gaps
    assert validator.validate(df) is None

  def test_maxgap_with_non_datetime(self):
    """Test MaxGap requires datetime data."""
    data = pd.Series([1, 2, 3])
    validator = MaxGap("1D")
    with pytest.raises(ValueError, match="MaxGap requires datetime data"):
      validator.validate(data)

  def test_maxgap_empty_series(self):
    """Test MaxGap with empty datetime Series."""
    data = pd.Series([], dtype="datetime64[ns]")
    validator = MaxGap("1D")
    validator = MaxGap("1D")
    assert validator.validate(data) is None

  def test_maxgap_single_value(self):
    """Test MaxGap with single value passes."""
    data = pd.Series([pd.Timestamp("2023-01-01")])
    validator = MaxGap("1D")
    validator = MaxGap("1D")
    assert validator.validate(data) is None


class TestMaxDiff:
  """Test MaxDiff validator for numeric gap validation."""

  def test_maxdiff_with_valid_series(self):
    """Test MaxDiff passes when differences are within limit."""
    data = pd.Series([10, 12, 14, 15, 17])
    validator = MaxDiff(5)
    validator = MaxDiff(5)
    assert validator.validate(data) is None

  def test_maxdiff_fails_when_diff_exceeds_limit(self):
    """Test MaxDiff fails when difference exceeds limit."""
    data = pd.Series([10, 12, 20])  # 8-point jump!
    validator = MaxDiff(5)
    with pytest.raises(ValueError, match="Difference exceeds maximum"):
      validator.validate(data)

  def test_maxdiff_with_negative_changes(self):
    """Test MaxDiff handles negative changes (uses abs diff)."""
    data = pd.Series([20, 18, 15, 14])  # All diffs <= 3
    validator = MaxDiff(3)
    validator = MaxDiff(3)
    assert validator.validate(data) is None

  def test_maxdiff_fails_negative_large_jump(self):
    """Test MaxDiff fails on large negative jump."""
    data = pd.Series([20, 10])  # -10 jump exceeds 5
    validator = MaxDiff(5)
    with pytest.raises(ValueError, match="Difference exceeds maximum"):
      validator.validate(data)

  def test_maxdiff_with_float_limit(self):
    """Test MaxDiff with float limit."""
    data = pd.Series([1.0, 1.2, 1.5, 1.6])
    validator = MaxDiff(0.5)
    validator = MaxDiff(0.5)
    assert validator.validate(data) is None

  def test_maxdiff_with_hascolumn(self):
    """Test MaxDiff with HasColumn wrapper."""
    df = pd.DataFrame({"price": [100, 102, 105, 104]})
    validator = HasColumn("price", MaxDiff(5))
    validator = HasColumn("price", MaxDiff(5))
    assert validator.validate(df) is None

  def test_maxdiff_requires_series(self):
    """Test MaxDiff requires a pandas Series."""
    df = pd.DataFrame({"a": [1, 2, 3]})
    validator = MaxDiff(5)
    with pytest.raises(TypeError, match="MaxDiff requires a pandas Series"):
      validator.validate(df)

  def test_maxdiff_requires_numeric(self):
    """Test MaxDiff requires numeric data."""
    data = pd.Series(["a", "b", "c"])
    validator = MaxDiff(5)
    with pytest.raises(ValueError, match="MaxDiff requires numeric data"):
      validator.validate(data)

  def test_maxdiff_empty_series(self):
    """Test MaxDiff with empty Series."""
    data = pd.Series([], dtype="float64")
    validator = MaxDiff(5)
    validator = MaxDiff(5)
    assert validator.validate(data) is None

  def test_maxdiff_single_value(self):
    """Test MaxDiff with single value passes."""
    data = pd.Series([42])
    validator = MaxDiff(5)
    assert validator.validate(data) is None

  def test_maxdiff_fails_on_nan_by_default(self):
    """Test MaxDiff fails on data with NaNs by default."""
    data = pd.Series([1.0, 2.0, np.nan, 3.0])
    validator = MaxDiff(5)
    with pytest.raises(ValueError, match="MaxDiff cannot validate data with NaN"):
      validator.validate(data)

  def test_maxdiff_with_ignoring_nans(self):
    """Test MaxDiff with IgnoringNaNs wrapper."""
    # [1.0, NaN, 4.0] -> jumps [1.0, 4.0] -> diff 3 <= 5 -> Pass
    data = pd.Series([1.0, float("nan"), 4.0])
    validator = IgnoringNaNs(MaxDiff(5))
    assert validator.validate(data) is None

    # [1.0, NaN, 10.0] -> jumps [1.0, 10.0] -> diff 9 > 5 -> Fail
    data_large = pd.Series([1.0, float("nan"), 10.0])
    with pytest.raises(ValueError, match="Difference exceeds maximum 5"):
      validator.validate(data_large)
