"""Additional NaN handling tests - unique cases not covered in test_isnan_comprehensive.py

Ported from datawarden v1 test_nan_handling.py, focusing on:
- Holistic validators with | IsNaN
- Index validators with NaN
- Comparison validators NaN behavior
"""

import numpy as np
import pandas as pd
import pytest

from datawarden import MonoUp, Rows, Unique, Validated, validate
from datawarden.exceptions import ValidationError
from datawarden.validators import (
  Datetime,
  Finite,
  Ge,
  Index,
  IsNaN,
  Le,
  Positive,
)
from datawarden.validators.base import Or


class TestOrIsNaNHolistic:
  """Test | IsNaN with holistic validators."""

  def test_unique_or_isnan(self):
    """Test Unique | IsNaN allows multiple NaNs while requiring unique values otherwise."""

    @validate
    def process(data: Validated[pd.Series, Or(Unique, IsNaN)]) -> pd.Series:
      return data

    # Should pass: unique 1, 2, and multiple NaNs
    data = pd.Series([1.0, np.nan, 2.0, np.nan])
    result = process(data)
    assert result.equals(data)

    # Should fail: duplicated 1
    with pytest.raises(ValidationError):
      process(pd.Series([1.0, np.nan, 1.0]))

  def test_monoup_or_isnan(self):
    """Test MonoUp | IsNaN allows NaN while checking monotonicity."""

    @validate
    def process(data: Validated[pd.Series, Or(MonoUp, IsNaN)]) -> pd.Series:
      return data

    # Should pass: [1, 2, nan, 3] -> 1<2, 2<3 (if MonoUp ignores NaN correctly)
    # Wait, my previous analysis said it might still fail at 3 if not NaN-aware.
    # Let's see if the current implementation of is_monotonic_increasing handles it.
    data = pd.Series([1.0, 2.0, np.nan, 3.0])
    try:
      result = process(data)
      assert result.equals(data)
    except ValidationError:
      # If it fails, then MonoUp is not NaN-aware and we should keep it skipped or fix it.
      # But let's check first.
      pytest.fail("MonoUp | IsNaN failed - MonoUp might not be NaN-aware")


class TestOrIsNaNWithIndex:
  """Test | IsNaN with Index validators."""

  def test_with_index(self):
    """Test | IsNaN allows NaN in index."""

    @validate
    def process(data: Validated[pd.Series, Or(Index(Datetime), IsNaN)]) -> pd.Series:
      return data

    # Valid datetime index with NaN values
    dates = pd.date_range("2024-01-01", periods=3, freq="D")
    data = pd.Series([1.0, np.nan, 3.0], index=dates)
    result = process(data)
    assert result.equals(data)

  def test_with_index_all_nan(self):
    """Test all-NaN series with Index validator."""

    @validate
    def process(data: Validated[pd.Series, Or(Index(Datetime), IsNaN)]) -> pd.Series:
      return data

    dates = pd.date_range("2024-01-01", periods=3, freq="D")
    all_nan = pd.Series([np.nan, np.nan, np.nan], index=dates)
    result = process(all_nan)
    assert result.equals(all_nan)


class TestComparisonNaNBehavior:
  """Test that comparison validators reject NaN by default."""

  def test_ge_rejects_nan(self):
    """Test Ge rejects NaN values by default."""

    @validate
    def process(data: Validated[pd.Series, Ge(0)]) -> pd.Series:
      return data

    # NaN should fail
    with pytest.raises(ValidationError):
      process(pd.Series([1.0, np.nan, 3.0]))

  def test_le_rejects_nan(self):
    """Test Le rejects NaN values by default."""

    @validate
    def process(data: Validated[pd.Series, Le(100)]) -> pd.Series:
      return data

    # NaN should fail
    with pytest.raises(ValidationError):
      process(pd.Series([50.0, np.nan, 75.0]))

  def test_comparison_without_nan_passes(self):
    """Test comparison validators pass without NaN."""

    @validate
    def process(data: Validated[pd.Series, Ge(0), Le(100)]) -> pd.Series:
      return data

    valid_data = pd.Series([10.0, 50.0, 90.0])
    result = process(valid_data)
    assert result.equals(valid_data)


class TestOrIsNaNWithRows:
  """Test | IsNaN with Rows validator."""

  def test_logic_wrapping_rows(self):
    """Test Rows | IsNaN allows NaNs row-wise."""

    @validate
    def process(
      data: Validated[pd.DataFrame, Or(Rows(lambda r: r["a"] > 0), IsNaN)],
    ) -> pd.DataFrame:
      return data

    # Row 0: a=1 (>0) -> Pass
    # Row 1: a=nan (IsNaN) -> Pass
    # Row 2: a=-1 (Fail both) -> Fail
    df = pd.DataFrame({"a": [1.0, np.nan, -1.0]})

    with pytest.raises(ValidationError):
      process(df)

    # Valid case
    df_valid = pd.DataFrame({"a": [1.0, np.nan, 2.0]})
    result = process(df_valid)
    assert result.equals(df_valid)


class TestOrIsNaNMultipleValidators:
  """Test | IsNaN with multiple validators."""

  def test_multiple_validators_wrapped(self):
    """Test | IsNaN with multiple validators in a list."""

    @validate
    def process(data: Validated[pd.Series, Or(Ge(0), Le(100), IsNaN)]) -> pd.Series:
      return data

    # Should pass - values satisfy Ge(0) OR Le(100) OR IsNaN
    valid_data = pd.Series([10.0, np.nan, 50.0, np.nan, 90.0])
    result = process(valid_data)
    assert result.equals(valid_data)

  def test_with_positive_validator(self):
    """Test | IsNaN with Positive validator."""

    @validate
    def process(data: Validated[pd.Series, Or(Positive, IsNaN)]) -> pd.Series:
      return data

    valid_data = pd.Series([1.0, np.nan, 2.0, np.nan, 3.0])
    result = process(valid_data)
    assert result.equals(valid_data)

    # Negative should fail
    with pytest.raises(ValidationError):
      process(pd.Series([1.0, np.nan, -2.0]))

  def test_with_finite_validator(self):
    """Test | IsNaN with Finite validator."""

    @validate
    def process(data: Validated[pd.Series, Or(Finite, IsNaN)]) -> pd.Series:
      return data

    # NaN is allowed, Inf is not
    valid_data = pd.Series([1.0, np.nan, 2.0])
    result = process(valid_data)
    assert result.equals(valid_data)

    # Inf should fail
    with pytest.raises(ValidationError):
      process(pd.Series([1.0, np.inf, np.nan]))


class TestOrIsNaNEdgeCases:
  """Test edge cases for | IsNaN pattern."""

  def test_series_all_nans(self):
    """Test that all-NaN series passes with | IsNaN."""

    @validate
    def process(data: Validated[pd.Series, Or(Ge(0), IsNaN)]) -> pd.Series:
      return data

    all_nan = pd.Series([np.nan, np.nan, np.nan])
    result = process(all_nan)
    assert result.equals(all_nan)

  def test_dataframe_with_nans(self):
    """Test | IsNaN with DataFrames."""

    @validate
    def process(data: Validated[pd.DataFrame, Or(Ge(0), IsNaN)]) -> pd.DataFrame:
      return data

    df = pd.DataFrame({"a": [1.0, np.nan, 3.0], "b": [4.0, 5.0, np.nan]})
    result = process(df)
    assert result.equals(df)

  def test_dataframe_fails_with_invalid_values(self):
    """Test | IsNaN still validates non-NaN values in DataFrame."""

    @validate
    def process(data: Validated[pd.DataFrame, Or(Ge(0), IsNaN)]) -> pd.DataFrame:
      return data

    # Negative value should fail
    with pytest.raises(ValidationError):
      process(pd.DataFrame({"a": [1.0, np.nan, -3.0]}))
