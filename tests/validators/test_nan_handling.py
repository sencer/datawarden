"""Tests for IgnoringNaNs wrapper and NaN handling in comparisons."""

from typing import Annotated

import numpy as np
import pandas as pd
import pytest

from datawarden import (
  Datetime,
  Finite,
  Ge,
  IgnoringNaNs,
  Index,
  Is,
  Le,
  Lt,
  MonoUp,
  Negative,
  Not,
  Positive,
  Shape,
  validate,
)


class TestIgnoringNaNsWrapper:
  """Test the IgnoringNaNs wrapper validator."""

  def test_ignoring_nans_holistic_is(self):
    """CRITICAL: IgnoringNaNs should pass the whole DataFrame to holistic Is."""

    @validate
    def func(
      df: Annotated[pd.DataFrame, IgnoringNaNs(Is(lambda df: df["a"] < df["b"]))],
    ):
      pass

    # This should pass: non-NaN rows satisfy a < b
    df = pd.DataFrame({"a": [1, 2, np.nan], "b": [2, 3, 4]})
    func(df)

  def test_ignoring_nans_holistic_shape(self):
    """CRITICAL: IgnoringNaNs should work with Shape (which is holistic)."""

    @validate
    def func(df: Annotated[pd.DataFrame, IgnoringNaNs(Shape(rows=2))]):
      pass

    # 3 rows total, but only 2 after dropna()
    df = pd.DataFrame({"a": [1, 2, np.nan], "b": [4, 5, 6]})
    func(df)

  def test_reset_propagation_ignoring_nans(self):
    """MEDIUM: reset() should propagate through IgnoringNaNs."""
    mono = MonoUp()
    # bypass instantiate_validator by manual assignment
    v = IgnoringNaNs.__new__(IgnoringNaNs)
    v.wrapped = mono

    s = pd.Series([1, 2, 3])
    v.validate(s)
    assert mono._last_val == 3

    v.reset()
    assert mono._last_val is None, "MonoUp was not reset via IgnoringNaNs"

  def test_marker_mode_basic(self):
    """Test that IgnoringNaNs() marker wraps all validators."""

    @validate
    def process(data: Annotated[pd.Series, Ge(0), Le(100), IgnoringNaNs]) -> pd.Series:
      return data

    # Should pass - NaN values are ignored
    valid_data = pd.Series([10.0, np.nan, 50.0, np.nan, 90.0])
    result = process(valid_data)
    assert result.equals(valid_data)

  def test_marker_mode_rejects_invalid(self):
    """Test that marker mode still validates non-NaN values."""

    @validate
    def process(data: Annotated[pd.Series, Ge(0), Le(100), IgnoringNaNs]) -> pd.Series:
      return data

    invalid_data = pd.Series([10.0, np.nan, 150.0])  # 150 > 100
    with pytest.raises(ValueError, match="must be <= 100"):
      process(invalid_data)

  def test_marker_mode_equivalent_to_explicit(self):
    """Test that marker mode is equivalent to explicit wrapping."""
    # These should behave identically

    @validate
    def process_marker(
      data: Annotated[pd.Series, Ge(0), Lt(10), IgnoringNaNs],
    ) -> pd.Series:
      return data

    @validate
    def process_explicit(
      data: Annotated[pd.Series, IgnoringNaNs(Ge(0)), IgnoringNaNs(Lt(10))],
    ) -> pd.Series:
      return data

    data = pd.Series([1.0, np.nan, 5.0, np.nan, 9.0])
    r1 = process_marker(data)
    r2 = process_explicit(data)
    assert r1.equals(r2)

  def test_series_with_nans_and_valid_values(self):
    """Test that IgnoringNaNs allows NaN but validates non-NaN values."""

    @validate
    def process(data: Annotated[pd.Series, IgnoringNaNs(Ge(0))]) -> pd.Series:
      return data

    valid_data = pd.Series([1.0, np.nan, 2.0, np.nan, 3.0])
    result = process(valid_data)
    assert result.equals(valid_data)

  def test_series_with_nans_and_invalid_values(self):
    """Test that IgnoringNaNs still validates non-NaN values."""

    @validate
    def process(data: Annotated[pd.Series, IgnoringNaNs(Ge(0))]) -> pd.Series:
      return data

    invalid_data = pd.Series([1.0, np.nan, -2.0, np.nan, 3.0])
    with pytest.raises(ValueError, match="must be >= 0"):
      process(invalid_data)

  def test_series_all_nans(self):
    """Test that all-NaN series passes when wrapped."""

    @validate
    def process(data: Annotated[pd.Series, IgnoringNaNs(Ge(0))]) -> pd.Series:
      return data

    all_nan = pd.Series([np.nan, np.nan, np.nan])
    result = process(all_nan)
    assert result.equals(all_nan)

  def test_dataframe_with_nans(self):
    """Test IgnoringNaNs with DataFrames."""

    @validate
    def process(data: Annotated[pd.DataFrame, IgnoringNaNs(Ge(0))]) -> pd.DataFrame:
      return data

    df = pd.DataFrame({"a": [1.0, np.nan, 3.0], "b": [np.nan, 2.0, np.nan]})
    assert process(df).equals(df)

  def test_dataframe_fails_with_invalid_values(self):
    """Test IgnoringNaNs with DataFrame fails on invalid non-NaN values."""
    validator = IgnoringNaNs(Ge(0))
    df = pd.DataFrame({"a": [1.0, np.nan, -3.0], "b": [np.nan, 2.0, np.nan]})
    with pytest.raises(ValueError, match="must be >= 0"):
      validator.validate(df)

  def test_multiple_validators_wrapped(self):
    """Test IgnoringNaNs with multiple validators."""

    @validate
    def process(
      data: Annotated[pd.Series, IgnoringNaNs(Ge(0)), IgnoringNaNs(Le(100))],
    ) -> pd.Series:
      return data

    valid_data = pd.Series([10.0, np.nan, 50.0, np.nan, 90.0])
    result = process(valid_data)
    assert result.equals(valid_data)

    invalid_data = pd.Series([10.0, np.nan, 150.0])
    with pytest.raises(ValueError, match="must be <= 100"):
      process(invalid_data)

  def test_with_positive_validator(self):
    """Test IgnoringNaNs works with Positive validator."""

    validator = IgnoringNaNs(Positive)
    data = pd.Series([1.0, np.nan, 2.0, np.nan, 3.0])
    assert validator.validate(data) is None

    # Fails on zero
    invalid = pd.Series([1.0, np.nan, 0.0])
    with pytest.raises(ValueError, match="must be positive"):
      validator.validate(invalid)

  def test_with_nonnegative_validator(self):
    """Test IgnoringNaNs works with Not(Negative) validator."""

    validator = IgnoringNaNs(Not(Negative()))
    data = pd.Series([0.0, np.nan, 1.0, np.nan, 2.0])
    assert validator.validate(data) is None

    # Fails on negative
    invalid = pd.Series([0.0, np.nan, -1.0])
    with pytest.raises(ValueError, match="must be non-negative"):
      validator.validate(invalid)

  def test_with_finite_validator(self):
    """Test IgnoringNaNs works with Finite validator."""

    validator = IgnoringNaNs(Finite)
    data = pd.Series([1.0, np.nan, 2.0, np.nan, 3.0])
    assert validator.validate(data) is None

    # Fails on Inf
    invalid = pd.Series([1.0, np.nan, np.inf])
    with pytest.raises(ValueError, match="must be finite"):
      validator.validate(invalid)

  def test_with_index(self):
    """Test IgnoringNaNs works with pd.Index."""
    validator = IgnoringNaNs(Ge(0))
    # pd.Index with NaN values
    idx = pd.Index([1.0, np.nan, 2.0, np.nan, 3.0])
    # pd.Index with NaN values
    idx = pd.Index([1.0, np.nan, 2.0, np.nan, 3.0])
    assert validator.validate(idx) is None

  def test_with_index_invalid(self):
    """Test IgnoringNaNs with pd.Index fails on invalid values."""
    validator = IgnoringNaNs(Ge(0))
    idx = pd.Index([1.0, np.nan, -2.0])
    with pytest.raises(ValueError, match="must be >= 0"):
      validator.validate(idx)

  def test_with_index_all_nan(self):
    """Test IgnoringNaNs with all-NaN Index passes."""
    validator = IgnoringNaNs(Ge(0))
    idx = pd.Index([np.nan, np.nan, np.nan])
    validator = IgnoringNaNs(Ge(0))
    idx = pd.Index([np.nan, np.nan, np.nan])
    assert validator.validate(idx) is None

  def test_with_index_and_datetime(self):
    """Regression test: IgnoringNaNs(Index(Datetime)) on Index with NaTs."""
    validator = IgnoringNaNs(Index(Datetime))
    # Index with NaT
    idx = pd.to_datetime(["2024-01-01", "NaT", "2024-01-03"])
    assert validator.validate(idx) is None


class TestComparisonNaNHandling:
  """Test that comparison validators reject NaN by default."""

  def test_ge_rejects_nan(self):
    """Test that Ge raises on NaN values."""

    @validate
    def process(data: Annotated[pd.Series, Ge(0)]) -> pd.Series:
      return data

    with pytest.raises(ValueError, match="Cannot perform >= comparison with NaN"):
      process(pd.Series([1.0, 2.0, np.nan]))

  def test_le_rejects_nan(self):
    """Test that Le raises on NaN values."""

    @validate
    def process(data: Annotated[pd.Series, Le(100)]) -> pd.Series:
      return data

    with pytest.raises(ValueError, match="Cannot perform <= comparison with NaN"):
      process(pd.Series([50.0, np.nan, 75.0]))

  def test_comparison_error_message_helpful(self):
    """Test that error message mentions IgnoringNaNs."""

    @validate
    def process(data: Annotated[pd.Series, Ge(0)]) -> pd.Series:
      return data

    with pytest.raises(ValueError, match="IgnoringNaNs wrapper"):
      process(pd.Series([1.0, np.nan]))

  def test_comparison_without_nan_passes(self):
    """Test that comparisons work fine without NaN."""

    @validate
    def process(data: Annotated[pd.Series, Ge(0), Le(100)]) -> pd.Series:
      return data

    valid_data = pd.Series([10.0, 50.0, 90.0])
    result = process(valid_data)
    assert result.equals(valid_data)

  def test_ge_unrelated_nan_success(self):
    """Test that Ge does not fail if an unrelated column has a NaN."""
    # Column 'c' has a NaN, but we only validate 'a' >= 'b'
    df = pd.DataFrame({"a": [2, 3], "b": [1, 2], "c": [np.nan, 4]})
    v = Ge("a", "b")
    # Should pass
    v.validate(df)

  def test_ignoring_nans_masking_prevention(self):
    """Test that IgnoringNaNs does not mask errors due to unrelated NaNs."""

    @validate
    def func(df: Annotated[pd.DataFrame, IgnoringNaNs(Finite)]):
      pass

    # We want to catch the Inf in 'a', even if 'b' has a NaN
    df = pd.DataFrame({"a": [np.inf, 10.0], "b": [np.nan, 2.0]})

    # The transformation should turn IgnoringNaNs(Finite()) into Finite()
    # because Finite() already ignores NaNs. Finite() is holistic (promoted) and
    # selects numeric columns surgically.
    with pytest.raises(ValueError, match="Data must be finite"):
      func(df)
