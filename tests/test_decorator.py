"""Tests for the @validate decorator - Core functionality."""

import numpy as np
import pandas as pd
import pytest

from datawarden import Validated, validate
from datawarden.exceptions import ValidationError
from datawarden.validators import (
  Column,
  Columns,
  Finite,
  Ge,
  IsNaN,
  NotEmpty,
  NotNaN,
  Positive,
)
from datawarden.validators.base import Not, Or


class TestValidatedDecorator:
  """Tests for @validate decorator basic functionality."""

  def test_function_with_validation(self) -> None:
    """Test @validate decorator validates arguments."""

    @validate
    def process(data: Validated[pd.Series, Finite]) -> float:
      return data.sum()

    valid_data = pd.Series([1.0, 2.0, 3.0])
    result = process(valid_data)
    assert result == 6.0

  def test_function_rejects_invalid_data(self) -> None:
    """Test @validate decorator rejects invalid data."""

    @validate
    def process(data: Validated[pd.Series, Finite]) -> float:
      return data.sum()

    invalid_data = pd.Series([1.0, np.inf, 3.0])
    with pytest.raises(ValidationError):
      process(invalid_data)

  def test_multiple_validators(self) -> None:
    """Test multiple validators in chain."""

    @validate
    def process(data: Validated[pd.Series, Finite, Positive]) -> float:
      return data.sum()

    # Valid data
    valid_data = pd.Series([1.0, 2.0, 3.0])
    result = process(valid_data)
    assert result == 6.0

    # Fails Finite check (Inf)
    with pytest.raises(ValidationError):
      process(pd.Series([1.0, np.inf, 3.0]))

    # Fails Positive check
    with pytest.raises(ValidationError):
      process(pd.Series([1.0, 0.0, 3.0]))

  def test_dataframe_validation(self) -> None:
    """Test DataFrame validation."""

    @validate
    def process(
      data: Validated[pd.DataFrame, Columns(["a", "b"]), Finite],
    ) -> pd.Series:
      return data["a"] + data["b"]

    valid_data = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    result = process(valid_data)
    assert result.tolist() == [4, 6]

    # Missing column
    with pytest.raises(ValidationError, match="not found"):
      process(pd.DataFrame({"a": [1, 2]}))

  def test_preserves_function_metadata(self) -> None:
    """Test decorator preserves function name and docstring."""

    @validate
    def my_function(data: Validated[pd.Series, Finite]) -> float:
      """My docstring."""
      return data.sum()

    assert my_function.__name__ == "my_function"
    assert my_function.__doc__ == "My docstring."

  def test_works_with_methods(self) -> None:
    """Test @validate works with class methods."""

    class Processor:
      @validate
      def process(self, data: Validated[pd.Series, Finite]) -> float:
        return data.sum()

    processor = Processor()
    result = processor.process(pd.Series([1.0, 2.0, 3.0]))
    assert result == 6.0

  def test_optional_validated_argument(self) -> None:
    """Test optional validated argument."""

    @validate
    def process(data: Validated[pd.Series, Finite] | None = None) -> float:
      if data is None:
        return 0.0
      return data.sum()

    # Pass None
    assert process(None) == 0.0
    # Pass valid
    assert process(pd.Series([1.0, 2.0])) == 3.0
    # Pass invalid
    with pytest.raises(ValidationError):
      process(pd.Series([np.nan, 1.0]))

  def test_multiple_arguments(self) -> None:
    """Test validation with multiple arguments."""

    @validate
    def combine(
      data1: Validated[pd.Series, Finite],
      data2: Validated[pd.Series, Finite],
    ) -> pd.Series:
      return data1 + data2

    valid1 = pd.Series([1.0, 2.0])
    valid2 = pd.Series([3.0, 4.0])
    result = combine(valid1, valid2)
    assert result.tolist() == [4.0, 6.0]

    # First argument invalid
    with pytest.raises(ValidationError):
      combine(pd.Series([np.inf, 2.0]), valid2)

    # Second argument invalid
    with pytest.raises(ValidationError):
      combine(valid1, pd.Series([3.0, np.inf]))

  def test_non_validated_arguments_ignored(self) -> None:
    """Test non-validated arguments are not validated."""

    @validate
    def process(
      data: Validated[pd.Series, Finite],
      multiplier: float,
    ) -> pd.Series:
      return data * multiplier

    valid_data = pd.Series([1.0, 2.0, 3.0])
    result = process(valid_data, multiplier=2.0)
    assert result.tolist() == [2.0, 4.0, 6.0]

  def test_default_argument_values(self) -> None:
    """Test validation with default argument values."""

    @validate
    def process(data: Validated[pd.Series, Finite] | None = None) -> float:
      if data is None:
        data = pd.Series([1.0, 2.0])
      return data.sum()

    # Use default
    assert process() == 3.0
    # Override with valid
    assert process(pd.Series([3.0, 4.0])) == 7.0
    # Override with invalid
    with pytest.raises(ValidationError):
      process(pd.Series([np.inf, 1.0]))


class TestComplexValidations:
  """Tests for complex validation scenarios."""

  def test_ohlc_validation(self) -> None:
    """Test validation of OHLC data."""

    @validate
    def calculate_true_range(
      data: Validated[
        pd.DataFrame,
        Column("high"),
        Column("low"),
        Column("close"),
        Ge("high", "low"),
      ],
    ) -> pd.Series:
      hl = data["high"] - data["low"]
      hc = abs(data["high"] - data["close"].shift(1))
      lc = abs(data["low"] - data["close"].shift(1))
      return pd.concat([hl, hc, lc], axis=1).max(axis=1)

    # Valid OHLC (high >= low)
    valid_data = pd.DataFrame({
      "high": [102, 105, 104],
      "low": [100, 103, 101],
      "close": [101, 104, 102],
    })
    result = calculate_true_range(valid_data)
    assert len(result) == 3

    # High < Low should fail (100 < 102)
    invalid_data = pd.DataFrame({
      "high": [100, 105, 104],
      "low": [102, 103, 101],
      "close": [101, 104, 102],
    })
    with pytest.raises(ValidationError):
      calculate_true_range(invalid_data)

  def test_percentage_returns_validation(self) -> None:
    """Test validation for percentage returns calculation."""

    @validate
    def calculate_returns(prices: Validated[pd.Series, Finite, Positive]) -> pd.Series:
      return prices.pct_change(fill_method=None)

    # Valid prices
    valid_prices = pd.Series([100.0, 102.0, 101.0, 103.0])
    result = calculate_returns(valid_prices)
    assert len(result) == 4

    # Zero price fails Positive check
    with pytest.raises(ValidationError):
      calculate_returns(pd.Series([100.0, 0.0, 101.0]))

    # Inf price fails Finite check
    with pytest.raises(ValidationError):
      calculate_returns(pd.Series([100.0, np.inf, 101.0]))

  def test_isnan_logic_wraps_column(self) -> None:
    """Test Column with Positive | IsNaN pattern."""

    @validate
    def process(df: Validated[pd.DataFrame, Column("a", Or(Positive, IsNaN))]) -> float:
      return df["a"].sum()

    # DataFrame with NaNs in 'a' -> | IsNaN should allow NaNs
    df = pd.DataFrame({"a": [1.0, np.nan, 3.0]})
    assert process(df) == 4.0

    # Negative value should still fail
    with pytest.raises(ValidationError):
      process(pd.DataFrame({"a": [1.0, np.nan, -1.0]}))


class TestEdgeCases:
  """Tests for edge cases and error conditions."""

  def test_empty_series_allowed_by_default(self) -> None:
    """Test validation allows empty Series by default."""

    @validate
    def process(data: Validated[pd.Series, Finite]) -> int:
      return len(data)

    empty_data = pd.Series([], dtype=float)
    # Should not raise
    assert process(empty_data) == 0

  def test_kwargs_arguments(self) -> None:
    """Test validation works with keyword arguments."""

    @validate
    def process(data: Validated[pd.Series, Finite]) -> float:
      return data.sum()

    valid_data = pd.Series([1.0, 2.0, 3.0])

    # Positional
    result = process(valid_data)
    assert result == 6.0

    # Keyword
    result = process(data=valid_data)
    assert result == 6.0


class TestOptInStrictness:
  """Tests for opt-in strictness (Not(IsNaN), NotEmpty)."""

  def test_explicit_non_nan(self) -> None:
    """Test explicit Not(IsNaN) validator."""

    @validate
    def process(data: Validated[pd.Series, Not(IsNaN)]) -> float:
      return data.sum()

    # Valid data
    assert process(pd.Series([1, 2, 3])) == 6.0

    # NaN data fails
    with pytest.raises(ValidationError):
      process(pd.Series([1, np.nan, 3]))

  def test_explicit_non_empty(self) -> None:
    """Test explicit NotEmpty validator."""

    @validate
    def process(data: Validated[pd.Series, NotEmpty]) -> int:
      return len(data)

    # Valid data
    assert process(pd.Series([1])) == 1

    # Empty data fails
    with pytest.raises(ValidationError):
      process(pd.Series([], dtype=float))

  def test_column_explicit_strict(self) -> None:
    """Test Column with explicit strictness."""

    @validate
    def process(
      data: Validated[pd.DataFrame, Column("a", Not(IsNaN), NotEmpty)],
    ) -> int:
      return len(data)

    # NaN fails
    with pytest.raises(ValidationError):
      process(pd.DataFrame({"a": [1, np.nan]}))

    # Empty fails
    with pytest.raises(ValidationError):
      process(pd.DataFrame({"a": []}, dtype=float))

  def test_mixed_column_validation(self) -> None:
    """Test mixed strict and lax columns."""

    @validate
    def process(
      data: Validated[
        pd.DataFrame,
        Column("strict", NotNaN),
        Column("lax"),
      ],
    ) -> int:
      return len(data)

    # Valid case: strict is clean, lax has NaNs
    df_valid = pd.DataFrame({
      "strict": [1, 2, 3],
      "lax": [1, np.nan, 3],
    })
    assert process(df_valid) == 3

    # Fail case: strict has NaNs
    df_fail = pd.DataFrame({
      "strict": [1, np.nan, 3],
      "lax": [1, 2, 3],
    })
    with pytest.raises(ValidationError):
      process(df_fail)
