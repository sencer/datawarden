"""Tests for Is and Rows lambda validators."""

from typing import Annotated

import pandas as pd
import pytest

from datawarden import Is, Rows, validate


class TestIsValidator:
  """Tests for Is element-wise predicate validator."""

  def test_series_valid(self):
    """Test Is with valid Series data."""
    validator = Is(lambda x: (x >= 0) & (x <= 100))
    data = pd.Series([0, 50, 100])
    validator = Is(lambda x: (x >= 0) & (x <= 100))
    data = pd.Series([0, 50, 100])
    assert validator.validate(data) is None

  def test_series_invalid(self):
    """Test Is with invalid Series data."""
    validator = Is(lambda x: x > 0)
    data = pd.Series([1, 2, -3, 4])
    with pytest.raises(ValueError, match="1 values failed"):
      validator.validate(data)

  def test_series_with_name(self):
    """Test Is with custom error name."""
    validator = Is(lambda x: x > 0, name="values must be positive")
    data = pd.Series([1, 2, -3])
    with pytest.raises(ValueError, match="values must be positive"):
      validator.validate(data)

  def test_dataframe_valid(self):
    """Test Is with valid DataFrame data."""
    validator = Is(lambda x: x >= 0)
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    validator = Is(lambda x: x >= 0)
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    assert validator.validate(df) is None

  def test_dataframe_invalid_column(self):
    """Test Is with invalid DataFrame column."""
    validator = Is(lambda x: x > 0)
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, -5, 6]})
    with pytest.raises(ValueError, match="Column 'b' failed"):
      validator.validate(df)

  def test_index_valid(self):
    """Test Is with valid Index."""
    validator = Is(lambda x: x > 0)
    idx = pd.Index([1, 2, 3])
    validator = Is(lambda x: x > 0)
    idx = pd.Index([1, 2, 3])
    assert validator.validate(idx) is None

  def test_index_invalid(self):
    """Test Is with invalid Index."""
    validator = Is(lambda x: x > 0)
    idx = pd.Index([1, 2, -3])
    with pytest.raises(ValueError, match="Index values failed"):
      validator.validate(idx)

  def test_with_decorator(self):
    """Test Is validator with @validate decorator."""

    @validate
    def process(data: Annotated[pd.Series, Is(lambda x: x**2 < 100)]) -> pd.Series:
      return data

    valid = pd.Series([1, 2, 3, 9])
    result = process(valid)
    assert result.equals(valid)

    invalid = pd.Series([1, 2, 10])  # 10**2 = 100, not < 100
    with pytest.raises(ValueError, match="1 values failed"):
      process(invalid)


class TestRowsValidator:
  """Tests for Rows row-wise predicate validator."""

  def test_valid_rows(self):
    """Test Rows with all valid rows."""
    validator = Rows(lambda row: row.sum() < 100)
    df = pd.DataFrame({"a": [10, 20], "b": [20, 30]})
    validator = Rows(lambda row: row.sum() < 100)
    df = pd.DataFrame({"a": [10, 20], "b": [20, 30]})
    assert validator.validate(df) is None

  def test_invalid_rows(self):
    """Test Rows with invalid rows."""
    validator = Rows(lambda row: row.sum() < 100)
    df = pd.DataFrame({"a": [10, 60], "b": [20, 50]})  # Row 1: 110 > 100
    with pytest.raises(ValueError, match="1 rows failed"):
      validator.validate(df)

  def test_with_name(self):
    """Test Rows with custom error name."""
    validator = Rows(lambda row: row["high"] >= row["low"], name="high must be >= low")
    df = pd.DataFrame({"high": [10, 5], "low": [5, 10]})  # Row 1: 5 < 10
    with pytest.raises(ValueError, match="high must be >= low"):
      validator.validate(df)

  def test_column_comparison(self):
    """Test Rows for column comparisons."""
    validator = Rows(lambda row: row["high"] >= row["low"])
    valid_df = pd.DataFrame({"high": [10, 20, 30], "low": [5, 15, 25]})
    validator = Rows(lambda row: row["high"] >= row["low"])
    valid_df = pd.DataFrame({"high": [10, 20, 30], "low": [5, 15, 25]})
    assert validator.validate(valid_df) is None

    invalid_df = pd.DataFrame({"high": [10, 5], "low": [5, 10]})
    with pytest.raises(ValueError, match="1 rows failed"):
      validator.validate(invalid_df)

  def test_non_dataframe_raises(self):
    """Test Rows raises TypeError for non-DataFrame."""
    validator = Rows(lambda row: row.sum() < 100)
    with pytest.raises(TypeError, match="requires a pandas DataFrame"):
      validator.validate(pd.Series([1, 2, 3]))

  def test_with_decorator(self):
    """Test Rows validator with @validate decorator."""

    @validate
    def process(
      data: Annotated[pd.DataFrame, Rows(lambda row: row["a"] + row["b"] < 100)],
    ) -> pd.DataFrame:
      return data

    valid = pd.DataFrame({"a": [10, 20], "b": [20, 30]})
    result = process(valid)
    assert result.equals(valid)

    invalid = pd.DataFrame({"a": [10, 60], "b": [20, 50]})
    with pytest.raises(ValueError, match="1 rows failed"):
      process(invalid)

  def test_shows_failed_indices(self):
    """Test that error message shows failed row indices."""
    validator = Rows(lambda row: row.sum() < 10)
    df = pd.DataFrame({"a": [1, 5, 1], "b": [1, 5, 1]}, index=["x", "y", "z"])
    with pytest.raises(ValueError, match=r"at indices.*y"):
      validator.validate(df)
