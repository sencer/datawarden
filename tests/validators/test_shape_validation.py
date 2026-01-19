"""Tests for Shape validator."""

import pandas as pd
import pytest

from datawarden import Validated, validate
from datawarden.exceptions import ValidationError
from datawarden.validators import Shape


class TestShapeDataFrame:
  """Tests for Shape validator with DataFrames."""

  def test_validate_exact_shape_passes(self):
    """Test Shape with exact dimensions."""

    @validate
    def process(df: Validated[pd.DataFrame, Shape(rows=3, cols=2)]) -> pd.DataFrame:
      return df

    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    result = process(df)
    assert result.equals(df)

  def test_validate_exact_shape_incorrect_rows_raises_error(self):
    """Test Shape fails when rows don't match."""

    @validate
    def process(df: Validated[pd.DataFrame, Shape(rows=3, cols=2)]) -> pd.DataFrame:
      return df

    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    with pytest.raises(ValidationError):
      process(df)

  def test_validate_exact_shape_incorrect_cols_raises_error(self):
    """Test Shape fails when cols don't match."""

    @validate
    def process(df: Validated[pd.DataFrame, Shape(rows=3, cols=2)]) -> pd.DataFrame:
      return df

    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6], "c": [7, 8, 9]})
    with pytest.raises(ValidationError):
      process(df)

  def test_validate_rows_only(self):
    """Test Shape with only rows constraint."""

    @validate
    def process(df: Validated[pd.DataFrame, Shape(rows=5)]) -> pd.DataFrame:
      return df

    df = pd.DataFrame({"a": range(5), "b": range(5), "c": range(5)})
    result = process(df)
    assert result.equals(df)


class TestShapeSeries:
  """Tests for Shape validator with Series."""

  def test_validate_series_exact_shape_passes(self):
    """Test Shape with Series and exact row count."""

    @validate
    def process(data: Validated[pd.Series, Shape(rows=5)]) -> pd.Series:
      return data

    series = pd.Series([1, 2, 3, 4, 5])
    result = process(series)
    assert result.equals(series)

  def test_validate_series_exact_shape_incorrect_rows_raises_error(self):
    """Test Shape with Series fails when rows don't match."""

    @validate
    def process(data: Validated[pd.Series, Shape(rows=5)]) -> pd.Series:
      return data

    series = pd.Series([1, 2, 3])
    with pytest.raises(ValidationError):
      process(series)


class TestShapeFailures:
  """Tests for Shape validator failure cases."""

  def test_shape_row_failure(self):
    """Test Shape fails on row mismatch with specific error."""

    @validate
    def process(data: Validated[pd.Series, Shape(rows=5)]) -> pd.Series:
      return data

    with pytest.raises(ValidationError):
      process(pd.Series([1, 2]))

  def test_shape_col_failure(self):
    """Test Shape fails on col mismatch."""

    @validate
    def process(df: Validated[pd.DataFrame, Shape(rows=5, cols=2)]) -> pd.DataFrame:
      return df

    with pytest.raises(ValidationError):
      process(pd.DataFrame({"a": [1, 2, 3, 4, 5]}))
