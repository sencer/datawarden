"""Tests for OneOf validator."""

import pandas as pd
import pytest

from datawarden import Validated, validate
from datawarden.exceptions import ValidationError
from datawarden.validators import Index, OneOf


class TestOneOf:
  """Tests for OneOf validator."""

  def test_validate_series_with_valid_strings_passes(self):
    """Test OneOf validator with valid string Series."""

    @validate
    def process(data: Validated[pd.Series, OneOf("a", "b", "c")]) -> pd.Series:
      return data

    data = pd.Series(["a", "b", "a", "c"])
    result = process(data)
    assert result.equals(data)

  def test_validate_series_with_invalid_strings_raises_error(self):
    """Test OneOf validator rejects invalid values."""

    @validate
    def process(data: Validated[pd.Series, OneOf("a", "b", "c")]) -> pd.Series:
      return data

    data = pd.Series(["a", "b", "d"])
    with pytest.raises(ValidationError):
      process(data)

  def test_validate_series_with_valid_numeric_values_passes(self):
    """Test OneOf with numeric values."""

    @validate
    def process(data: Validated[pd.Series, OneOf(1, 2, 3)]) -> pd.Series:
      return data

    data = pd.Series([1, 2, 1, 3])
    result = process(data)
    assert result.equals(data)

  def test_validate_series_with_invalid_numeric_values_raises_error(self):
    """Test OneOf rejects invalid numeric values."""

    @validate
    def process(data: Validated[pd.Series, OneOf(1, 2, 3)]) -> pd.Series:
      return data

    data = pd.Series([1, 2, 4])
    with pytest.raises(ValidationError):
      process(data)

  def test_validate_dataframe_index_with_valid_values_passes(self):
    """Test OneOf with Index[] wrapper for DataFrame index."""

    @validate
    def process(
      df: Validated[pd.DataFrame, Index(OneOf("x", "y", "z"))],
    ) -> pd.DataFrame:
      return df

    df = pd.DataFrame({"a": [1, 2, 3]}, index=["x", "y", "z"])
    result = process(df)
    assert result.equals(df)

  def test_validate_dataframe_index_with_invalid_values_raises_error(self):
    """Test OneOf with Index[] wrapper rejects invalid index."""

    @validate
    def process(
      df: Validated[pd.DataFrame, Index(OneOf("x", "y", "z"))],
    ) -> pd.DataFrame:
      return df

    df = pd.DataFrame({"a": [1, 2, 3]}, index=["x", "y", "w"])
    with pytest.raises(ValidationError):
      process(df)

  def test_validate_series_with_single_allowed_value_passes(self):
    """Test OneOf with single allowed value."""

    @validate
    def process(data: Validated[pd.Series, OneOf("only")]) -> pd.Series:
      return data

    data = pd.Series(["only"])
    result = process(data)
    assert result.equals(data)
