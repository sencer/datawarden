"""Tests for Is lambda validator."""

import pandas as pd
import pytest

from datawarden import Validated, validate
from datawarden.exceptions import ValidationError
from datawarden.validators.value import Is


class TestIsValidator:
  """Tests for Is lambda validator."""

  def test_series_valid(self):
    """Test Is with valid Series."""

    @validate
    def process(data: Validated[pd.Series, Is(lambda x: (x > 0).all())]) -> pd.Series:
      return data

    data = pd.Series([1, 2, 3])
    result = process(data)
    assert result.equals(data)

  def test_series_invalid(self):
    """Test Is with invalid Series."""

    @validate
    def process(data: Validated[pd.Series, Is(lambda x: (x > 0).all())]) -> pd.Series:
      return data

    data = pd.Series([1, -2, 3])
    with pytest.raises(ValidationError):
      process(data)

  def test_dataframe_valid(self):
    """Test Is with valid DataFrame."""

    @validate
    def process(
      df: Validated[pd.DataFrame, Is(lambda d: (d > 0).all().all())],
    ) -> pd.DataFrame:
      return df

    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    result = process(df)
    assert result.equals(df)

  def test_dataframe_invalid(self):
    """Test Is with invalid DataFrame."""

    @validate
    def process(
      df: Validated[pd.DataFrame, Is(lambda d: (d > 0).all().all())],
    ) -> pd.DataFrame:
      return df

    df = pd.DataFrame({"a": [1, -2], "b": [3, 4]})
    with pytest.raises(ValidationError):
      process(df)

  def test_is_with_name(self):
    """Test Is with custom name."""

    @validate
    def process(
      data: Validated[pd.Series, Is(lambda x: (x > 0).all(), name="positive")],
    ) -> pd.Series:
      return data

    data = pd.Series([1, -2, 3])
    with pytest.raises(ValidationError):
      process(data)
