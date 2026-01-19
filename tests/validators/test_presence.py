"""Tests for presence validators: Empty, NotEmpty."""

import pandas as pd
import pytest

from datawarden import Validated, validate
from datawarden.exceptions import ValidationError
from datawarden.validators import Empty, NotEmpty


class TestEmpty:
  """Tests for Empty validator."""

  def test_validate_with_empty_series_passes(self):
    """Test Empty with empty Series."""

    @validate
    def process(data: Validated[pd.Series, Empty]) -> pd.Series:
      return data

    data = pd.Series([], dtype=float)
    result = process(data)
    assert result.equals(data)

  def test_validate_with_non_empty_series_raises_error(self):
    """Test Empty rejects non-empty Series."""

    @validate
    def process(data: Validated[pd.Series, Empty]) -> pd.Series:
      return data

    data = pd.Series([1, 2, 3])
    with pytest.raises(ValidationError):
      process(data)

  def test_validate_with_empty_dataframe_passes(self):
    """Test Empty with empty DataFrame."""

    @validate
    def process(data: Validated[pd.DataFrame, Empty]) -> pd.DataFrame:
      return data

    data = pd.DataFrame()
    result = process(data)
    assert result.equals(data)

  def test_validate_with_non_empty_dataframe_raises_error(self):
    """Test Empty rejects non-empty DataFrame."""

    @validate
    def process(data: Validated[pd.DataFrame, Empty]) -> pd.DataFrame:
      return data

    data = pd.DataFrame({"a": [1]})
    with pytest.raises(ValidationError):
      process(data)


class TestNotEmpty:
  """Tests for NotEmpty validator."""

  def test_validate_with_non_empty_series_passes(self):
    """Test NotEmpty with non-empty Series."""

    @validate
    def process(data: Validated[pd.Series, NotEmpty]) -> pd.Series:
      return data

    data = pd.Series([1, 2, 3])
    result = process(data)
    assert result.equals(data)

  def test_validate_with_empty_series_raises_error(self):
    """Test NotEmpty rejects empty Series."""

    @validate
    def process(data: Validated[pd.Series, NotEmpty]) -> pd.Series:
      return data

    data = pd.Series([], dtype=float)
    with pytest.raises(ValidationError):
      process(data)

  def test_validate_with_non_empty_dataframe_passes(self):
    """Test NotEmpty with non-empty DataFrame."""

    @validate
    def process(data: Validated[pd.DataFrame, NotEmpty]) -> pd.DataFrame:
      return data

    data = pd.DataFrame({"a": [1]})
    result = process(data)
    assert result.equals(data)

  def test_validate_with_empty_dataframe_raises_error(self):
    """Test NotEmpty rejects empty DataFrame."""

    @validate
    def process(data: Validated[pd.DataFrame, NotEmpty]) -> pd.DataFrame:
      return data

    data = pd.DataFrame()
    with pytest.raises(ValidationError):
      process(data)
