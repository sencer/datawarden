"""Tests for numeric validators: Positive, Negative, Not(Positive), Not(Negative)."""

import pandas as pd
import pytest

from datawarden import Validated, validate
from datawarden.exceptions import ValidationError
from datawarden.validators import Negative, NonNegative, NonPositive, Positive


class TestPositive:
  """Tests for Positive validator."""

  def test_validate_with_valid_series_passes(self) -> None:
    """Test Positive validator with valid Series."""

    @validate
    def process(data: Validated[pd.Series, Positive]) -> pd.Series:
      return data

    data = pd.Series([1.0, 2.0, 3.0])
    result = process(data)
    assert result.equals(data)

  def test_validate_with_zero_values_raises_error(self) -> None:
    """Test Positive validator rejects zero."""

    @validate
    def process(data: Validated[pd.Series, Positive]) -> pd.Series:
      return data

    data = pd.Series([1.0, 0.0, 3.0])
    with pytest.raises(ValidationError):
      process(data)

  def test_validate_with_negative_values_raises_error(self) -> None:
    """Test Positive validator rejects negative values."""

    @validate
    def process(data: Validated[pd.Series, Positive]) -> pd.Series:
      return data

    data = pd.Series([1.0, -1.0, 3.0])
    with pytest.raises(ValidationError):
      process(data)

  def test_validate_with_all_positive_values_passes(self) -> None:
    """Test Positive validator with all positive values."""

    @validate
    def process(data: Validated[pd.Series, Positive]) -> pd.Series:
      return data

    data = pd.Series([0.1, 100.0, 0.001])
    result = process(data)
    assert result.equals(data)


class TestNegative:
  """Tests for Negative validator."""

  def test_validate_with_valid_series_passes(self) -> None:
    """Test Negative validator with valid Series."""

    @validate
    def process(data: Validated[pd.Series, Negative]) -> pd.Series:
      return data

    data = pd.Series([-1.0, -2.0, -3.0])
    result = process(data)
    assert result.equals(data)

  def test_validate_with_zero_values_raises_error(self) -> None:
    """Test Negative validator rejects zero."""

    @validate
    def process(data: Validated[pd.Series, Negative]) -> pd.Series:
      return data

    data = pd.Series([-1.0, 0.0, -3.0])
    with pytest.raises(ValidationError):
      process(data)

  def test_validate_with_positive_values_raises_error(self) -> None:
    """Test Negative validator rejects positive values."""

    @validate
    def process(data: Validated[pd.Series, Negative]) -> pd.Series:
      return data

    data = pd.Series([-1.0, 1.0, -3.0])
    with pytest.raises(ValidationError):
      process(data)

  def test_validate_with_all_negative_values_passes(self) -> None:
    """Test Negative validator with all negative values."""

    @validate
    def process(data: Validated[pd.Series, Negative]) -> pd.Series:
      return data

    data = pd.Series([-0.1, -100.0, -0.001])
    result = process(data)
    assert result.equals(data)

  def test_validate_with_valid_dataframe_passes(self) -> None:
    """Test Negative validator with DataFrame."""

    @validate
    def process(data: Validated[pd.DataFrame, Negative]) -> pd.DataFrame:
      return data

    data = pd.DataFrame({"a": [-1.0, -2.0], "b": [-3.0, -4.0]})
    result = process(data)
    assert result.equals(data)


class TestNonPositive:
  """Tests for NonPositive validator."""

  def test_validate_with_valid_series_passes(self) -> None:
    """Test NonPositive validator with valid Series."""

    @validate
    def process(data: Validated[pd.Series, NonPositive]) -> pd.Series:
      return data

    data = pd.Series([0.0, -1.0, -2.0])
    result = process(data)
    assert result.equals(data)

  def test_validate_with_positive_values_raises_error(self) -> None:
    """Test NonPositive validator rejects positive values."""

    @validate
    def process(data: Validated[pd.Series, NonPositive]) -> pd.Series:
      return data

    data = pd.Series([-1.0, 1.0, -3.0])
    with pytest.raises(ValidationError):
      process(data)

  def test_validate_with_zero_values_passes(self) -> None:
    """Test NonPositive validator allows zero."""

    @validate
    def process(data: Validated[pd.Series, NonPositive]) -> pd.Series:
      return data

    data = pd.Series([0.0, 0.0, 0.0])
    result = process(data)
    assert result.equals(data)


class TestNonNegative:
  """Tests for NonNegative validator."""

  def test_validate_with_valid_series_passes(self) -> None:
    """Test NonNegative validator with valid Series."""

    @validate
    def process(data: Validated[pd.Series, NonNegative]) -> pd.Series:
      return data

    data = pd.Series([0.0, 1.0, 2.0])
    result = process(data)
    assert result.equals(data)

  def test_validate_with_negative_values_raises_error(self) -> None:
    """Test NonNegative validator rejects negative values."""

    @validate
    def process(data: Validated[pd.Series, NonNegative]) -> pd.Series:
      return data

    data = pd.Series([1.0, -1.0, 3.0])
    with pytest.raises(ValidationError):
      process(data)

  def test_validate_with_zero_values_passes(self) -> None:
    """Test NonNegative validator allows zero."""

    @validate
    def process(data: Validated[pd.Series, NonNegative]) -> pd.Series:
      return data

    data = pd.Series([0.0, 0.0, 0.0])
    result = process(data)
    assert result.equals(data)

  def test_validate_with_valid_dataframe_passes(self) -> None:
    """Test NonNegative validator with DataFrame."""

    @validate
    def process(data: Validated[pd.DataFrame, NonNegative]) -> pd.DataFrame:
      return data

    data = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
    result = process(data)
    assert result.equals(data)
