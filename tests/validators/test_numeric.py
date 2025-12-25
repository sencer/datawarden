"""Tests for numeric validators: NonNegative and Positive."""
# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false
# pyright: reportCallIssue=false, reportAttributeAccessIssue=false

import numpy as np
import pandas as pd
import pytest

from datawarden import NonNegative, Positive


class TestNonNegative:
  """Tests for NonNegative validator."""

  def test_validate_with_valid_series_passes(self):
    """Test NonNegative validator with valid Series."""
    data = pd.Series([0.0, 1.0, 2.0])
    validator = NonNegative()
    assert validator.validate(data) is None

  def test_validate_with_negative_values_raises_error(self):
    """Test NonNegative validator rejects negative values."""
    data = pd.Series([1.0, -1.0, 3.0])
    validator = NonNegative()
    with pytest.raises(ValueError, match="must be non-negative"):
      validator.validate(data)

  def test_validate_with_zero_values_passes(self):
    """Test NonNegative validator allows zero."""
    data = pd.Series([0.0, 0.0, 0.0])
    validator = NonNegative()
    assert validator.validate(data) is None

  def test_validate_with_valid_dataframe_passes(self):
    """Test NonNegative validator with DataFrame."""
    data = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
    validator = NonNegative()
    assert validator.validate(data) is None

  def test_validate_with_nan_values_raises_error(self):
    """Test NonNegative validator rejects NaN values (without wrapper)."""
    data = pd.Series([1.0, np.nan, 3.0])
    validator = NonNegative()
    with pytest.raises(ValueError, match="Cannot validate non-negative with NaN"):
      validator.validate(data)


class TestPositive:
  """Tests for Positive validator."""

  def test_validate_with_valid_series_passes(self):
    """Test Positive validator with valid Series."""
    data = pd.Series([1.0, 2.0, 3.0])
    validator = Positive()
    assert validator.validate(data) is None

  def test_validate_with_zero_values_raises_error(self):
    """Test Positive validator rejects zero."""
    data = pd.Series([1.0, 0.0, 3.0])
    validator = Positive()
    with pytest.raises(ValueError, match="must be positive"):
      validator.validate(data)

  def test_validate_with_negative_values_raises_error(self):
    """Test Positive validator rejects negative values."""
    data = pd.Series([1.0, -1.0, 3.0])
    validator = Positive()
    with pytest.raises(ValueError, match="must be positive"):
      validator.validate(data)

  def test_validate_with_all_positive_values_passes(self):
    """Test Positive validator with all positive values."""
    data = pd.Series([0.1, 100.0, 0.001])
    validator = Positive()
    assert validator.validate(data) is None

  def test_validate_with_nan_values_raises_error(self):
    """Test that Positive rejects NaN (without wrapper)."""
    v = Positive()
    with pytest.raises(ValueError, match="Cannot validate positive with NaN"):
      v.validate(pd.Series([1, np.nan]))
