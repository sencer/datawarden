"""Tests for numeric validators: Not(Negative) and Positive."""
# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false
# pyright: reportCallIssue=false, reportAttributeAccessIssue=false

import numpy as np
import pandas as pd
import pytest

from datawarden import Negative, Not, Positive


class TestNotNegative:
  """Tests for Not(Negative) validator."""

  def test_validate_with_valid_series_passes(self):
    """Test Not(Negative) validator with valid Series."""
    data = pd.Series([0.0, 1.0, 2.0])
    validator = Not(Negative())
    assert validator.validate(data) is None

  def test_validate_with_negative_values_raises_error(self):
    """Test Not(Negative) validator rejects negative values."""
    data = pd.Series([1.0, -1.0, 3.0])
    validator = Not(Negative())
    with pytest.raises(ValueError, match="must be >= 0"):
      validator.validate(data)

  def test_validate_with_zero_values_passes(self):
    """Test Not(Negative) validator allows zero."""
    data = pd.Series([0.0, 0.0, 0.0])
    validator = Not(Negative())
    assert validator.validate(data) is None

  def test_validate_with_valid_dataframe_passes(self):
    """Test Not(Negative) validator with DataFrame."""
    data = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
    validator = Not(Negative())
    assert validator.validate(data) is None

  def test_validate_with_nan_values_raises_error(self):
    """Test Not(Negative) validator rejects NaN values (without wrapper)."""
    data = pd.Series([1.0, np.nan, 3.0])
    validator = Not(Negative())
    with pytest.raises(ValueError, match="Cannot perform >= comparison with NaN"):
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


class TestNegative:
  """Tests for Negative validator."""

  def test_validate_with_valid_series_passes(self):
    """Test Negative validator with valid Series."""
    data = pd.Series([-1.0, -2.0, -3.0])
    validator = Negative()
    assert validator.validate(data) is None

  def test_validate_with_zero_values_raises_error(self):
    """Test Negative validator rejects zero."""
    data = pd.Series([-1.0, 0.0, -3.0])
    validator = Negative()
    with pytest.raises(ValueError, match="must be negative"):
      validator.validate(data)

  def test_validate_with_positive_values_raises_error(self):
    """Test Negative validator rejects positive values."""
    data = pd.Series([-1.0, 1.0, -3.0])
    validator = Negative()
    with pytest.raises(ValueError, match="must be negative"):
      validator.validate(data)

  def test_validate_with_all_negative_values_passes(self):
    """Test Negative validator with all negative values."""
    data = pd.Series([-0.1, -100.0, -0.001])
    validator = Negative()
    assert validator.validate(data) is None

  def test_validate_with_nan_values_raises_error(self):
    """Test that Negative rejects NaN (without wrapper)."""
    v = Negative()
    with pytest.raises(ValueError, match="Cannot validate negative with NaN"):
      v.validate(pd.Series([-1, np.nan]))

  def test_validate_with_valid_dataframe_passes(self):
    """Test Negative validator with DataFrame."""
    data = pd.DataFrame({"a": [-1.0, -2.0], "b": [-3.0, -4.0]})
    validator = Negative()
    assert validator.validate(data) is None


class TestNotPositive:
  """Tests for Not(Positive) validator."""

  def test_validate_with_valid_series_passes(self):
    """Test Not(Positive) validator with valid Series."""
    data = pd.Series([0.0, -1.0, -2.0])
    validator = Not(Positive())
    assert validator.validate(data) is None

  def test_validate_with_positive_values_raises_error(self):
    """Test Not(Positive) validator rejects positive values."""
    data = pd.Series([-1.0, 1.0, -3.0])
    validator = Not(Positive())
    with pytest.raises(ValueError, match="must be <= 0"):
      validator.validate(data)

  def test_validate_with_zero_values_passes(self):
    """Test Not(Positive) validator allows zero."""
    data = pd.Series([0.0, 0.0, 0.0])
    validator = Not(Positive())
    assert validator.validate(data) is None

  def test_validate_with_valid_dataframe_passes(self):
    """Test Not(Positive) validator with DataFrame."""
    data = pd.DataFrame({"a": [-1.0, 0.0], "b": [-3.0, -4.0]})
    validator = Not(Positive())
    assert validator.validate(data) is None

  def test_validate_with_nan_values_raises_error(self):
    """Test Not(Positive) validator rejects NaN values (without wrapper)."""
    data = pd.Series([-1.0, np.nan, -3.0])
    validator = Not(Positive())
    with pytest.raises(ValueError, match="Cannot perform <= comparison with NaN"):
      validator.validate(data)
