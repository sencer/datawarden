"""Tests for presence validators: NotNaN and NotEmpty."""
# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false
# pyright: reportCallIssue=false, reportAttributeAccessIssue=false

import numpy as np
import pandas as pd
import pytest

from datawarden import Empty, IsNaN, Not, NotEmpty, NotNaN


class TestNotNaN:
  """Tests for NotNaN validator."""

  def test_validate_with_valid_series_passes(self):
    """Test NotNaN validator with valid Series."""
    data = pd.Series([1.0, 2.0, 3.0])
    validator = NotNaN()
    assert validator.validate(data) is None

  def test_validate_with_nan_values_raises_error(self):
    """Test Not(IsNaN) validator rejects NaN."""
    data = pd.Series([1.0, np.nan, 3.0])
    validator = NotNaN()
    with pytest.raises(ValueError, match="Data must not contain NaN"):
      validator.validate(data)

  def test_validate_with_inf_values_passes(self):
    """Test Not(IsNaN) validator allows Inf."""
    data = pd.Series([1.0, np.inf, 3.0])
    validator = NotNaN()
    assert validator.validate(data) is None

  def test_validate_with_empty_series_passes(self):
    """Test Not(IsNaN) validator with empty Series."""
    data = pd.Series([], dtype=float)
    validator = Not(IsNaN())
    assert validator.validate(data) is None


class TestNotEmpty:
  """Tests for NotEmpty validator."""

  def test_validate_with_valid_series_passes(self):
    """Test NotEmpty validator with valid Series."""
    data = pd.Series([1.0, 2.0, 3.0])
    validator = NotEmpty()
    assert validator.validate(data) is None

  def test_validate_with_empty_series_raises_error(self):
    """Test NotEmpty validator rejects empty Series."""
    data = pd.Series([], dtype=float)
    validator = NotEmpty()
    with pytest.raises(ValueError, match="Data must be non-empty"):
      validator.validate(data)

  def test_validate_with_valid_dataframe_passes(self):
    """Test NotEmpty validator with valid DataFrame."""
    data = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
    validator = NotEmpty()
    assert validator.validate(data) is None

  def test_validate_with_empty_dataframe_raises_error(self):
    """Test NotEmpty validator rejects empty DataFrame."""
    data = pd.DataFrame({"a": [], "b": []})
    validator = NotEmpty()
    with pytest.raises(ValueError, match="Data must be non-empty"):
      validator.validate(data)

  def test_validate_with_valid_index_passes(self):
    """Test NotEmpty validator with valid Index."""
    data = pd.Index([1, 2, 3])
    validator = NotEmpty()
    assert validator.validate(data) is None

  def test_validate_with_empty_index_raises_error(self):
    """Test NotEmpty validator rejects empty Index."""
    data = pd.Index([], dtype=int)
    validator = NotEmpty()
    with pytest.raises(ValueError, match="Data must be non-empty"):
      validator.validate(data)

  def test_validate_with_non_pandas_type_raises_type_error(self):
    """Test NotEmpty validator with non-pandas type raises TypeError."""
    validator = NotEmpty()
    with pytest.raises(TypeError, match="requires pandas"):
      validator.validate(42)

  def test_empty_not_empty_negate(self):
    e = Empty()
    assert isinstance(e.negate(), NotEmpty)

    ne = NotEmpty()
    assert isinstance(ne.negate(), Empty)
