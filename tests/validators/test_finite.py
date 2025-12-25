"""Tests for Finite and StrictFinite validators."""
# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false
# pyright: reportCallIssue=false, reportAttributeAccessIssue=false

import numpy as np
import pandas as pd
import pytest

from datawarden import Finite, StrictFinite


class TestFinite:
  """Tests for Finite validator (rejects Inf, allows NaN)."""

  def test_validate_with_valid_series_passes(self):
    """Test Finite validator with valid Series."""
    data = pd.Series([1.0, 2.0, 3.0])
    validator = Finite()
    assert validator.validate(data) is None

  def test_validate_with_valid_dataframe_passes(self):
    """Test Finite validator with valid DataFrame."""
    data = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
    validator = Finite()
    assert validator.validate(data) is None

  def test_validate_with_inf_values_raises_error(self):
    """Test Finite validator rejects Inf."""
    data = pd.Series([1.0, np.inf, 3.0])
    validator = Finite()
    with pytest.raises(ValueError, match="must be finite"):
      validator.validate(data)

  def test_validate_with_nan_values_passes(self):
    """Test Finite validator allows NaN (use with Nullable marker)."""
    data = pd.Series([1.0, np.nan, 3.0])
    validator = Finite()
    # Finite allows NaN - the NonNaN check happens via decorator/Nullable
    assert validator.validate(data) is None

  def test_validate_dataframe_with_inf_values_raises_error(self):
    """Test Finite validator rejects DataFrame with Inf."""
    data = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, np.inf]})
    validator = Finite()
    with pytest.raises(ValueError, match="must be finite"):
      validator.validate(data)

  def test_validate_with_empty_series_passes(self):
    """Test Finite validator with empty Series."""
    data = pd.Series([], dtype=float)
    validator = Finite()
    assert validator.validate(data) is None

  def test_validate_with_non_pandas_type_raises_type_error(self):
    """Test Finite validator with non-pandas type raises TypeError."""
    validator = Finite()
    with pytest.raises(TypeError, match="requires pandas"):
      validator.validate(42)


class TestStrictFinite:
  """Tests for StrictFinite validator (rejects both Inf and NaN)."""

  def test_validate_with_valid_series_passes(self):
    """Test StrictFinite validator with valid Series."""
    data = pd.Series([1.0, 2.0, 3.0])
    validator = StrictFinite()
    assert validator.validate(data) is None

  def test_validate_with_inf_values_raises_error(self):
    """Test StrictFinite validator rejects Inf."""
    data = pd.Series([1.0, np.inf, 3.0])
    validator = StrictFinite()
    with pytest.raises(ValueError, match="must be finite"):
      validator.validate(data)

  def test_validate_with_nan_values_raises_error(self):
    """Test StrictFinite validator rejects NaN."""
    data = pd.Series([1.0, np.nan, 3.0])
    validator = StrictFinite()
    with pytest.raises(ValueError, match="must be finite"):
      validator.validate(data)

  def test_validate_dataframe_with_inf_values_raises_error(self):
    """Test StrictFinite validator rejects DataFrame with Inf."""
    data = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, np.inf]})
    validator = StrictFinite()
    with pytest.raises(ValueError, match="must be finite"):
      validator.validate(data)

  def test_validate_dataframe_with_nan_values_raises_error(self):
    """Test StrictFinite validator rejects DataFrame with NaN."""
    data = pd.DataFrame({"a": [1.0, np.nan], "b": [3.0, 4.0]})
    validator = StrictFinite()
    with pytest.raises(ValueError, match="must be finite"):
      validator.validate(data)
