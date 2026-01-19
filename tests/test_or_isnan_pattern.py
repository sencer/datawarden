"""Tests for Or | IsNaN pattern (replacement for IgnoringNaNs).

Verifies that `A | IsNaN` correctly allows NaN values while still
validating non-NaN values against the constraint.

Ported from datawarden v1 test_nan_handling.py.
"""

import numpy as np
import pandas as pd
import pytest

from datawarden import Validated, validate
from datawarden.context import ValidationContext
from datawarden.exceptions import ValidationError
from datawarden.validators import Ge, IsNaN, Le, Positive
from datawarden.validators.base import Or


class TestOrIsNaNBasic:
  """Basic tests for A | IsNaN pattern."""

  def test_or_isnan_allows_nan_values(self):
    """Test that A | IsNaN allows NaN values."""
    v = Or(Ge(0), IsNaN)
    ctx = _make_context()

    # Data with NaN should pass
    data = pd.Series([1.0, np.nan, 3.0, np.nan])
    res = v.validate(data, ctx)
    assert res.success

  def test_or_isnan_validates_non_nan(self):
    """Test that A | IsNaN still validates non-NaN values."""
    v = Or(Ge(0), IsNaN)
    ctx = _make_context()

    # Non-NaN values that pass constraint
    res = v.validate(pd.Series([1.0, 2.0, 3.0]), ctx)
    assert res.success

    # Non-NaN values that fail constraint
    res = v.validate(pd.Series([-1.0, -2.0]), ctx)
    assert not res.success

  def test_or_isnan_mixed_valid(self):
    """Test A | IsNaN with valid NaN and non-NaN values."""
    v = Or(Ge(0), IsNaN)
    ctx = _make_context()

    # Mix of NaN (valid) and positive (valid)
    data = pd.Series([1.0, np.nan, 3.0])
    res = v.validate(data, ctx)
    assert res.success

  def test_or_isnan_mixed_invalid(self):
    """Test A | IsNaN rejects invalid non-NaN values."""
    v = Or(Ge(0), IsNaN)
    ctx = _make_context()

    # Mix of valid (NaN, positive) and invalid (negative)
    data = pd.Series([1.0, np.nan, -3.0])
    res = v.validate(data, ctx)
    assert not res.success

  def test_or_isnan_all_nan(self):
    """Test that all-NaN data passes with A | IsNaN."""
    v = Or(Ge(0), IsNaN)
    ctx = _make_context()

    data = pd.Series([np.nan, np.nan, np.nan])
    res = v.validate(data, ctx)
    assert res.success


class TestOrIsNaNWithDecorator:
  """Tests for A | IsNaN pattern with @validate decorator."""

  def test_decorator_allows_nan(self):
    """Test @validate with A | IsNaN allows NaN values."""

    @validate
    def process(
      data: Validated[pd.Series, Or(Ge(0), IsNaN)],
    ) -> pd.Series:
      return data

    # Should pass
    result = process(pd.Series([1.0, np.nan, 3.0]))
    assert len(result) == 3

  def test_decorator_rejects_invalid_non_nan(self):
    """Test @validate with A | IsNaN rejects invalid non-NaN."""

    @validate
    def process(
      data: Validated[pd.Series, Or(Ge(0), IsNaN)],
    ) -> pd.Series:
      return data

    # Should fail due to -1
    with pytest.raises(ValidationError):
      process(pd.Series([1.0, np.nan, -1.0]))


class TestOrIsNaNDataFrame:
  """Tests for A | IsNaN pattern on DataFrames."""

  def test_dataframe_with_nans(self):
    """Test A | IsNaN on DataFrame with NaN values."""
    v = Or(Ge(0), IsNaN)
    ctx = _make_context()

    df = pd.DataFrame({
      "a": [1.0, np.nan, 3.0],
      "b": [np.nan, 2.0, np.nan],
    })
    res = v.validate(df, ctx)
    assert res.success

  def test_dataframe_fails_invalid(self):
    """Test A | IsNaN on DataFrame fails on invalid non-NaN."""
    v = Or(Ge(0), IsNaN)
    ctx = _make_context()

    df = pd.DataFrame({
      "a": [1.0, np.nan, -3.0],  # -3 fails
      "b": [np.nan, 2.0, np.nan],
    })
    res = v.validate(df, ctx)
    assert not res.success


class TestOrIsNaNComposition:
  """Tests for composing Or | IsNaN with other validators."""

  def test_and_with_or_isnan(self):
    """Test And(A | IsNaN, B) composition."""
    # Value must be >= 0 (or NaN) AND <= 100 (or NaN)
    v = Or(Ge(0), IsNaN) & Or(Le(100), IsNaN)
    ctx = _make_context()

    # Valid: in range or NaN
    res = v.validate(pd.Series([50.0, np.nan, 100.0]), ctx)
    assert res.success

    # Invalid: > 100
    res = v.validate(pd.Series([50.0, 150.0]), ctx)
    assert not res.success

  def test_complex_or_pattern(self):
    """Test complex Or pattern with multiple conditions."""
    # Valid if: Positive OR IsNaN
    v = Or(Positive, IsNaN)
    ctx = _make_context()

    res = v.validate(pd.Series([1.0, np.nan, 5.0]), ctx)
    assert res.success

    assert not v.validate(pd.Series([0.0]), ctx).success  # 0 is not positive


def _make_context() -> ValidationContext:
  """Create a minimal validation context for testing."""
  return ValidationContext(root_data=None)
