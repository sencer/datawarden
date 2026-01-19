"""Tests for validators working on pd.Index objects.

Verifies that NumericValidators and other validators correctly support
pd.Index as input, not just pd.Series and pd.DataFrame.

Ported from datawarden v1 test_standardized_index.py.
"""

import numpy as np
import pandas as pd

from datawarden.context import ValidationContext
from datawarden.validators import (
  Eq,
  Finite,
  Ge,
  Gt,
  Infinite,
  IsNaN,
  Le,
  Lt,
  Ne,
  Negative,
  NonNegative,
  NonPositive,
  NotNaN,
  Positive,
)


def _make_context() -> ValidationContext:
  """Create a minimal validation context for testing."""
  return ValidationContext(root_data=None)


class TestComparisonOnIndex:
  """Tests for comparison validators on pd.Index."""

  def test_ge_on_index(self):
    """Test Ge validator works on pd.Index."""
    ctx = _make_context()
    idx = pd.Index([1, 2, 3])

    # Valid
    res = Ge(0).validate(idx, ctx)
    assert res.success

    # Invalid
    res = Ge(5).validate(idx, ctx)
    assert not res.success

  def test_le_on_index(self):
    """Test Le validator works on pd.Index."""
    ctx = _make_context()
    idx = pd.Index([1, 2, 3])

    # Valid
    res = Le(5).validate(idx, ctx)
    assert res.success

    # Invalid
    res = Le(0).validate(idx, ctx)
    assert not res.success

  def test_gt_on_index(self):
    """Test Gt validator works on pd.Index."""
    ctx = _make_context()
    idx = pd.Index([1, 2, 3])

    # Valid
    res = Gt(0).validate(idx, ctx)
    assert res.success

    # Invalid
    res = Gt(3).validate(idx, ctx)
    assert not res.success

  def test_lt_on_index(self):
    """Test Lt validator works on pd.Index."""
    ctx = _make_context()
    idx = pd.Index([1, 2, 3])

    # Valid
    res = Lt(5).validate(idx, ctx)
    assert res.success

    # Invalid
    res = Lt(1).validate(idx, ctx)
    assert not res.success


class TestNumericOnIndex:
  """Tests for numeric validators on pd.Index."""

  def test_positive_on_index(self):
    """Test Positive validator works on pd.Index."""
    ctx = _make_context()

    # Valid
    idx = pd.Index([1, 2, 3])
    res = Positive().validate(idx, ctx)
    assert res.success

    # Invalid (contains 0)
    idx_zero = pd.Index([0, 1, 2])
    res = Positive().validate(idx_zero, ctx)
    assert not res.success

  def test_negative_on_index(self):
    """Test Negative validator works on pd.Index."""
    ctx = _make_context()

    # Valid
    idx = pd.Index([-1, -2, -3])
    res = Negative().validate(idx, ctx)
    assert res.success

    # Invalid (contains 0)
    idx_zero = pd.Index([-1, 0, -2])
    res = Negative().validate(idx_zero, ctx)
    assert not res.success

  def test_non_negative_on_index(self):
    """Test NonNegative validator works on pd.Index."""
    ctx = _make_context()

    # Valid
    idx = pd.Index([0, 1, 2])
    res = NonNegative().validate(idx, ctx)
    assert res.success

    # Invalid
    idx_neg = pd.Index([1, -1, 2])
    res = NonNegative().validate(idx_neg, ctx)
    assert not res.success

  def test_non_positive_on_index(self):
    """Test NonPositive validator works on pd.Index."""
    ctx = _make_context()

    # Valid
    idx = pd.Index([0, -1, -2])
    res = NonPositive().validate(idx, ctx)
    assert res.success

    # Invalid
    idx_pos = pd.Index([-1, 1, -2])
    res = NonPositive().validate(idx_pos, ctx)
    assert not res.success


class TestFiniteOnIndex:
  """Tests for Finite/Infinite validators on pd.Index."""

  def test_finite_on_index(self):
    """Test Finite validator works on pd.Index."""
    ctx = _make_context()

    # Valid
    idx = pd.Index([1.0, 2.0, 3.0])
    res = Finite().validate(idx, ctx)
    assert res.success

    # Invalid (contains Inf)
    idx_inf = pd.Index([1.0, np.inf, 3.0])
    res = Finite().validate(idx_inf, ctx)
    assert not res.success

  def test_infinite_on_index(self):
    """Test Infinite validator works on pd.Index."""
    ctx = _make_context()

    # Valid (all infinite)
    idx = pd.Index([np.inf, -np.inf, np.inf])
    res = Infinite().validate(idx, ctx)
    assert res.success

    # Invalid (contains finite)
    idx_finite = pd.Index([np.inf, 1.0, -np.inf])
    res = Infinite().validate(idx_finite, ctx)
    assert not res.success


class TestNaNOnIndex:
  """Tests for NaN-related validators on pd.Index."""

  def test_isnan_on_index(self):
    """Test IsNaN validator works on pd.Index."""
    ctx = _make_context()

    # Valid (all NaN)
    idx = pd.Index([np.nan, np.nan, np.nan])
    res = IsNaN().validate(idx, ctx)
    assert res.success

    # Invalid (contains non-NaN)
    idx_mixed = pd.Index([1.0, np.nan, 3.0])
    res = IsNaN().validate(idx_mixed, ctx)
    assert not res.success

  def test_notnan_on_index(self):
    """Test NotNaN validator works on pd.Index."""
    ctx = _make_context()

    # Valid (no NaN)
    idx = pd.Index([1.0, 2.0, 3.0])
    res = NotNaN().validate(idx, ctx)
    assert res.success

    # Invalid (contains NaN)
    idx_nan = pd.Index([1.0, np.nan, 3.0])
    res = NotNaN().validate(idx_nan, ctx)
    assert not res.success


class TestEqualityOnIndex:
  """Tests for Eq/Ne validators on pd.Index."""

  def test_eq_on_index(self):
    """Test Eq validator works on pd.Index."""
    ctx = _make_context()

    # Valid (all equal to 5)
    idx = pd.Index([5, 5, 5])
    res = Eq(5).validate(idx, ctx)
    assert res.success

    # Invalid
    idx_diff = pd.Index([5, 6, 5])
    res = Eq(5).validate(idx_diff, ctx)
    assert not res.success

  def test_ne_on_index(self):
    """Test Ne validator works on pd.Index."""
    ctx = _make_context()

    # Valid (none equal to 5)
    idx = pd.Index([1, 2, 3])
    res = Ne(5).validate(idx, ctx)
    assert res.success

    # Invalid (contains 5)
    idx_has_5 = pd.Index([1, 5, 3])
    res = Ne(5).validate(idx_has_5, ctx)
    assert not res.success


class TestNegationOnIndex:
  """Tests for negated validators on pd.Index."""

  def test_not_positive_on_index(self):
    """Test ~Positive() (NonPositive) works on pd.Index."""
    ctx = _make_context()
    v = ~Positive()
    # Valid - all non-positive
    idx = pd.Index([0, -1, -2])
    res = v.validate(idx, ctx)
    assert res.success

    # Invalid (has positive)
    idx_pos = pd.Index([-1, 1, -2])
    res = v.validate(idx_pos, ctx)
    assert not res.success

  def test_not_negative_on_index(self):
    """Test ~Negative() (NonNegative) works on pd.Index."""
    ctx = _make_context()
    v = ~Negative()
    # Valid - all non-negative
    idx = pd.Index([0, 1, 2])
    res = v.validate(idx, ctx)
    assert res.success

    # Invalid (has negative)
    idx_neg = pd.Index([1, -1, 2])
    res = v.validate(idx_neg, ctx)
    assert not res.success


class TestIndexWithMask:
  """Tests that validation results include proper mask for Index."""

  def test_mask_returned_for_failed_index(self):
    """Test that failed validation returns mask with correct shape."""
    ctx = _make_context()
    idx = pd.Index([1, -2, 3, -4])

    res = Positive().validate(idx, ctx)
    assert not res.success
    assert res.mask is not None
    # Mask should be a Series with same length
    assert len(res.mask) == 4
    # Mask values: [True, False, True, False]
    np.testing.assert_array_equal(res.mask.values, [True, False, True, False])
