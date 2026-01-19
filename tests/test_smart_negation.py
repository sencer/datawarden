"""Tests for smart negation (negate() methods) in validators.

Tests verify that negate() returns optimized validators instead of generic Not().
For example, Ge(5).negate() should return Lt(5), not Not(Ge(5)).
"""

import numpy as np
import pandas as pd

from datawarden.context import ValidationContext
from datawarden.validators import (
  Eq,
  Ge,
  Gt,
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
from datawarden.validators.base import Not


class TestSmartNegation:
  """Tests for smart negation via negate() methods."""

  def test_not_positive_uses_smart_negation(self):
    """Test that ~Positive() uses NonPositive via smart negation."""
    v = ~Positive()
    assert isinstance(v, NonPositive)
    # Verify it's Le(0) effectively
    assert v.value == 0

  def test_not_negative_uses_smart_negation(self):
    """Test that ~Negative() uses NonNegative via smart negation."""
    v = ~Negative()
    assert isinstance(v, NonNegative)
    # Verify it's Ge(0) effectively
    assert v.value == 0

  def test_not_nonpositive_uses_smart_negation(self):
    """Test that ~NonPositive() uses Positive via smart negation."""
    v = ~NonPositive()
    assert isinstance(v, Positive)

  def test_not_nonnegative_uses_smart_negation(self):
    """Test that ~NonNegative() uses Negative via smart negation."""
    v = ~NonNegative()
    assert isinstance(v, Negative)


class TestComparisonNegation:
  """Tests for comparison operator negation."""

  def test_ge_negates_to_lt(self):
    """Test Ge(5).negate() returns Lt(5)."""
    v = Ge(5)
    neg = v.negate()
    assert isinstance(neg, Lt)
    assert neg.value == 5

  def test_lt_negates_to_ge(self):
    """Test Lt(5).negate() returns Ge(5)."""
    v = Lt(5)
    neg = v.negate()
    assert isinstance(neg, Ge)
    assert neg.value == 5

  def test_le_negates_to_gt(self):
    """Test Le(5).negate() returns Gt(5)."""
    v = Le(5)
    neg = v.negate()
    assert isinstance(neg, Gt)
    assert neg.value == 5

  def test_gt_negates_to_le(self):
    """Test Gt(5).negate() returns Le(5)."""
    v = Gt(5)
    neg = v.negate()
    assert isinstance(neg, Le)
    assert neg.value == 5

  def test_eq_negates_to_ne(self):
    """Test Eq(5).negate() returns Ne(5)."""
    v = Eq(5)
    neg = v.negate()
    assert isinstance(neg, Ne)
    assert neg.value == 5

  def test_ne_negates_to_eq(self):
    """Test Ne(5).negate() returns Eq(5)."""
    v = Ne(5)
    neg = v.negate()
    assert isinstance(neg, Eq)
    assert neg.value == 5

  def test_isnan_negates_to_notnan(self):
    """Test IsNaN().negate() returns NotNaN."""
    neg = IsNaN().negate()
    assert isinstance(neg, NotNaN)

  def test_notnan_negates_to_isnan(self):
    """Test NotNaN().negate() returns IsNaN."""
    neg = NotNaN().negate()
    assert isinstance(neg, IsNaN)

  def test_not_ge_uses_smart_negation(self):
    """Test ~Ge(5) uses Lt(5) via smart negation."""
    v = ~Ge(5)
    assert isinstance(v, Lt)

  def test_not_lt_uses_smart_negation(self):
    """Test ~Lt(5) uses Ge(5) via smart negation."""
    v = ~Lt(5)
    assert isinstance(v, Ge)

  def test_nary_comparison_uses_smart_negation(self):
    """Test that N-ary comparisons use smart negation."""
    # Ge("a", "b") -> Lt("a", "b")
    v = Ge("a", "b")
    neg = v.negate()
    assert isinstance(neg, Lt)
    assert list(neg.targets) == ["a", "b"]

    # N-ary negation for N > 2 should fall back to generic Not to preserve De Morgan's laws
    v2 = Lt("x", "y", "z")
    neg2 = v2.negate()
    assert isinstance(neg2, Not)

    # Le("col1", "col2") -> Gt("col1", "col2")
    v4 = Le("col1", "col2")
    neg4 = v4.negate()
    assert isinstance(neg4, Gt)
    assert list(neg4.targets) == ["col1", "col2"]


class TestNegationIntegration:
  """Integration tests for smart negation with validation."""

  def test_negated_positive_validates_correctly(self):
    """Test ~Positive() validates non-positive values."""
    v = ~Positive()
    ctx = _make_context()

    # Valid: non-positive values
    res = v.validate(pd.Series([0, -1, -5]), ctx)
    assert res.success

    # Invalid: positive value
    res = v.validate(pd.Series([0, 1, -1]), ctx)
    assert not res.success

  def test_negated_ge_validates_correctly(self):
    """Test ~Ge(0) validates negative values."""
    v = ~Ge(0)
    ctx = _make_context()

    # Valid: negative values only
    res = v.validate(pd.Series([-1, -2, -3]), ctx)
    assert res.success
    # Invalid - contains zero or positive
    res = v.validate(pd.Series([-1, 0, -2]), ctx)
    assert not res.success


class TestDoubleNegation:
  """Tests for double negation behavior."""

  def test_double_negation_ge(self):
    """Test ~~Ge(10) effectively becomes Ge(10)."""
    v = Ge(10)
    not_v = ~v  # Lt(10)
    not_not_v = ~not_v  # Ge(10)

    # Should be equivalent to Ge(10)
    assert isinstance(not_not_v, Ge)
    assert not_not_v.value == 10

  def test_double_negation_positive(self):
    """Test ~~Positive() effectively becomes Positive."""
    v = Positive()
    not_v = ~v  # NonPositive
    not_not_v = ~not_v  # Positive

    assert isinstance(not_not_v, Positive)

  def test_double_negation_validation(self):
    """Test double negation validates correctly."""
    v = ~~Ge(10)
    ctx = _make_context()

    data = pd.Series([5, 15])
    res = v.validate(data, ctx)

    # ~~Ge(10) should only pass. >= 10
    assert not res.success  # 5 fails
    assert res.mask is not None
    np.testing.assert_array_equal(res.mask.values, [False, True])


def _make_context() -> ValidationContext:
  """Create a minimal validation context for testing."""
  return ValidationContext(root_data=None)
