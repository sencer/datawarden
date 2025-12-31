"""Tests for smart negation (negate() methods) in validators."""

import pandas as pd
import pytest

from datawarden import (
  Between,
  Ge,
  Gt,
  Le,
  Lt,
  Negative,
  NonNegative,
  NonPositive,
  Not,
  Positive,
  Validated,
  validate,
)
from datawarden.validators.value import Outside


class TestSmartNegation:
  """Tests for smart negation via negate() methods."""

  def test_not_positive_uses_smart_negation(self):
    """Test that Not(Positive) uses NonPositive via smart negation."""
    v = Not(Positive())
    assert v._using_smart_negation
    assert isinstance(v.wrapped, NonPositive)

  def test_not_negative_uses_smart_negation(self):
    """Test that Not(Negative) uses NonNegative via smart negation."""
    v = Not(Negative())
    assert v._using_smart_negation
    assert isinstance(v.wrapped, NonNegative)

  def test_not_nonnegative_uses_smart_negation(self):
    """Test that Not(NonNegative) uses Negative via smart negation."""
    v = Not(NonNegative())
    assert v._using_smart_negation
    assert isinstance(v.wrapped, Negative)

  def test_not_nonpositive_uses_smart_negation(self):
    """Test that Not(NonPositive) uses Positive via smart negation."""
    v = Not(NonPositive())
    assert v._using_smart_negation
    assert isinstance(v.wrapped, Positive)


class TestComparisonNegation:
  """Tests for comparison operator negation."""

  def test_ge_negates_to_lt(self):
    """Test Ge(5).negate() returns Lt(5)."""
    v = Ge(5)
    neg = v.negate()
    assert isinstance(neg, Lt)
    assert neg.targets == (5,)

  def test_lt_negates_to_ge(self):
    """Test Lt(5).negate() returns Ge(5)."""
    v = Lt(5)
    neg = v.negate()
    assert isinstance(neg, Ge)
    assert neg.targets == (5,)

  def test_le_negates_to_gt(self):
    """Test Le(5).negate() returns Gt(5)."""
    v = Le(5)
    neg = v.negate()
    assert isinstance(neg, Gt)
    assert neg.targets == (5,)

  def test_gt_negates_to_le(self):
    """Test Gt(5).negate() returns Le(5)."""
    v = Gt(5)
    neg = v.negate()
    assert isinstance(neg, Le)
    assert neg.targets == (5,)

  def test_not_ge_uses_smart_negation(self):
    """Test Not(Ge(5)) uses Lt(5) via smart negation."""
    v = Not(Ge(5))
    assert v._using_smart_negation
    assert isinstance(v.wrapped, Lt)

    # Validate behavior
    data = pd.Series([3, 5, 7])  # 3 is valid (< 5), 5 and 7 are not valid
    with pytest.raises(ValueError, match=r"< 5"):
      v.validate(data)

  def test_not_lt_uses_smart_negation(self):
    """Test Not(Lt(5)) uses Ge(5) via smart negation."""
    v = Not(Lt(5))
    assert v._using_smart_negation
    assert isinstance(v.wrapped, Ge)

    # Validate behavior: values must be >= 5
    data = pd.Series([3, 5, 7])  # 3 is not valid
    with pytest.raises(ValueError, match=r">= 5"):
      v.validate(data)


class TestBetweenOutsideNegation:
  """Tests for Between/Outside negation."""

  def test_between_negates_to_outside(self):
    """Test Between(5, 10).negate() returns Outside(5, 10)."""
    v = Between(5, 10)
    neg = v.negate()
    assert isinstance(neg, Outside)
    assert neg.lower == 5
    assert neg.upper == 10

  def test_outside_negates_to_between(self):
    """Test Outside(5, 10).negate() returns Between(5, 10)."""
    v = Outside(5, 10)
    neg = v.negate()
    assert isinstance(neg, Between)
    assert neg.lower == 5
    assert neg.upper == 10

  def test_not_between_uses_smart_negation(self):
    """Test Not(Between(5, 10)) uses Outside via smart negation."""
    v = Not(Between(5, 10))
    assert v._using_smart_negation
    assert isinstance(v.wrapped, Outside)

    # Validate behavior: values must be < 5 OR > 10
    valid = pd.Series([4.0, 11.0])
    v.validate(valid)  # Should pass

    invalid = pd.Series([4.0, 5.0, 11.0])  # 5 is in [5, 10]
    with pytest.raises(ValueError, match=r"outside \[5, 10\]"):
      v.validate(invalid)

  def test_not_outside_uses_smart_negation(self):
    """Test Not(Outside(5, 10)) uses Between via smart negation."""
    v = Not(Outside(5, 10))
    assert v._using_smart_negation
    assert isinstance(v.wrapped, Between)

    # Validate behavior: values must be in [5, 10]
    valid = pd.Series([5.0, 7.0, 10.0])
    v.validate(valid)  # Should pass

    invalid = pd.Series([4.0, 7.0])  # 4 is outside
    with pytest.raises(ValueError, match=r">= 5"):
      v.validate(invalid)


class TestNegationPreservesIgnoreNan:
  """Tests that ignore_nan is preserved through negation."""

  def test_positive_negate_preserves_ignore_nan(self):
    """Test Positive(ignore_nan=True).negate() preserves the flag."""
    v = Positive(ignore_nan=True)
    neg = v.negate()
    assert neg.ignore_nan is True

  def test_ge_negate_preserves_ignore_nan(self):
    """Test Ge(5, ignore_nan=True).negate() preserves the flag."""
    v = Ge(5, ignore_nan=True)
    neg = v.negate()
    assert neg.ignore_nan is True

  def test_between_negate_preserves_ignore_nan(self):
    """Test Between(5, 10, ignore_nan=True).negate() preserves the flag."""
    v = Between(5, 10, ignore_nan=True)
    neg = v.negate()
    assert neg.ignore_nan is True


class TestNegationIntegration:
  """Integration tests for smart negation with the @validate decorator."""

  def test_decorator_with_not_positive(self):
    """Test @validate with Not(Positive) using smart negation."""

    @validate
    def func(data: Validated[pd.Series, Not(Positive)]):
      return data.sum()

    # Valid: non-positive values
    assert func(pd.Series([0, -1, -5])) == -6

    # Invalid: positive value
    with pytest.raises(ValueError, match="non-positive"):
      func(pd.Series([0, 1, -1]))

  def test_decorator_with_not_ge(self):
    """Test @validate with Not(Ge(0)) using smart negation."""

    @validate
    def func(data: Validated[pd.Series, Not(Ge(0))]):
      return data.sum()

    # Valid: negative values only
    assert func(pd.Series([-1, -2, -3])) == -6

    with pytest.raises(ValueError, match="< 0"):
      func(pd.Series([-1, 0, -2]))

  def test_decorator_with_not_between(self):
    """Test @validate with Not(Between(0, 100)) using smart negation."""

    @validate
    def func(data: Validated[pd.Series, Not(Between(0, 100))]):
      return data.sum()

    # Valid: values outside [0, 100]
    assert func(pd.Series([-5, 150])) == 145

    with pytest.raises(ValueError, match=r"outside \[0, 100\]"):
      func(pd.Series([-5, 50, 150]))
