"""Tests for De Morgan's laws in validator logic.

Verifies that:
- Not(A | B) behaves equivalently to And(Not(A), Not(B))
- Not(A & B) behaves equivalently to Or(Not(A), Not(B))

Note: Smart negation is used where possible, so Not(Positive | IsNaN)
may not literally become And(Not(Positive), Not(IsNaN)), but the validation
result should be equivalent.
"""

import numpy as np
import pandas as pd

from datawarden.context import ValidationContext
from datawarden.validators import Ge, IsNaN, Positive
from datawarden.validators.base import And, Not, Or


def _make_context() -> ValidationContext:
  """Create a minimal validation context for testing."""
  return ValidationContext(root_data=None)


class TestDeMorganLaws:
  """Tests for De Morgan's law behavior in Not(Or(...)) and Not(And(...))."""

  def test_not_or_equivalent_to_and_not(self):
    """Test Not(A | B) is equivalent to And(Not(A), Not(B))."""
    # Not(Positive | IsNaN) = Not(Positive) & Not(IsNaN) = NonPositive & NotNaN
    # = values that are <= 0 AND not NaN = strictly non-positive finite numbers
    v = Not(Or(Positive, IsNaN))
    ctx = _make_context()

    data = pd.Series([1, -1, np.nan])
    # Positive | IsNaN -> [T, F, T]
    # Not(...) -> [F, T, F]

    res = v.validate(data, ctx)
    assert not res.success  # Has failures
    assert res.mask is not None
    np.testing.assert_array_equal(res.mask.values, [False, True, False])

  def test_not_and_equivalent_to_or_not(self):
    """Test Not(A & B) is equivalent to Or(Not(A), Not(B))."""
    # Not(Ge(0) & Positive) = Not(Ge(0)) | Not(Positive)
    # = Lt(0) | NonPositive = values < 0 OR values <= 0 = values <= 0
    v = Not(And(Ge(0), Positive))
    ctx = _make_context()

    data = pd.Series([-1, 0, 1])
    # Ge(0) -> [F, T, T]
    # Positive -> [F, F, T]
    # Ge(0) & Positive -> [F, F, T]
    # Not(...) -> [T, T, F]

    res = v.validate(data, ctx)
    assert not res.success  # Has failures
    assert res.mask is not None
    np.testing.assert_array_equal(res.mask.values, [True, True, False])


class TestNotOrIsNaN:
  """Specific tests for Not(Positive | IsNaN) pattern."""

  def test_not_positive_or_isnan_only_allows_non_positive(self):
    """Test Not(Positive | IsNaN) only allows non-positive values."""
    v = Not(Or(Positive, IsNaN))
    ctx = _make_context()

    # Should pass for negative values
    res = v.validate(pd.Series([-1, -2, -3]), ctx)
    assert res.success

    # Should pass for zero
    res = v.validate(pd.Series([0]), ctx)
    assert res.success

    # Should fail for positive
    res = v.validate(pd.Series([1]), ctx)
    assert not res.success

    # Should fail for NaN
    res = v.validate(pd.Series([np.nan]), ctx)
    assert not res.success


class TestNestedNegation:
  """Tests for deeply nested Not operations."""

  def test_triple_negation(self):
    """Test ~~~Ge(0) = ~Ge(0) = Lt(0)."""
    v = ~~~Ge(0)
    ctx = _make_context()

    # Lt(0) should pass for negative, fail for non-negative
    res = v.validate(pd.Series([-1, -2]), ctx)
    assert res.success

    res = v.validate(pd.Series([0, 1]), ctx)
    assert not res.success

  def test_not_not_or(self):
    """Test ~~Or(A, B) behavior."""
    v = ~~Or(Positive, IsNaN)
    ctx = _make_context()

    # ~~Or() should behave like Or() for simple cases
    # (smart negation may optimize differently)
    res = v.validate(pd.Series([1, np.nan]), ctx)
    assert res.success

    res = v.validate(pd.Series([0]), ctx)
    assert not res.success  # 0 is neither positive nor NaN
