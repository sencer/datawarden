"""Tests for logic edge cases: And/Or flattening, tautology/contradiction detection."""

import numpy as np
import pandas as pd

from datawarden.context import ValidationContext
from datawarden.validators import Ge, Le, Lt, Positive
from datawarden.validators.base import And, Fail, Not, Or, Pass


class TestAndFlattening:
  """Tests for And validator optimization."""

  def test_and_flattening(self):
    """Test that nested And validators are flattened."""
    v = And(And(Ge(0), Le(10)), Ge(5))
    assert len(v.validators) == 3
    assert all(isinstance(x, (Ge, Le)) for x in v.validators)

  def test_and_preserves_order(self):
    """Test that And flattening preserves validator order."""
    a, b, c = Ge(0), Le(10), Positive
    v = And(And(a, b), c)
    assert len(v.validators) == 3


class TestOrFlattening:
  """Tests for Or validator optimization."""

  def test_or_flattening(self):
    """Test that nested Or validators are flattened."""
    v = Or(Or(Ge(0), Le(-10)), Lt(0))
    assert len(v.validators) == 3

  def test_or_preserves_order(self):
    """Test that Or flattening preserves validator order."""
    a, b, c = Ge(0), Le(-10), Lt(0)
    v = Or(Or(a, b), c)
    assert len(v.validators) == 3


class TestTautologyDetection:
  """Tests for Or(A, ~A) -> Pass tautology detection."""

  def test_or_tautology_detection(self):
    """Test that Or(A, ~A) is detected as tautology."""
    v = Lt(10)
    v_neg = ~v  # Ge(10)
    or_v = Or(v, v_neg)

    # transform() should detect tautology and return [Pass()]
    transformed = or_v.transform()
    assert len(transformed) == 1
    assert isinstance(transformed[0], Pass)

  def test_or_with_pass_always_passes(self):
    """Test that Or with Pass validator always succeeds."""
    v = Or(Pass(), Ge(100))
    ctx = _make_context()

    # Should pass because Pass() always passes
    res = v.validate(pd.Series([1, 2, 3]), ctx)
    assert res.success


class TestContradictionDetection:
  """Tests for And(A, ~A) -> Fail contradiction detection."""

  def test_and_contradiction_detection(self):
    """Test that And(A, ~A) is detected as contradiction."""
    v = Ge(10)
    v_neg = ~v  # Lt(10)
    and_v = And(v, v_neg)

    # transform() should detect contradiction and return [Fail()]
    transformed = and_v.transform()
    assert len(transformed) == 1
    assert isinstance(transformed[0], Fail)


class TestOperatorInjection:
  """Tests for __or__, __and__, __invert__ operators."""

  def test_or_operator(self):
    """Test | operator creates Or validator."""
    v = Ge(10) | Le(0)
    assert isinstance(v, Or)

  def test_and_operator(self):
    """Test & operator creates And validator."""
    v = Ge(0) & Le(20)
    assert isinstance(v, And)

  def test_invert_operator(self):
    """Test ~ operator uses negate()."""
    v = ~Ge(10)
    # Smart negation should return Lt(10), not Not(Ge(10))
    assert isinstance(v, Lt)

  def test_invert_on_non_negatable(self):
    """Test ~ on validator without smart negate falls back to Not."""
    v = Pass()
    neg = ~v
    assert isinstance(neg, Not)


class TestDoubleNegation:
  """Tests for double negation optimization."""

  def test_double_negation_smart_path(self):
    """Test ~~Ge(10) becomes Ge(10) through smart negation."""
    v = Ge(10)
    not_v = ~v  # Lt(10)
    not_not_v = ~not_v  # Ge(10)

    assert isinstance(not_not_v, Ge)
    assert not_not_v.value == 10

  def test_double_negation_validation(self):
    """Test double negation validates correctly."""
    v = Ge(10)
    not_v = ~v
    not_not_v = ~not_v
    ctx = _make_context()

    data = pd.Series([5, 15])
    res = not_not_v.validate(data, ctx)

    assert not res.success
    np.testing.assert_array_equal(res.mask.values, [False, True])


class TestNotValidation:
  """Tests for Not validator behavior."""

  def test_not_inverts_result(self):
    """Test Not inverts validation result."""
    v = Not(Ge(0))
    ctx = _make_context()

    # Ge(0) passes for positive, so Not(Ge(0)) fails for positive
    res = v.validate(pd.Series([-1, -2, -3]), ctx)
    assert res.success

    res = v.validate(pd.Series([1, 2, 3]), ctx)
    assert not res.success

  def test_not_pass_always_fails(self):
    """Test Not(Pass) always fails."""
    v = Not(Pass())
    ctx = _make_context()

    res = v.validate(pd.Series([1, 2, 3]), ctx)
    assert not res.success

  def test_not_fail_always_succeeds(self):
    """Test Not(Fail) always succeeds."""
    v = Not(Fail())
    ctx = _make_context()

    res = v.validate(pd.Series([1, 2, 3]), ctx)
    assert res.success


def _make_context() -> ValidationContext:
  """Create a minimal validation context for testing."""
  return ValidationContext(root_data=None)
