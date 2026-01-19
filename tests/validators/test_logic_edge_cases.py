"""Tests for logic edge cases including tautology detection."""

import numpy as np
import pandas as pd

from datawarden.context import ValidationContext
from datawarden.validators import Ge, IsNaN, Le, Lt, Negative, Positive
from datawarden.validators.base import And, Not, Or, Pass


def test_logic_double_negation():
  """Test double negation optimization."""
  v = Ge(10)
  not_v = ~v
  not_not_v = ~not_v

  # ~Ge(10) -> Lt(10). ~Lt(10) -> Ge(10)
  assert ">=10" in str(not_not_v)

  data = pd.Series([5, 15])
  ctx = ValidationContext(root_data=data)

  # Test that double negation works correctly
  result = not_not_v.validate(data, ctx)
  assert not result.success  # 5 is not >= 10

  data2 = pd.Series([15, 20])
  ctx2 = ValidationContext(root_data=data2)
  result2 = not_not_v.validate(data2, ctx2)
  assert result2.success  # Both >= 10


def test_not_describe_special_cases():
  """Test string representations of Not validators."""
  assert str(Not(IsNaN)) == "~IsNaN"
  assert str(Not(Positive)) == "~Positive"
  assert str(Not(Negative)) == "~Negative"


def test_and_flattening():
  """Test that nested And validators ARE flattened (implemented)."""
  v = And(And(Ge(0), Le(10)), Ge(5))
  # Now flattens for better debugging (<3% performance cost)
  assert len(v.validators) == 3  # Flattened: Ge(0), Le(10), Ge(5)
  assert all(isinstance(x, (Ge, Le)) for x in v.validators)


def test_or_tautology_detection():
  """Test tautology detection: Or(V, ~V) -> Pass."""
  v3 = Lt(10)
  v4 = ~v3  # Ge(10)
  v = Or(v3, v4)
  transformed = v.transform()[0]
  assert isinstance(transformed, Pass)


def test_operator_injection():
  """Test that | & ~ operators work correctly."""
  v = Ge(10)
  assert isinstance(v | Le(0), Or)
  assert isinstance(v & Le(20), And)
  # ~v should be Lt(10)
  assert isinstance(~v, Lt)


def test_and_empty():
  """Test empty And validator raises error (no validators provided)."""
  v = And()
  data = pd.Series([1, 2])
  ctx = ValidationContext(root_data=data)
  # Empty And should raise IndexError - this is intentional
  # Users should not create empty And validators
  try:
    result = v.validate(data, ctx)
    # If it doesn't raise, it should at least fail validation
    assert not result.success, "Empty And should fail"
  except IndexError:
    # This is expected behavior
    pass


def test_or_with_pass():
  """Test Or with Pass always succeeds."""
  v = Or(Pass(), Ge(0))
  data = pd.Series([1, 2, 3, 4, 5])
  ctx = ValidationContext(root_data=data)
  result = v.validate(data, ctx)
  assert result.success

  # Even negative values should pass because of Pass
  data2 = pd.Series([-1, -2])
  ctx2 = ValidationContext(root_data=data2)
  result2 = v.validate(data2, ctx2)
  assert result2.success


def test_not_isnan():
  """Test Not(IsNaN) correctly identifies non-NaN values."""
  v = Not(IsNaN)
  data = pd.Series([1, np.nan])
  ctx = ValidationContext(root_data=data)
  result = v.validate(data, ctx)
  # Should fail because there's a NaN
  assert not result.success

  data2 = pd.Series([1, 2, 3])
  ctx2 = ValidationContext(root_data=data2)
  result2 = v.validate(data2, ctx2)
  # Should pass - no NaNs
  assert result2.success
