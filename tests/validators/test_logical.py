import pandas as pd

from datawarden.context import ValidationContext
from datawarden.validators.base import And, BaseValidator, Fail, Not, Or, Pass
from datawarden.validators.numeric import Gt, Lt

CTX = ValidationContext(root_data=None)


def test_and():
  # Vectorized
  s = pd.Series([1, 5, 10])
  v = And(Gt(2), Lt(8))
  res = v.validate(s, CTX)
  pd.testing.assert_series_equal(res.mask, pd.Series([False, True, False]))

  # Boolean
  v = And(Pass(), Pass())
  assert v.validate(None, CTX).success
  v = And(Pass(), Fail())
  assert not v.validate(None, CTX).success


def test_or():
  # Vectorized
  s = pd.Series([1, 5, 10])
  v = Or(Gt(8), Lt(2))
  res = v.validate(s, CTX)
  pd.testing.assert_series_equal(res.mask, pd.Series([True, False, True]))

  # Boolean
  v = Or(Pass(), Fail())
  assert v.validate(None, CTX).success
  v = Or(Fail(), Fail())
  assert not v.validate(None, CTX).success


def test_not():
  v = Not(Pass())
  assert not v.validate(None, CTX).success
  v = Not(Fail())
  assert v.validate(None, CTX).success

  # Vectorized
  s = pd.Series([1, 5, 10])
  # Not(Gt(5)) -> <= 5
  v = Not(Gt(5))
  res = v.validate(s, CTX)
  pd.testing.assert_series_equal(res.mask, pd.Series([True, True, False]))

  # Not(Gt(2) & Lt(8)) -> (<= 2 | >= 8)
  v = Not(Gt(2) & Lt(8))
  res = v.validate(s, CTX)
  pd.testing.assert_series_equal(res.mask, pd.Series([True, False, True]))


def test_operator_overloads():
  v = Gt(5) & Lt(10)
  assert isinstance(v, And)
  v = Gt(5) | Lt(10)
  assert isinstance(v, Or)
  v = ~Gt(5)
  assert isinstance(v, BaseValidator)
  assert "<=" in str(v)
