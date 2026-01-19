from hypothesis import given, strategies as st
import pandas as pd
import pytest

from datawarden import And, Gt, Lt, Not, Or, Validated, validate
from datawarden.exceptions import ValidationError


@given(st.floats(allow_nan=False, allow_infinity=False))
def test_gt_property(val: float) -> None:
  threshold = 0.0

  @validate
  def func(_x: Validated[pd.Series, Gt(threshold)]) -> bool:
    return True

  s = pd.Series([val])
  if val > threshold:
    assert func(s) is True
  else:
    with pytest.raises(ValidationError):
      func(s)


@given(st.floats(allow_nan=False, allow_infinity=False))
def test_lt_property(val: float) -> None:
  threshold = 100.0

  @validate
  def func(_x: Validated[pd.Series, Lt(threshold)]) -> bool:
    return True

  s = pd.Series([val])
  if val < threshold:
    assert func(s) is True
  else:
    with pytest.raises(ValidationError):
      func(s)


@given(st.lists(st.floats(allow_nan=False, allow_infinity=False), min_size=1))
def test_and_logic_property(vals: list[float]) -> None:
  # Verifies logic: And(Gt(0), Lt(10))
  # Should pass if 0 < x < 10

  @validate
  def func(_df: Validated[pd.DataFrame, And(Gt(0), Lt(10))]) -> bool:
    return True

  df = pd.DataFrame({"a": vals})

  # Check manually
  mask = (df["a"] > 0) & (df["a"] < 10)
  expected_success = mask.all()

  if expected_success:
    assert func(df) is True
  else:
    with pytest.raises(ValidationError):
      func(df)


@given(st.lists(st.floats(allow_nan=False, allow_infinity=False), min_size=1))
def test_or_logic_property(vals: list[float]) -> None:
  # Verifies logic: Or(Lt(0), Gt(10))
  # Should pass if x < 0 OR x > 10

  @validate
  def func(_df: Validated[pd.DataFrame, Or(Lt(0), Gt(10))]) -> bool:
    return True

  df = pd.DataFrame({"a": vals})

  # Check manually
  mask = (df["a"] < 0) | (df["a"] > 10)
  expected_success = mask.all()

  if expected_success:
    assert func(df) is True
  else:
    with pytest.raises(ValidationError):
      func(df)


@given(st.floats(allow_nan=False, allow_infinity=False))
def test_not_logic_property(val: float) -> None:
  # Logic: Not(Gt(0)) -> Le(0)

  @validate
  def func(_x: Validated[pd.Series, Not(Gt(0))]) -> bool:
    return True

  s = pd.Series([val])
  if not (val > 0):
    assert func(s) is True
  else:
    with pytest.raises(ValidationError):
      func(s)
