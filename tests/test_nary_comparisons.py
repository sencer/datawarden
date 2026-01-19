import pandas as pd
import pytest

from datawarden import Ge, Gt, Le, Lt, Validated, validate
from datawarden.exceptions import ValidationError


def test_ge_nary_success() -> None:
  @validate
  def func(df: Validated[pd.DataFrame, Ge("high", "close", "low")]) -> bool:
    del df
    return True

  # Ge("high", "close", "low") validates: high >= close >= low
  df = pd.DataFrame({"high": [20, 25], "close": [15, 20], "low": [10, 20]})
  assert func(df) is True


def test_ge_nary_failure() -> None:
  @validate
  def func(df: Validated[pd.DataFrame, Ge("high", "close", "low")]) -> bool:
    del df
    return True

  df = pd.DataFrame({
    "high": [30, 20],
    "close": [35, 20],  # 35 > 30, fails high >= close
    "low": [10, 20],
  })
  with pytest.raises(ValidationError) as excinfo:
    func(df)
  assert "Ge('high', 'close', 'low')" in str(excinfo.value)


def test_gt_nary() -> None:
  @validate
  def func(df: Validated[pd.DataFrame, Gt("b", "a")]) -> bool:
    del df
    return True

  # Gt("b", "a") validates: b > a
  assert func(pd.DataFrame({"a": [1], "b": [2]})) is True
  with pytest.raises(ValidationError):
    func(pd.DataFrame({"a": [2], "b": [1]}))  # 1 < 2, fails b > a


def test_le_nary() -> None:
  @validate
  def func(df: Validated[pd.DataFrame, Le("c", "b", "a")]) -> bool:
    del df
    return True

  # Le("c", "b", "a") validates: c <= b <= a
  assert func(pd.DataFrame({"a": [10], "b": [5], "c": [5]})) is True
  with pytest.raises(ValidationError):
    func(pd.DataFrame({"a": [10], "b": [11], "c": [5]}))  # 11 > 10, fails b <= a


def test_lt_nary() -> None:
  @validate
  def func(df: Validated[pd.DataFrame, Lt("b", "a")]) -> bool:
    del df
    return True

  # Lt("b", "a") validates: b < a
  assert func(pd.DataFrame({"a": [10], "b": [9]})) is True
  with pytest.raises(ValidationError):
    func(pd.DataFrame({"a": [10], "b": [11]}))  # 11 > 10, fails b < a


def test_nary_missing_column() -> None:
  @validate
  def func(df: Validated[pd.DataFrame, Ge("a", "b")]) -> bool:
    del df
    return True

  with pytest.raises(ValidationError) as excinfo:
    func(pd.DataFrame({"c": [1]}))
  assert "Missing columns" in str(excinfo.value)


def test_unary_still_works() -> None:
  @validate
  def func(s: Validated[pd.Series, Ge(0)]) -> bool:
    del s
    return True

  assert func(pd.Series([0, 1])) is True
  with pytest.raises(ValidationError):
    func(pd.Series([-1]))
