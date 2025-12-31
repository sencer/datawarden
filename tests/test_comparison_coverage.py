import pandas as pd
import pytest

from datawarden.validators import Ge, Gt, Le, Lt


def test_n_ary_comparison():
  df = pd.DataFrame({"a": [10], "b": [5], "c": [1]})

  # Valid
  Ge("a", "b", "c").validate(df)
  Gt("a", "b", "c").validate(df)

  # Invalid
  with pytest.raises(ValueError):
    Ge("c", "b").validate(df)  # 1 >= 5 False

  with pytest.raises(ValueError):
    Gt("a", "a").validate(df)  # 10 > 10 False


def test_n_ary_comparison_missing_columns():
  df = pd.DataFrame({"a": [10]})
  v = Ge("a", "b")
  with pytest.raises(ValueError, match="Missing columns"):
    v.validate(df)


def test_n_ary_comparison_type_error():
  v = Ge("a", "b")
  # Not a DataFrame
  with pytest.raises(TypeError):
    v.validate(pd.Series([1, 2]))

  # Invalid args (non-string)
  # The validator requires targets to be strings for N-ary (if length > 1)
  v_bad = Ge("a", 10)

  df = pd.DataFrame({"a": [10]})
  with pytest.raises(TypeError, match="requires string column names"):
    v_bad.validate(df)


def test_negation():
  assert isinstance(Ge(0).negate(), Lt)
  assert isinstance(Le(0).negate(), Gt)
  assert isinstance(Gt(0).negate(), Le)
  assert isinstance(Lt(0).negate(), Ge)


def test_comparison_scalar_failures():
  # We added scalar support
  Ge(0).validate(0)
  Ge(0).validate(1)
  with pytest.raises(ValueError):
    Ge(0).validate(-1)

  Lt(10).validate(9)
  with pytest.raises(ValueError):
    Lt(10).validate(10)
