import pandas as pd
import pytest

from datawarden.validators import Ge, Le


def test_n_ary_explicit_coverage():
  # This test is designed to hit the loop in comparison.py lines 150-175

  # 3 columns to ensure loop runs twice
  df = pd.DataFrame({"a": [10, 10, 10], "b": [5, 5, 5], "c": [1, 1, 1]})

  # a >= b >= c. Valid.
  v = Ge("a", "b", "c")
  v.validate(df)

  # a >= b >= c. Invalid at second step (b >= c)
  df_invalid = pd.DataFrame({
    "a": [10],
    "b": [5],
    "c": [6],  # 5 >= 6 is False
  })

  with pytest.raises(ValueError, match="b must be >= c"):
    v.validate(df_invalid)

  # Invalid at first step
  df_invalid_1 = pd.DataFrame({"a": [4], "b": [5], "c": [1]})
  with pytest.raises(ValueError, match="a must be >= b"):
    v.validate(df_invalid_1)


def test_n_ary_validate_vectorized():
  # Explicitly test the vectorized path
  df = pd.DataFrame({"a": [10, 10, 10], "b": [5, 5, 5], "c": [1, 1, 1]})
  v = Ge("a", "b", "c")
  mask = v.validate_vectorized(df)
  assert mask.all()

  df_invalid = pd.DataFrame({"a": [10], "b": [5], "c": [6]})
  mask_inv = v.validate_vectorized(df_invalid)
  assert not mask_inv.all()
  # first row is valid (10>=5, 5>=6 False) -> False
  assert not mask_inv[0]

  # Test error path in validate_vectorized
  with pytest.raises(TypeError):
    v.validate_vectorized(pd.Series([1]))


def test_n_ary_le():
  v = Le("c", "b", "a")
  df = pd.DataFrame({"a": [10], "b": [5], "c": [1]})
  v.validate(df)
