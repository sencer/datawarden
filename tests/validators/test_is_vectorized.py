"""Tests for vectorized Is validator on DataFrames."""

import pandas as pd
import pytest

from datawarden import Is, Validated, validate


def test_is_vectorized_dataframe():
  """Test that Is validator can handle whole DataFrame vectorized checks."""
  df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6], "c": [5, 7, 9]})

  # Check a + b == c
  validator = Is(lambda df: df["a"] + df["b"] == df["c"])
  assert validator.validate(df) is None

  # Invalid data
  bad_df = pd.DataFrame({
    "a": [1, 2, 3],
    "b": [4, 5, 6],
    "c": [5, 7, 10],  # 3+6 != 10
  })
  with pytest.raises(ValueError, match="1 values failed"):
    validator.validate(bad_df)


@validate
def process_data(df: Validated[pd.DataFrame, Is(lambda df: df.sum(axis=1) < 20)]):
  return df


def test_is_decorator_vectorized():
  """Test Is validator within @validate decorator for DataFrames."""
  df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})  # sums: 4, 6 < 20
  assert process_data(df) is df

  bad_df = pd.DataFrame({"a": [10, 20], "b": [10, 20]})  # sums: 20, 40 NOT < 20
  with pytest.raises(ValueError, match="2 values failed"):
    process_data(bad_df)
