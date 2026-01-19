"""Tests for logic validators: Not."""

import pandas as pd
import pytest

from datawarden import Validated, validate
from datawarden.exceptions import ValidationError
from datawarden.validators import Ge, Le
from datawarden.validators.base import And, Not


class TestNotValidator:
  """Tests for the Not logical validator."""

  def test_not_between_series(self):
    """Test Not(Ge & Le) on Series."""

    @validate
    def func(data: Validated[pd.Series, Not(And(Ge(5), Le(10)))]) -> pd.Series:
      return data

    # Valid: < 5 or > 10
    valid = pd.Series([4.0, 11.0, 0.0, 100.0])
    result = func(valid)
    assert result.equals(valid)

    # Invalid: 5 is in [5, 10] range
    invalid = pd.Series([4.0, 5.0, 11.0])
    with pytest.raises(ValidationError):
      func(invalid)

  def test_not_ge(self):
    """Test Not(Ge) -> Lt."""

    @validate
    def func(data: Validated[pd.Series, Not(Ge(0))]) -> pd.Series:
      return data

    # Valid: < 0
    valid = pd.Series([-1, -2, -3])
    result = func(valid)
    assert result.equals(valid)

    # Invalid: >= 0
    with pytest.raises(ValidationError):
      func(pd.Series([1, 2, 3]))

  def test_not_le(self):
    """Test Not(Le) -> Gt."""

    @validate
    def func(data: Validated[pd.Series, Not(Le(0))]) -> pd.Series:
      return data

    # Valid: > 0
    valid = pd.Series([1, 2, 3])
    result = func(valid)
    assert result.equals(valid)

    # Invalid: <= 0
    with pytest.raises(ValidationError):
      func(pd.Series([-1, 0, -2]))

  def test_not_dataframe(self):
    """Test Not on DataFrame (vectorized)."""

    @validate
    def func(df: Validated[pd.DataFrame, Not(And(Ge(0), Le(1)))]) -> pd.DataFrame:
      return df

    # Values must NOT be in [0, 1]
    valid_df = pd.DataFrame({"a": [2, 3], "b": [-1, 5]})
    result = func(valid_df)
    assert result.equals(valid_df)

    invalid_df = pd.DataFrame({"a": [2, 0.5], "b": [-1, 5]})
    with pytest.raises(ValidationError):
      func(invalid_df)

  def test_double_negation(self):
    """Test double negation optimization."""

    @validate
    def func(data: Validated[pd.Series, Not(Not(Ge(0)))]) -> pd.Series:
      return data

    # Not(Not(Ge(0))) = Ge(0)
    valid = pd.Series([0, 1, 2])
    result = func(valid)
    assert result.equals(valid)

    with pytest.raises(ValidationError):
      func(pd.Series([-1, -2]))
