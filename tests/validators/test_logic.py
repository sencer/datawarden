import pandas as pd
import pytest

from datawarden import Between, Is, Not, Validated, validate


class TestNotValidator:
  """Tests for the Not logical validator."""

  def test_not_between_series(self):
    """Test Not(Between) on Series."""

    @validate
    def func(data: Validated[pd.Series, Not(Between(5, 10))]):
      return data

    # Valid: < 5 or > 10
    valid = pd.Series([4.0, 11.0, 0.0, 100.0])
    assert func(valid).equals(valid)

    # Invalid: 5 is in [5, 10] range
    invalid = pd.Series([4.0, 5.0, 11.0])
    with pytest.raises(ValueError, match="must be outside"):
      func(invalid)

  def test_not_custom_is(self):
    """Test Not(Is(...)) logic."""

    # Is(x > 0) -> Not -> x <= 0
    @validate
    def func(data: Validated[pd.Series, Not(Is(lambda x: x > 0, name="positive"))]):
      return data

    valid = pd.Series([0, -1, -5])
    assert func(valid).equals(valid)

    invalid = pd.Series([-1, 1])
    with pytest.raises(ValueError, match="must be non-positive"):
      func(invalid)

  def test_not_dataframe(self):
    """Test Not on DataFrame (vectorized)."""

    @validate
    def func(df: Validated[pd.DataFrame, Not(Between(0, 1))]):
      return df

    # Values must NOT be in [0, 1]
    valid_df = pd.DataFrame({"a": [2, 3], "b": [-1, 5]})
    assert func(valid_df).equals(valid_df)

    invalid_df = pd.DataFrame({"a": [2, 0.5], "b": [-1, 5]})
    with pytest.raises(ValueError, match="must be outside"):
      func(invalid_df)

  def test_not_non_vectorized_raises(self):
    """Test that Not falls back to sequential validation for non-vectorized validators."""

    class DummyValidator:
      def validate(self, data):
        pass

      def describe(self):
        return "dummy"

    # Should raise ValueError because DummyValidator passes, so Not() fails
    v = Not(DummyValidator())
    s = pd.Series([1, 2])
    with pytest.raises(ValueError, match="Data must not dummy"):
      v.validate(s)
