"""Tests for comparison validators: Ge, Le, Gt, Lt."""
# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false
# pyright: reportCallIssue=false, reportAttributeAccessIssue=false

import pandas as pd
import pytest

from datawarden import Ge, Gt, Le, Lt


class TestGe:
  """Tests for Ge (column comparison) validator."""

  def test_valid_comparison(self):
    """Test Ge validator with valid column comparison."""
    data = pd.DataFrame({"high": [10, 20, 30], "low": [5, 10, 15]})
    validator = Ge("high", "low")
    validator = Ge("high", "low")
    assert validator.validate(data) is None

  def test_equal_values_allowed(self):
    """Test Ge validator allows equal values."""
    data = pd.DataFrame({"high": [10, 10, 10], "low": [10, 10, 10]})
    validator = Ge("high", "low")
    validator = Ge("high", "low")
    assert validator.validate(data) is None

  def test_invalid_comparison(self):
    """Test Ge validator rejects invalid comparison."""
    data = pd.DataFrame({"high": [10, 5, 30], "low": [5, 10, 15]})
    validator = Ge("high", "low")
    with pytest.raises(ValueError, match="high must be >= low"):
      validator.validate(data)

  def test_missing_columns(self):
    """Test Ge validator with missing columns raises error."""
    data = pd.DataFrame({"high": [10, 20]})
    validator = Ge("high", "low")
    # Should raise ValueError if column is missing
    with pytest.raises(ValueError, match="Missing columns for comparison"):
      validator.validate(data)

  def test_non_dataframe(self):
    """Test Ge validator with non-DataFrame."""
    data = pd.Series([1, 2, 3])
    validator = Ge("a", "b")
    with pytest.raises(TypeError, match="requires a pandas DataFrame"):
      validator.validate(data)

  def test_ge_with_numeric_column_names_fails(self):
    """Test Ge validator with non-string column names raises TypeError."""
    data = pd.DataFrame({"high": [10, 20, 30], "low": [5, 10, 15]})
    validator = Ge(1, 2)  # Non-string column names
    with pytest.raises(
      TypeError, match="Column comparison requires string column names"
    ):
      validator.validate(data)

  def test_unary_comparison(self):
    """Test Ge validator with unary comparison."""
    data = pd.Series([5, 6, 7])
    validator = Ge(5)
    validator = Ge(5)
    assert validator.validate(data) is None

  def test_unary_fails(self):
    """Test Ge validator fails with unary comparison."""
    data = pd.Series([4, 5, 6])
    validator = Ge(5)
    with pytest.raises(ValueError, match="Data must be >= 5"):
      validator.validate(data)


class TestLe:
  """Tests for Le (<=) validator."""

  def test_valid_comparison(self):
    """Test Le validator with valid column comparison."""
    data = pd.DataFrame({"low": [5, 10, 15], "high": [10, 20, 30]})
    validator = Le("low", "high")
    validator = Le("low", "high")
    assert validator.validate(data) is None

  def test_equal_values_allowed(self):
    """Test Le validator allows equal values."""
    data = pd.DataFrame({"low": [10, 10, 10], "high": [10, 10, 10]})
    validator = Le("low", "high")
    validator = Le("low", "high")
    assert validator.validate(data) is None

  def test_invalid_comparison(self):
    """Test Le validator rejects invalid comparison."""
    data = pd.DataFrame({"low": [10, 25, 15], "high": [10, 20, 30]})
    validator = Le("low", "high")
    with pytest.raises(ValueError, match="low must be <= high"):
      validator.validate(data)

  def test_le_with_numeric_column_names_fails(self):
    """Test Le validator with non-string column names raises TypeError."""
    data = pd.DataFrame({"high": [10, 20, 30], "low": [5, 10, 15]})
    validator = Le(1, 2)
    with pytest.raises(
      TypeError, match="Column comparison requires string column names"
    ):
      validator.validate(data)

  def test_unary_with_series(self):
    """Test Le validator with unary comparison on Series."""
    data = pd.Series([1, 2, 3])
    validator = Le(5)
    validator = Le(5)
    assert validator.validate(data) is None

  def test_unary_fails(self):
    """Test Le validator fails with unary comparison."""
    data = pd.Series([1, 2, 6])
    validator = Le(5)
    with pytest.raises(ValueError, match="Data must be <= 5"):
      validator.validate(data)


class TestGt:
  """Tests for Gt (>) validator."""

  def test_valid_comparison(self):
    """Test Gt validator with valid column comparison."""
    data = pd.DataFrame({"high": [20, 30, 40], "low": [10, 20, 30]})
    validator = Gt("high", "low")
    validator = Gt("high", "low")
    assert validator.validate(data) is None

  def test_equal_values_rejected(self):
    """Test Gt validator rejects equal values."""
    data = pd.DataFrame({"high": [10, 10, 10], "low": [10, 10, 10]})
    validator = Gt("high", "low")
    with pytest.raises(ValueError, match="high must be > low"):
      validator.validate(data)

  def test_invalid_comparison(self):
    """Test Gt validator rejects invalid comparison."""
    data = pd.DataFrame({"high": [10, 15, 30], "low": [10, 20, 15]})
    validator = Gt("high", "low")
    with pytest.raises(ValueError, match="high must be > low"):
      validator.validate(data)

  def test_gt_with_numeric_column_names_fails(self):
    """Test Gt validator with non-string column names raises TypeError."""
    data = pd.DataFrame({"high": [10, 20, 30], "low": [5, 10, 15]})
    validator = Gt(1, 2)
    with pytest.raises(
      TypeError, match="Column comparison requires string column names"
    ):
      validator.validate(data)

  def test_unary_with_series(self):
    """Test Gt validator with unary comparison on Series."""
    data = pd.Series([2, 3, 4])
    validator = Gt(1)
    validator = Gt(1)
    assert validator.validate(data) is None

  def test_unary_fails(self):
    """Test Gt validator fails with unary comparison."""
    data = pd.Series([1, 2, 3])
    validator = Gt(1)
    with pytest.raises(ValueError, match="Data must be > 1"):
      validator.validate(data)


class TestLt:
  """Tests for Lt (<) validator."""

  def test_valid_comparison(self):
    """Test Lt validator with valid column comparison."""
    data = pd.DataFrame({"low": [10, 20, 30], "high": [20, 30, 40]})
    validator = Lt("low", "high")
    validator = Lt("low", "high")
    assert validator.validate(data) is None

  def test_equal_values_rejected(self):
    """Test Lt validator rejects equal values."""
    data = pd.DataFrame({"low": [10, 10, 10], "high": [10, 10, 10]})
    validator = Lt("low", "high")
    with pytest.raises(ValueError, match="low must be < high"):
      validator.validate(data)

  def test_invalid_comparison(self):
    """Test Lt validator rejects invalid comparison."""
    data = pd.DataFrame({"low": [15, 20, 30], "high": [10, 30, 25]})
    validator = Lt("low", "high")
    with pytest.raises(ValueError, match="low must be < high"):
      validator.validate(data)

  def test_lt_with_numeric_column_names_fails(self):
    """Test Lt validator with non-string column names raises TypeError."""
    data = pd.DataFrame({"high": [10, 20, 30], "low": [5, 10, 15]})
    validator = Lt(1, 2)
    with pytest.raises(
      TypeError, match="Column comparison requires string column names"
    ):
      validator.validate(data)

  def test_unary_with_series(self):
    """Test Lt validator with unary comparison on Series."""
    data = pd.Series([1, 2, 3])
    validator = Lt(5)
    validator = Lt(5)
    assert validator.validate(data) is None

  def test_unary_fails(self):
    """Test Lt validator fails with unary comparison."""
    data = pd.Series([1, 2, 5])
    validator = Lt(5)
    with pytest.raises(ValueError, match="Data must be < 5"):
      validator.validate(data)


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

  df_invalid = pd.DataFrame({
    "a": [10],
    "b": [5],
    "c": [6],
  })
  mask_inv = v.validate_vectorized(df_invalid)
  assert not mask_inv.all()
  # first row is valid (10>=5, 5>=6 False) -> False
  assert not mask_inv[0]

  # Test error path in validate_vectorized
  with pytest.raises(TypeError):
    v.validate_vectorized(pd.Series([1]))


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
