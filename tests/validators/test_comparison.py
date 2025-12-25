"""Tests for comparison validators: Ge, Le, Gt, Lt."""
# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false
# pyright: reportCallIssue=false, reportAttributeAccessIssue=false

import pandas as pd
import pytest

from validated import Ge, Gt, Le, Lt


class TestGe:
  """Tests for Ge (column comparison) validator."""

  def test_valid_comparison(self):
    """Test Ge validator with valid column comparison."""
    data = pd.DataFrame({"high": [10, 20, 30], "low": [5, 10, 15]})
    validator = Ge("high", "low")
    result = validator.validate(data)
    assert result.equals(data)

  def test_equal_values_allowed(self):
    """Test Ge validator allows equal values."""
    data = pd.DataFrame({"high": [10, 10, 10], "low": [10, 10, 10]})
    validator = Ge("high", "low")
    result = validator.validate(data)
    assert result.equals(data)

  def test_invalid_comparison(self):
    """Test Ge validator rejects invalid comparison."""
    data = pd.DataFrame({"high": [10, 5, 30], "low": [5, 10, 15]})
    validator = Ge("high", "low")
    with pytest.raises(ValueError, match="high must be >= low"):
      validator.validate(data)

  def test_missing_columns(self):
    """Test Ge validator with missing columns."""
    data = pd.DataFrame({"high": [10, 20]})
    validator = Ge("high", "low")
    # Should not raise if column is missing
    result = validator.validate(data)
    assert result.equals(data)

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
    result = validator.validate(data)
    assert result.equals(data)

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
    result = validator.validate(data)
    assert result.equals(data)

  def test_equal_values_allowed(self):
    """Test Le validator allows equal values."""
    data = pd.DataFrame({"low": [10, 10, 10], "high": [10, 10, 10]})
    validator = Le("low", "high")
    result = validator.validate(data)
    assert result.equals(data)

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
    result = validator.validate(data)
    assert result.equals(data)

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
    result = validator.validate(data)
    assert result.equals(data)

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
    result = validator.validate(data)
    assert result.equals(data)

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
    result = validator.validate(data)
    assert result.equals(data)

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
    result = validator.validate(data)
    assert result.equals(data)

  def test_unary_fails(self):
    """Test Lt validator fails with unary comparison."""
    data = pd.Series([1, 2, 5])
    validator = Lt(5)
    with pytest.raises(ValueError, match="Data must be < 5"):
      validator.validate(data)
