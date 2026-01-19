"""Tests for Numba-accelerated multi-column comparison validators."""

import numpy as np
import pandas as pd
import pytest

from datawarden import Validated, validate
from datawarden.context import ValidationContext
from datawarden.exceptions import ValidationError
from datawarden.validators import Ge, Gt, Le, Lt


class TestNumbaColumnComparison:
  """Test Numba acceleration for Ge['a', 'b'] style validators."""

  def test_ge_two_columns_valid(self) -> None:
    """Test Ge['a', 'b'] passes when a >= b for all rows."""
    df = pd.DataFrame({"a": [10.0, 20.0, 30.0], "b": [5.0, 10.0, 15.0]})
    validator = Ge("a", "b")
    ctx = ValidationContext(root_data=df)
    result = validator.validate(df, ctx)
    assert result.success

  def test_ge_two_columns_equal_valid(self) -> None:
    """Test Ge['a', 'b'] passes when a == b for all rows."""
    df = pd.DataFrame({"a": [10.0, 20.0, 30.0], "b": [10.0, 20.0, 30.0]})
    validator = Ge("a", "b")
    ctx = ValidationContext(root_data=df)
    result = validator.validate(df, ctx)
    assert result.success

  def test_ge_two_columns_invalid(self) -> None:
    """Test Ge['a', 'b'] fails when any a < b."""
    df = pd.DataFrame({"a": [10.0, 5.0, 30.0], "b": [5.0, 10.0, 15.0]})
    validator = Ge("a", "b")
    ctx = ValidationContext(root_data=df)
    result = validator.validate(df, ctx)
    assert not result.success
    # Should have mask showing which rows failed
    assert result.mask is not None

  def test_gt_two_columns_valid(self) -> None:
    """Test Gt['a', 'b'] passes when a > b for all rows."""
    df = pd.DataFrame({"a": [10.0, 20.0, 30.0], "b": [5.0, 10.0, 15.0]})
    validator = Gt("a", "b")
    ctx = ValidationContext(root_data=df)
    result = validator.validate(df, ctx)
    assert result.success

  def test_gt_two_columns_equal_invalid(self) -> None:
    """Test Gt['a', 'b'] fails when a == b."""
    df = pd.DataFrame({"a": [10.0, 20.0, 30.0], "b": [10.0, 20.0, 30.0]})
    validator = Gt("a", "b")
    ctx = ValidationContext(root_data=df)
    result = validator.validate(df, ctx)
    assert not result.success

  def test_le_two_columns_valid(self) -> None:
    """Test Le['a', 'b'] passes when a <= b for all rows."""
    df = pd.DataFrame({"a": [5.0, 10.0, 15.0], "b": [10.0, 20.0, 30.0]})
    validator = Le("a", "b")
    ctx = ValidationContext(root_data=df)
    result = validator.validate(df, ctx)
    assert result.success

  def test_lt_two_columns_valid(self) -> None:
    """Test Lt['a', 'b'] passes when a < b for all rows."""
    df = pd.DataFrame({"a": [5.0, 10.0, 15.0], "b": [10.0, 20.0, 30.0]})
    validator = Lt("a", "b")
    ctx = ValidationContext(root_data=df)
    result = validator.validate(df, ctx)
    assert result.success

  def test_three_columns_valid(self) -> None:
    """Test Ge['a', 'b', 'c'] passes when a >= b >= c."""
    df = pd.DataFrame({
      "a": [30.0, 30.0, 30.0],
      "b": [20.0, 20.0, 20.0],
      "c": [10.0, 10.0, 10.0],
    })
    validator = Ge("a", "b", "c")
    ctx = ValidationContext(root_data=df)
    result = validator.validate(df, ctx)
    assert result.success

  def test_three_columns_middle_fails(self) -> None:
    """Test Ge['a', 'b', 'c'] fails when b < c."""
    df = pd.DataFrame({
      "a": [30.0],
      "b": [5.0],  # b < c, should fail
      "c": [10.0],
    })
    validator = Ge("a", "b", "c")
    ctx = ValidationContext(root_data=df)
    result = validator.validate(df, ctx)
    assert not result.success

  def test_large_data_uses_numba(self) -> None:
    """Test that large data uses Numba acceleration."""
    # Create large dataset that should trigger Numba
    n = 100_000
    a = np.random.rand(n) + 10.0
    b = np.random.rand(n)
    df = pd.DataFrame({"a": a, "b": b})

    validator = Ge("a", "b")
    ctx = ValidationContext(root_data=df)
    result = validator.validate(df, ctx)
    assert result.success

  def test_large_data_fails_correctly(self) -> None:
    """Test that large data correctly reports failures."""
    n = 100_000
    a = np.random.rand(n)
    b = np.random.rand(n) + 10.0  # b > a, should fail
    df = pd.DataFrame({"a": a, "b": b})

    validator = Ge("a", "b")
    ctx = ValidationContext(root_data=df)
    result = validator.validate(df, ctx)
    assert not result.success


class TestNumbaColumnComparisonWithDecorator:
  """Test @validate decorator with multi-column comparison."""

  def test_validate_decorator_ge_columns(self) -> None:
    """Test @validate decorator works with Ge['a', 'b']."""

    @validate
    def process(df: Validated[pd.DataFrame, Ge("high", "low")]) -> bool:
      del df  # Used for validation
      return True

    # Valid case
    df_valid = pd.DataFrame({"high": [10.0, 20.0], "low": [5.0, 10.0]})
    assert process(df_valid)

    # Invalid case
    df_invalid = pd.DataFrame({"high": [5.0, 20.0], "low": [10.0, 10.0]})
    with pytest.raises(ValidationError):
      process(df_invalid)
