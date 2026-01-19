"""Tests for MinDiff validator."""

import numpy as np
import pandas as pd
import pytest

from datawarden import Validated, validate
from datawarden.context import ValidationContext
from datawarden.exceptions import ValidationError
from datawarden.validators import And, MinDiff, MonoUp, Not, Positive


class TestMinDiff:
  """Test MinDiff validator."""

  def test_mindiff_valid(self) -> None:
    """Test MinDiff passes when all diffs >= value."""
    s = pd.Series([1.0, 3.0, 6.0, 10.0])  # diffs: 2, 3, 4
    ctx = ValidationContext(root_data=s)
    result = MinDiff(1.5).validate(s, ctx)
    assert result.success

  def test_mindiff_invalid(self) -> None:
    """Test MinDiff fails when any diff < value."""
    s = pd.Series([1.0, 1.5, 6.0, 10.0])  # diffs: 0.5, 4.5, 4 - 0.5 < 1.5
    ctx = ValidationContext(root_data=s)
    result = MinDiff(1.5).validate(s, ctx)
    assert not result.success

  def test_mindiff_exact_boundary(self) -> None:
    """Test MinDiff passes when diff == value (inclusive)."""
    s = pd.Series([1.0, 2.5])  # diff: 1.5
    ctx = ValidationContext(root_data=s)
    result = MinDiff(1.5).validate(s, ctx)
    assert result.success

  def test_mindiff_empty(self) -> None:
    """Test MinDiff passes for empty series."""
    s = pd.Series([], dtype=float)
    ctx = ValidationContext(root_data=s)
    result = MinDiff(1.0).validate(s, ctx)
    assert result.success

  def test_mindiff_single_element(self) -> None:
    """Test MinDiff passes for single element."""
    s = pd.Series([5.0])
    ctx = ValidationContext(root_data=s)
    result = MinDiff(1.0).validate(s, ctx)
    assert result.success

  def test_mindiff_negative_diffs(self) -> None:
    """Test MinDiff uses absolute differences."""
    s = pd.Series([10.0, 5.0, -2.0])  # diffs: -5, -7 -> abs: 5, 7
    ctx = ValidationContext(root_data=s)
    result = MinDiff(3.0).validate(s, ctx)
    assert result.success

  def test_mindiff_dataframe(self) -> None:
    """Test MinDiff works with DataFrames."""
    df = pd.DataFrame({"a": [1.0, 4.0, 8.0], "b": [2.0, 6.0, 11.0]})
    ctx = ValidationContext(root_data=df)
    result = MinDiff(2.0).validate(df, ctx)
    assert result.success


class TestMinDiffNumba:
  """Test MinDiff Numba support."""

  def test_numba_properties(self) -> None:
    """Test MinDiff has correct Numba properties."""
    v = MinDiff(1.0)
    assert v.numba_supported
    assert v.numba_fusable

  def test_numba_expr(self) -> None:
    """Test MinDiff builds correct Numba expression."""
    v = MinDiff(1.5)
    expr = v.numba_expr("x")
    assert "i == 0" in expr
    assert "abs" in expr
    assert ">=" in expr
    assert "1.5" in expr

  def test_mindiff_large_data(self) -> None:
    """Test MinDiff with large dataset (triggers Numba)."""
    n = 100_000
    s = pd.Series(np.cumsum(np.ones(n) * 2.0))  # all diffs = 2.0
    ctx = ValidationContext(root_data=s)
    result = MinDiff(1.0).validate(s, ctx)
    assert result.success

  def test_mindiff_large_data_fails(self) -> None:
    """Test MinDiff fails correctly with large dataset."""
    n = 100_000
    data = np.cumsum(np.ones(n) * 2.0)
    data[50_000] = data[49_999] + 0.1  # Break minimum diff
    s = pd.Series(data)
    ctx = ValidationContext(root_data=s)
    result = MinDiff(1.0).validate(s, ctx)
    assert not result.success


class TestMinDiffComposition:
  """Test MinDiff composition with other validators."""

  def test_positive_and_mindiff(self) -> None:
    """Test Positive & MinDiff composition."""
    s = pd.Series([1.0, 3.0, 6.0, 10.0])
    validator = And(Positive, MinDiff(1.5))
    ctx = ValidationContext(root_data=s)
    result = validator.validate(s, ctx)
    assert result.success

  def test_positive_and_mindiff_large_data(self) -> None:
    """Test composition with large data uses Numba fusion."""
    n = 100_000
    s = pd.Series(np.cumsum(np.ones(n) * 2.0) + 1.0)
    validator = And(Positive, MinDiff(1.0))
    ctx = ValidationContext(root_data=s)
    result = validator.validate(s, ctx)
    assert result.success


class TestNotNumbaFusable:
  """Test Not validator numba_fusable property."""

  def test_not_monoup_fusable(self) -> None:
    """Test Not(MonoUp) is fusable."""
    assert Not(MonoUp).numba_fusable

  def test_not_positive_fusable(self) -> None:
    """Test Not(Positive) is fusable."""
    assert Not(Positive).numba_fusable

  def test_and_with_negated_fusable(self) -> None:
    """Test And(Positive, ~MonoUp()) supports Numba."""
    comp = And(Positive, Not(MonoUp))
    assert comp.numba_supported
    assert comp.numba_fusable


class TestMinDiffDecorator:
  """Test @validate decorator with MinDiff."""

  def test_validate_decorator_mindiff(self) -> None:
    """Test @validate decorator works with MinDiff."""

    @validate
    def process(s: Validated[pd.Series, MinDiff(0.5)]) -> bool:
      del s
      return True

    # Valid case
    s_valid = pd.Series([1.0, 2.0, 3.5])  # diffs: 1.0, 1.5
    assert process(s_valid)

    # Invalid case
    s_invalid = pd.Series([1.0, 1.2, 3.0])  # diff 0.2 < 0.5
    with pytest.raises(ValidationError):
      process(s_invalid)
