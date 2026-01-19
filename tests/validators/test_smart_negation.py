"""Tests for smart negation (negate() methods) in validators."""

import pandas as pd
import pytest

from datawarden import Validated, validate
from datawarden.exceptions import ValidationError
from datawarden.validators import Ge, Gt, Le, Lt, Negative, Positive
from datawarden.validators.base import Not


class TestSmartNegation:
  """Tests for smart negation via negate() methods."""

  def test_not_positive_uses_smart_negation(self) -> None:
    """Test that Not(Positive) uses NonPositive via smart negation."""

    @validate
    def func(data: Validated[pd.Series, Not(Positive)]) -> float:
      return data.sum()

    # Valid: non-positive values
    assert func(pd.Series([0, -1, -5])) == -6

    # Invalid: positive value
    with pytest.raises(ValidationError):
      func(pd.Series([0, 1, -1]))

  def test_not_negative_uses_smart_negation(self) -> None:
    """Test that Not(Negative) uses NonNegative via smart negation."""

    @validate
    def func(data: Validated[pd.Series, Not(Negative)]) -> float:
      return data.sum()

    # Valid: non-negative values
    assert func(pd.Series([0, 1, 5])) == 6

    # Invalid: negative value
    with pytest.raises(ValidationError):
      func(pd.Series([0, -1, 1]))


class TestComparisonNegation:
  """Tests for comparison operator negation."""

  def test_not_ge_uses_smart_negation(self) -> None:
    """Test ~Ge(5) uses Lt(5) via smart negation."""

    @validate
    def func(data: Validated[pd.Series, Not(Ge(5))]) -> float:
      return data.sum()

    # Valid: < 5
    assert func(pd.Series([1, 2, 3])) == 6

    # Invalid: >= 5
    with pytest.raises(ValidationError):
      func(pd.Series([3, 5, 7]))

  def test_not_lt_uses_smart_negation(self) -> None:
    """Test ~Lt(5) uses Ge(5) via smart negation."""

    @validate
    def func(data: Validated[pd.Series, Not(Lt(5))]) -> float:
      return data.sum()

    # Valid: >= 5
    assert func(pd.Series([5, 6, 7])) == 18

    # Invalid: < 5
    with pytest.raises(ValidationError):
      func(pd.Series([3, 5, 7]))

  def test_not_le_uses_smart_negation(self) -> None:
    """Test ~Le(5) uses Gt(5) via smart negation."""

    @validate
    def func(data: Validated[pd.Series, Not(Le(5))]) -> float:
      return data.sum()

    # Valid: > 5
    assert func(pd.Series([6, 7, 8])) == 21

    # Invalid: <= 5
    with pytest.raises(ValidationError):
      func(pd.Series([5, 6, 7]))

  def test_not_gt_uses_smart_negation(self) -> None:
    """Test ~Gt(5) uses Le(5) via smart negation."""

    @validate
    def func(data: Validated[pd.Series, Not(Gt(5))]) -> float:
      return data.sum()

    # Valid: <= 5
    assert func(pd.Series([3, 4, 5])) == 12

    # Invalid: > 5
    with pytest.raises(ValidationError):
      func(pd.Series([5, 6, 7]))
