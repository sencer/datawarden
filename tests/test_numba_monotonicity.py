"""Tests for Numba-accelerated monotonicity validators."""

import numpy as np
import pandas as pd
import pytest

from datawarden import Validated, validate
from datawarden.context import ValidationContext
from datawarden.exceptions import ValidationError
from datawarden.validators import And, MonoDown, MonoUp, Positive


class TestNumbaMonoUp:
  """Test Numba acceleration for MonoUp validator."""

  def test_mono_up_valid(self) -> None:
    """Test MonoUp passes for strictly increasing data."""
    s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    ctx = ValidationContext(root_data=s)
    result = MonoUp().validate(s, ctx)
    assert result.success

  def test_mono_up_equal_valid(self) -> None:
    """Test MonoUp passes for non-decreasing data (equal values allowed)."""
    s = pd.Series([1.0, 1.0, 2.0, 2.0, 3.0])
    ctx = ValidationContext(root_data=s)
    result = MonoUp().validate(s, ctx)
    assert result.success

  def test_mono_up_invalid(self) -> None:
    """Test MonoUp fails when values decrease."""
    s = pd.Series([1.0, 2.0, 1.5, 4.0, 5.0])  # 1.5 < 2.0
    ctx = ValidationContext(root_data=s)
    result = MonoUp().validate(s, ctx)
    assert not result.success

  def test_mono_up_large_data(self) -> None:
    """Test MonoUp with large dataset (triggers Numba)."""
    n = 100_000
    s = pd.Series(np.cumsum(np.random.rand(n)))  # Cumsum is increasing
    ctx = ValidationContext(root_data=s)
    result = MonoUp().validate(s, ctx)
    assert result.success

  def test_mono_up_large_data_fails(self) -> None:
    """Test MonoUp fails correctly with large dataset."""
    n = 100_000
    data = np.cumsum(np.random.rand(n))
    data[50_000] = 0  # Break monotonicity
    s = pd.Series(data)
    ctx = ValidationContext(root_data=s)
    result = MonoUp().validate(s, ctx)
    assert not result.success


class TestNumbaMonoDown:
  """Test Numba acceleration for MonoDown validator."""

  def test_mono_down_valid(self) -> None:
    """Test MonoDown passes for strictly decreasing data."""
    s = pd.Series([5.0, 4.0, 3.0, 2.0, 1.0])
    ctx = ValidationContext(root_data=s)
    result = MonoDown().validate(s, ctx)
    assert result.success

  def test_mono_down_equal_valid(self) -> None:
    """Test MonoDown passes for non-increasing data (equal values allowed)."""
    s = pd.Series([5.0, 5.0, 3.0, 3.0, 1.0])
    ctx = ValidationContext(root_data=s)
    result = MonoDown().validate(s, ctx)
    assert result.success

  def test_mono_down_invalid(self) -> None:
    """Test MonoDown fails when values increase."""
    s = pd.Series([5.0, 4.0, 4.5, 2.0, 1.0])  # 4.5 > 4.0
    ctx = ValidationContext(root_data=s)
    result = MonoDown().validate(s, ctx)
    assert not result.success


class TestNumbaMonotonicityComposition:
  """Test composition of monotonicity validators with other validators."""

  def test_positive_and_mono_up(self) -> None:
    """Test Positive & MonoUp composition."""
    s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    validator = And(Positive, MonoUp)
    ctx = ValidationContext(root_data=s)
    result = validator.validate(s, ctx)
    assert result.success

  def test_positive_and_mono_up_fails_negative(self) -> None:
    """Test Positive & MonoUp fails on negative value."""
    s = pd.Series([-1.0, 2.0, 3.0, 4.0, 5.0])  # -1.0 is not positive
    validator = And(Positive, MonoUp)
    ctx = ValidationContext(root_data=s)
    result = validator.validate(s, ctx)
    assert not result.success

  def test_positive_and_mono_up_fails_decreasing(self) -> None:
    """Test Positive & MonoUp fails on decreasing value."""
    s = pd.Series([1.0, 2.0, 1.5, 4.0, 5.0])  # Decreasing at 1.5
    validator = And(Positive, MonoUp)
    ctx = ValidationContext(root_data=s)
    result = validator.validate(s, ctx)
    assert not result.success

  def test_positive_and_mono_up_large_data(self) -> None:
    """Test Positive & MonoUp with large dataset (should fuse in Numba)."""
    n = 100_000
    s = pd.Series(np.cumsum(np.random.rand(n)) + 1.0)  # All positive & increasing
    validator = And(Positive, MonoUp)
    ctx = ValidationContext(root_data=s)
    result = validator.validate(s, ctx)
    assert result.success


class TestMonoUpWithDecorator:
  """Test @validate decorator with monotonicity validators."""

  def test_validate_decorator_mono_up(self) -> None:
    """Test @validate decorator works with MonoUp."""

    @validate
    def process(s: Validated[pd.Series, MonoUp]) -> bool:
      del s
      return True

    # Valid case
    s_valid = pd.Series([1.0, 2.0, 3.0])
    assert process(s_valid)

    # Invalid case
    s_invalid = pd.Series([1.0, 0.5, 3.0])
    with pytest.raises(ValidationError):
      process(s_invalid)
