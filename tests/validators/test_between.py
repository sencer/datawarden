"""Tests for Between validator."""
# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false
# pyright: reportCallIssue=false, reportAttributeAccessIssue=false

import numpy as np
import pandas as pd
import pytest

from datawarden import Between


class TestBetween:
  """Tests for Between validator."""

  def test_validate_inclusive_both_passes(self):
    """Test [lower, upper] behavior."""
    v = Between(0, 10, inclusive=(True, True))
    assert v.validate(pd.Series([0, 5, 10])) is None
    with pytest.raises(ValueError, match="must be >= 0"):
      v.validate(pd.Series([-1]))
    with pytest.raises(ValueError, match="must be <= 10"):
      v.validate(pd.Series([11]))

  def test_validate_inclusive_left_only_passes(self):
    """Test [lower, upper) behavior."""
    v = Between(0, 10, inclusive=(True, False))
    assert v.validate(pd.Series([0, 5, 9.9])) is None
    with pytest.raises(ValueError, match="must be < 10"):
      v.validate(pd.Series([10]))

  def test_validate_inclusive_right_only_passes(self):
    """Test (lower, upper] behavior."""
    v = Between(0, 10, inclusive=(False, True))
    assert v.validate(pd.Series([0.1, 5, 10])) is None
    with pytest.raises(ValueError, match="must be > 0"):
      v.validate(pd.Series([0]))

  def test_validate_exclusive_both_passes(self):
    """Test (lower, upper) behavior."""
    v = Between(0, 10, inclusive=(False, False))
    assert v.validate(pd.Series([0.1, 5, 9.9])) is None
    with pytest.raises(ValueError, match="must be > 0"):
      v.validate(pd.Series([0]))
    with pytest.raises(ValueError, match="must be < 10"):
      v.validate(pd.Series([10]))

  def test_validate_nan_handling_raises_error(self):
    """Test NaN rejection."""
    v = Between(0, 10)
    with pytest.raises(ValueError, match="Cannot validate range with NaN"):
      v.validate(pd.Series([5, np.nan]))

  def test_repr(self):
    v1 = Between(0, 10)
    assert repr(v1) == "Between(0, 10)"
    v2 = Between(0, 10, inclusive=(False, False))
    assert "inclusive=(False, False)" in repr(v2)
