"""Tests for Between validator."""
# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false
# pyright: reportCallIssue=false, reportAttributeAccessIssue=false

import numpy as np
import pandas as pd
import pytest

from datawarden import Between, Outside


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

  def test_between_outside_edge_cases(self):
    v = Between(0, 10)
    with pytest.raises(TypeError, match="Between requires numeric data"):
      v.validate(pd.Series(["a"]))

    v.validate(5)  # Valid
    with pytest.raises(ValueError):
      v.validate(11)

    assert v.validate_vectorized(pd.Series(["a"])).all()

    v_out = Outside(0, 10)
    v_out.validate(11)  # Valid
    with pytest.raises(ValueError):
      v_out.validate(5)

    assert v_out.validate_vectorized(pd.Series(["a"])).all()

  def test_between_failure_pandas(self):
    v = Between(0, 10)
    with pytest.raises(ValueError, match="Data must be >= 0"):
      v.validate(pd.Series([-1]))
    with pytest.raises(ValueError, match="Data must be <= 10"):
      v.validate(pd.Series([11]))

  def test_outside_failure_pandas(self):
    v = Outside(0, 10)
    with pytest.raises(ValueError, match="Data must be outside"):
      v.validate(pd.Series([5]))

  def test_between_outside_negation(self):
    b = Between(0, 10)
    o = b.negate()
    assert isinstance(o, Outside)
    assert o.lower == 0
    assert o.upper == 10

    b2 = o.negate()
    assert isinstance(b2, Between)
    assert b2.lower == 0
    assert b2.upper == 10
