"""Tests for standardized index validation."""

import numpy as np
import pandas as pd
import pytest

from datawarden import (
  Finite,
  Ge,
  Gt,
  Index,
  IsNaN,
  Le,
  Lt,
  Negative,
  Not,
  Positive,
  Shape,
  StrictFinite,
)


class TestStandardizedIndexValidation:
  """Tests for validators that now support pd.Index explicitly."""

  def test_finite_on_index(self):
    """Test Finite validator works on pd.Index."""
    # Valid
    idx = pd.Index([1.0, 2.0, 3.0])
    assert Finite().validate(idx) is None

    # Check invalid input (Inf)
    idx_inf = pd.Index([1.0, np.inf, 3.0])
    with pytest.raises(ValueError, match="contains Inf"):
      Finite().validate(idx_inf)

  def test_strict_finite_on_index(self):
    """Test StrictFinite validator works on pd.Index."""
    # Valid
    idx = pd.Index([1.0, 2.0, 3.0])
    assert StrictFinite().validate(idx) is None

    # Check invalid input (Inf)
    idx_inf = pd.Index([1.0, np.inf, 3.0])
    with pytest.raises(ValueError, match="contains NaN or Inf"):
      StrictFinite().validate(idx_inf)

    # Check invalid input (NaN)
    idx_nan = pd.Index([1.0, np.nan, 3.0])
    with pytest.raises(ValueError, match="contains NaN or Inf"):
      StrictFinite().validate(idx_nan)

  def test_index_wrapper_with_finite(self):
    """Test Index[Finite] correctly validates an Index object."""
    # This was previously a no-op, now it should validate
    idx_inf = pd.Index([1.0, np.inf, 3.0])
    validator = Index(Finite)
    with pytest.raises(ValueError, match="contains Inf"):
      validator.validate(idx_inf)

  def test_non_nan_on_index(self):
    """Test Not(IsNaN) validator works on pd.Index."""
    # Valid
    idx = pd.Index([1.0, 2.0, 3.0])
    assert Not(IsNaN()).validate(idx) is None

    # Check invalid input (NaN)
    idx_nan = pd.Index([1.0, np.nan, 3.0])
    with pytest.raises(ValueError, match="Data must not contain NaN"):
      Not(IsNaN()).validate(idx_nan)

  def test_comparison_on_index(self):
    """Test comparison validators (Ge, Le, etc.) on pd.Index."""
    idx = pd.Index([1, 2, 3])

    # Ge
    assert Ge(0).validate(idx) is None
    with pytest.raises(ValueError, match=">= 5"):
      Ge(5).validate(idx)

    # Le
    assert Le(5).validate(idx) is None
    with pytest.raises(ValueError, match="<= 0"):
      Le(0).validate(idx)

    # Gt
    assert Gt(0).validate(idx) is None
    with pytest.raises(ValueError, match="> 3"):
      Gt(3).validate(idx)

    # Lt
    assert Lt(5).validate(idx) is None
    with pytest.raises(ValueError, match="< 1"):
      Lt(1).validate(idx)

  def test_shape_on_index(self):
    """Test Shape validator on pd.Index (checking length)."""
    idx = pd.Index([1, 2, 3])

    # Exact match
    assert Shape(3).validate(idx) is None

    # Mismatch
    with pytest.raises(ValueError, match="Index must have == 5 rows"):
      Shape(5).validate(idx)

    # Constraint
    assert Shape(Ge(3)).validate(idx) is None
    with pytest.raises(ValueError, match="Index must have < 3 rows"):
      Shape(Lt(3)).validate(idx)

  def test_non_negative_on_index(self):
    """Test Not(Negative) validator works on pd.Index."""
    # Valid
    idx = pd.Index([0, 1, 2])
    assert Not(Negative()).validate(idx) is None

    # Check invalid input (< 0)
    idx_neg = pd.Index([1, -1, 2])
    with pytest.raises(ValueError, match="must be >= 0"):
      Not(Negative()).validate(idx_neg)

  def test_positive_on_index(self):
    """Test Positive validator works on pd.Index."""
    # Valid
    idx = pd.Index([1, 2, 3])
    assert Positive().validate(idx) is None

    # Check invalid input (<= 0)
    idx_zero = pd.Index([0, 1, 2])
    with pytest.raises(ValueError, match="must be positive"):
      Positive().validate(idx_zero)

  def test_negative_on_index(self):
    """Test Negative validator works on pd.Index."""

    # Valid
    idx = pd.Index([-1, -2, -3])
    assert Negative().validate(idx) is None

    # Check invalid input (>= 0)
    idx_pos = pd.Index([-1, 0, -2])
    with pytest.raises(ValueError, match="must be negative"):
      Negative().validate(idx_pos)

  def test_non_positive_on_index(self):
    """Test Not(Positive) validator works on pd.Index."""

    # Valid
    idx = pd.Index([0, -1, -2])
    assert Not(Positive()).validate(idx) is None

    # Check invalid input (> 0)
    idx_pos = pd.Index([-1, 1, -2])
    with pytest.raises(ValueError, match="must be <= 0"):
      Not(Positive()).validate(idx_pos)
