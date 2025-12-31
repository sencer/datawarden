"""Tests for Shape validator."""
# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false
# pyright: reportCallIssue=false, reportAttributeAccessIssue=false

from typing import Any

import pandas as pd
import pytest

from datawarden import Ge, Gt, Le, Lt, Shape


class TestShape:
  """Tests for Shape validator."""

  def test_validate_exact_shape_passes(self):
    """Test Shape with exact dimensions."""
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    validator = Shape(3, 2)
    assert validator.validate(df) is None

  def test_validate_exact_shape_incorrect_rows_raises_error(self):
    """Test Shape fails when rows don't match."""
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    validator = Shape(3, 2)
    with pytest.raises(ValueError, match="must have == 3 rows"):
      validator.validate(df)

  def test_validate_exact_shape_incorrect_cols_raises_error(self):
    """Test Shape fails when cols don't match."""
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6], "c": [7, 8, 9]})
    validator = Shape(3, 2)
    with pytest.raises(ValueError, match="must have == 2 columns"):
      validator.validate(df)

  def test_validate_ge_constraint_passes(self):
    """Test Shape with Ge constraint."""
    df = pd.DataFrame({"a": [1, 2, 3, 4, 5], "b": [1, 2, 3, 4, 5]})
    validator = Shape(Ge(3), Any)
    assert validator.validate(df) is None

  def test_validate_ge_constraint_fails_raises_error(self):
    """Test Shape with Ge constraint fails."""
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    validator = Shape(Ge(5), Any)
    with pytest.raises(ValueError, match="must have >= 5 rows"):
      validator.validate(df)

  def test_validate_le_constraint_passes(self):
    """Test Shape with Le constraint."""
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    validator = Shape(Any, Le(5))
    assert validator.validate(df) is None

  def test_validate_le_constraint_fails_raises_error(self):
    """Test Shape with Le constraint fails."""
    df = pd.DataFrame({"a": [1], "b": [2], "c": [3], "d": [4], "e": [5], "f": [6]})
    validator = Shape(Any, Le(3))
    with pytest.raises(ValueError, match="must have <= 3 columns"):
      validator.validate(df)

  def test_validate_gt_constraint_passes(self):
    """Test Shape with Gt constraint."""
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    validator = Shape(Gt(2), Any)
    assert validator.validate(df) is None

  def test_validate_gt_constraint_fails_raises_error(self):
    """Test Shape with Gt constraint fails (equal not allowed)."""
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    validator = Shape(Gt(2), Any)
    with pytest.raises(ValueError, match="must have > 2 rows"):
      validator.validate(df)

  def test_validate_lt_constraint_passes(self):
    """Test Shape with Lt constraint."""
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    validator = Shape(Any, Lt(5))
    assert validator.validate(df) is None

  def test_validate_lt_constraint_fails_raises_error(self):
    """Test Shape with Lt constraint fails."""
    df = pd.DataFrame({"a": [1], "b": [2], "c": [3], "d": [4], "e": [5]})
    validator = Shape(Any, Lt(5))
    with pytest.raises(ValueError, match="must have < 5 columns"):
      validator.validate(df)

  def test_validate_any_shape_passes(self):
    """Test Shape with Any for both dimensions."""
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    validator = Shape(Any, Any)
    assert validator.validate(df) is None

  def test_validate_series_exact_shape_passes(self):
    """Test Shape with Series and exact row count."""
    series = pd.Series([1, 2, 3, 4, 5])
    validator = Shape(5)
    assert validator.validate(series) is None

  def test_validate_series_exact_shape_incorrect_rows_raises_error(self):
    """Test Shape with Series fails when rows don't match."""
    series = pd.Series([1, 2, 3])
    validator = Shape(5)
    with pytest.raises(ValueError, match="must have == 5 rows"):
      validator.validate(series)

  def test_validate_series_ge_constraint_passes(self):
    """Test Shape with Series and Ge constraint."""
    series = pd.Series([1, 2, 3, 4, 5])
    validator = Shape(Ge(3))
    assert validator.validate(series) is None

  def test_validate_combined_constraints_passes(self):
    """Test Shape with different constraints for rows and cols."""
    df = pd.DataFrame({"a": range(10), "b": range(10), "c": range(10)})
    validator = Shape(Ge(5), Le(5))
    assert validator.validate(df) is None

  def test_initialization_with_invalid_constraint_type_raises_type_error(self):
    """Test passing invalid type to Shape constructor."""
    with pytest.raises(TypeError, match="Invalid shape constraint"):
      Shape("not_int_or_constraint")

  def test_internal_any_dim_describe_returns_correct_string(self):
    """Test internal describe method of Any wrapper."""
    # Since logic normally prevents describe() from reaching (check always True)
    # we inspect the internal object manually to ensure it works.
    v = Shape(Any)
    assert v.rows.describe() == "any"

  def test_shape_validator_failures(self):
    s_rows = Shape(5)
    with pytest.raises(ValueError, match="must have == 5 rows"):
      s_rows.validate(pd.Series([1, 2]))

    s_df = Shape(5, 2)
    with pytest.raises(ValueError, match="must have == 5 rows"):
      s_df.validate(pd.DataFrame({"a": [1]}, index=[0]))

    with pytest.raises(ValueError, match="must have == 2 columns"):
      s_df.validate(pd.DataFrame({"a": [1, 2, 3, 4, 5]}))

  def test_shape_invalid_constraint(self):
    with pytest.raises(TypeError, match="Invalid shape constraint"):
      Shape("bad")

    with pytest.raises(TypeError, match="Invalid shape constraint"):
      Shape(10, "bad")
