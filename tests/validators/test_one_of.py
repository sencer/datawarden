"""Tests for OneOf validator."""
# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false
# pyright: reportCallIssue=false, reportAttributeAccessIssue=false

import numpy as np
import pandas as pd
import pytest

from datawarden import Index, OneOf


class TestOneOf:
  """Tests for OneOf validator."""

  def test_validate_series_with_valid_strings_passes(self):
    """Test OneOf validator with valid string Series."""
    data = pd.Series(["a", "b", "a", "c"])
    validator = OneOf("a", "b", "c")
    assert validator.validate(data) is None

  def test_validate_series_with_invalid_strings_raises_error(self):
    """Test OneOf validator rejects invalid values."""
    data = pd.Series(["a", "b", "d"])
    validator = OneOf("a", "b", "c")
    with pytest.raises(ValueError, match="Values must be one of"):
      validator.validate(data)

  def test_validate_series_with_valid_numeric_values_passes(self):
    """Test OneOf with numeric values."""
    data = pd.Series([1, 2, 1, 3])
    validator = OneOf(1, 2, 3)
    assert validator.validate(data) is None

  def test_validate_series_with_invalid_numeric_values_raises_error(self):
    """Test OneOf rejects invalid numeric values."""
    data = pd.Series([1, 2, 4])
    validator = OneOf(1, 2, 3)
    with pytest.raises(ValueError, match="Values must be one of"):
      validator.validate(data)

  def test_validate_series_with_nan_values_passes(self):
    """Test OneOf ignores NaN values in Series."""
    data = pd.Series(["a", "b", np.nan, "c"])
    validator = OneOf("a", "b", "c")
    assert validator.validate(data) is None

  def test_validate_index_with_valid_values_passes(self):
    """Test OneOf with pd.Index."""
    data = pd.Index(["x", "y", "z"])
    validator = OneOf("x", "y", "z")
    assert validator.validate(data) is None

  def test_validate_index_with_invalid_values_raises_error(self):
    """Test OneOf rejects invalid Index values."""
    data = pd.Index(["x", "y", "w"])
    validator = OneOf("x", "y", "z")
    with pytest.raises(ValueError, match="Values must be one of"):
      validator.validate(data)

  def test_validate_dataframe_index_with_valid_values_passes(self):
    """Test OneOf with Index[] wrapper for DataFrame index."""
    df = pd.DataFrame({"a": [1, 2, 3]}, index=["x", "y", "z"])
    validator = Index(OneOf("x", "y", "z"))
    assert validator.validate(df) is None

  def test_validate_dataframe_index_with_invalid_values_raises_error(self):
    """Test OneOf with Index[] wrapper rejects invalid index."""
    df = pd.DataFrame({"a": [1, 2, 3]}, index=["x", "y", "w"])
    validator = Index(OneOf("x", "y", "z"))
    with pytest.raises(ValueError, match="Values must be one of"):
      validator.validate(df)

  def test_validate_series_with_single_allowed_value_passes(self):
    """Test OneOf with single allowed value."""
    data = pd.Series(["only"])
    validator = OneOf("only")
    assert validator.validate(data) is None

  def test_validate_with_constructor_syntax_passes(self):
    """Test OneOf with constructor syntax."""
    data = pd.Series(["a", "b"])
    validator = OneOf("a", "b", "c")
    assert validator.validate(data) is None

  def test_oneof_failure_pandas(self):
    v = OneOf(1, 2)
    with pytest.raises(ValueError, match="Values must be one of"):
      v.validate(pd.Series([3]))
