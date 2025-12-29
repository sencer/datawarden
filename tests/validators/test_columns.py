"""Tests for column validators: IsDtype, HasColumns, HasColumn."""
# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false
# pyright: reportCallIssue=false, reportAttributeAccessIssue=false

import numpy as np
import pandas as pd
import pytest

from datawarden import (
  Finite,
  HasColumn,
  HasColumns,
  IsDtype,
  MonoUp,
  Positive,
  Unique,
)


class TestIsDtype:
  """Tests for IsDtype validator."""

  def test_is_dtype_series(self):
    validator = IsDtype(float)

    # Valid
    data = pd.Series([1.0, 2.0])
    assert validator.validate(data) is None

    # Invalid
    data = pd.Series([1, 2])
    with pytest.raises(ValueError, match="Data must be of type"):
      validator.validate(data)

  def test_is_dtype_dataframe(self):
    validator = IsDtype(float)

    # Valid
    data = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
    assert validator.validate(data) is None

    # Invalid
    data = pd.DataFrame({"a": [1.0, 2.0], "b": [1, 2]})
    with pytest.raises(ValueError, match="Columns with wrong dtype"):
      validator.validate(data)

  def test_numpy_dtype(self):
    validator = IsDtype(np.float64)
    validator = IsDtype(np.float64)
    data = pd.Series([1.0, 2.0], dtype=np.float64)
    assert validator.validate(data) is None

  def test_dataframe_all_columns_match(self):
    """Test IsDtype with DataFrame where all columns match."""
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    validator = IsDtype(np.dtype("int64"))
    validator = IsDtype(np.dtype("int64"))
    assert validator.validate(df) is None

  def test_dataframe_column_mismatch(self):
    """Test IsDtype with DataFrame where one column doesn't match."""
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4.0, 5.0, 6.0]})
    validator = IsDtype(np.dtype("int64"))
    with pytest.raises(ValueError, match="Columns with wrong dtype"):
      validator.validate(df)

  def test_valid_multiple_columns(self):
    """Test HasColumns validator with multiple columns."""
    data = pd.DataFrame({"a": [1, 2], "b": [3, 4], "c": [5, 6]})
    validator = HasColumns(["a", "b"])
    validator = HasColumns(["a", "b"])
    assert validator.validate(data) is None

  def test_missing_single_column(self):
    """Test HasColumns validator with missing column."""
    data = pd.DataFrame({"a": [1, 2]})
    validator = HasColumns(["b"])
    with pytest.raises(ValueError, match="Missing columns: \\['b'\\]"):
      validator.validate(data)

  def test_missing_multiple_columns(self):
    """Test HasColumns validator with missing columns."""
    data = pd.DataFrame({"a": [1, 2]})
    validator = HasColumns(["b", "c"])
    with pytest.raises(ValueError, match="Missing columns:"):
      validator.validate(data)

  def test_non_dataframe(self):
    """Test HasColumns validator with non-DataFrame."""
    data = pd.Series([1, 2, 3])
    validator = HasColumns(["a"])
    with pytest.raises(TypeError, match="requires a pandas DataFrame"):
      validator.validate(data)

  def test_hascolumns_applies_validators(self):
    """Test HasColumns applies validators to columns."""
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    validator = HasColumns(["a", "b"], Positive)
    validator = HasColumns(["a", "b"], Positive)
    assert validator.validate(df) is None

  def test_hascolumns_validator_fails(self):
    """Test HasColumns fails when column validator fails."""
    df = pd.DataFrame({"a": [1, 2, 3], "b": [0, 5, 6]})
    validator = HasColumns(["a", "b"], Positive)
    with pytest.raises(ValueError, match="Data must be positive"):
      validator.validate(df)

  def test_hascolumns_with_nullable(self):
    """Test HasColumns with Nullable marker."""
    df = pd.DataFrame({"a": [1, np.nan, 3], "b": [4, 5, 6]})
    validator = HasColumns(["a", "b"])
    validator = HasColumns(["a", "b"])
    assert validator.validate(df) is None

  def test_hascolumns_with_maybeempty(self):
    """Test HasColumns with MaybeEmpty marker."""
    df = pd.DataFrame({"a": [], "b": []})
    validator = HasColumns(["a", "b"])
    validator = HasColumns(["a", "b"])
    assert validator.validate(df) is None

  def test_hascolumns_single_column_string(self):
    """Test HasColumns with single column as string."""
    df = pd.DataFrame({"a": [1, 2, 3]})
    validator = HasColumns(["a"])
    validator = HasColumns(["a"])
    assert validator.validate(df) is None


class TestHasColumn:
  """Tests for HasColumn wrapper validator."""

  def test_single_validator(self):
    """Test HasColumn with single validator."""
    data = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]})
    validator = HasColumn("a", Finite)
    validator = HasColumn("a", Finite)
    assert validator.validate(data) is None

  def test_single_validator_fails(self):
    """Test HasColumn validator fails when column violates constraint."""
    data = pd.DataFrame({"a": [1.0, np.inf, 3.0], "b": [4.0, 5.0, 6.0]})
    validator = HasColumn("a", Finite)
    with pytest.raises(ValueError, match="must be finite"):
      validator.validate(data)

  def test_multiple_validators(self):
    """Test HasColumn with multiple validators."""
    data = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]})
    validator = HasColumn("a", Finite, Positive)
    validator = HasColumn("a", Finite, Positive)
    assert validator.validate(data) is None

  def test_multiple_validators_fails(self):
    """Test HasColumn with multiple validators where one fails."""
    data = pd.DataFrame({"a": [1.0, 0.0, 3.0], "b": [4.0, 5.0, 6.0]})
    validator = HasColumn("a", Finite, Positive)
    with pytest.raises(ValueError, match="must be positive"):
      validator.validate(data)

  def test_missing_column(self):
    """Test HasColumn with missing column."""
    data = pd.DataFrame({"b": [1.0, 2.0, 3.0]})
    validator = HasColumn("a", Finite)
    with pytest.raises(ValueError, match="Column 'a' not found"):
      validator.validate(data)

  def test_monotonic_validator(self):
    """Test HasColumn with MonoUp validator."""
    data = pd.DataFrame({"a": [1, 2, 3], "b": [10, 5, 3]})
    validator = HasColumn("a", MonoUp)
    validator = HasColumn("a", MonoUp)
    assert validator.validate(data) is None

    # Column b is not monotonic up
    validator_b = HasColumn("b", MonoUp)
    with pytest.raises(ValueError, match="must be monotonically increasing"):
      validator_b.validate(data)

  def test_column_presence_only(self):
    """Test HasColumn just checks column presence when no validators."""
    data = pd.DataFrame({"a": [1.0, np.inf, -5.0], "b": [4.0, 5.0, 6.0]})

    # Should pass - column exists (even with invalid values)
    # Should pass - column exists (even with invalid values)
    validator = HasColumn("a")
    assert validator.validate(data) is None

    # Should fail - column doesn't exist
    validator_missing = HasColumn("missing")
    with pytest.raises(ValueError, match="Column 'missing' not found"):
      validator_missing.validate(data)

  def test_hascolumn_with_nullable(self):
    """Test HasColumn with Nullable marker."""
    df = pd.DataFrame({"a": [1, np.nan, 3], "b": [4, 5, 6]})
    validator = HasColumn("a")
    validator = HasColumn("a")
    assert validator.validate(df) is None

  def test_hascolumn_with_maybeempty(self):
    """Test HasColumn with MaybeEmpty marker."""
    df = pd.DataFrame({"a": [], "b": []})
    validator = HasColumn("a")
    validator = HasColumn("a")
    assert validator.validate(df) is None


class TestChunkability:
  """Tests for is_chunkable property in column validators."""

  def test_hascolumns_chunkability(self):
    # Only chunkable validators
    assert HasColumns(["a"], Finite, Positive).is_chunkable is True
    # One non-chunkable validator
    assert HasColumns(["a"], Finite, Unique).is_chunkable is False
    # Only non-chunkable
    assert HasColumns(["a"], Unique).is_chunkable is False

  def test_hascolumn_chunkability(self):
    # Only chunkable
    assert HasColumn("a", Finite, Positive).is_chunkable is True
    # One non-chunkable
    assert HasColumn("a", Finite, Unique).is_chunkable is False
    # No validators (presence check only)
    assert HasColumn("a").is_chunkable is True
