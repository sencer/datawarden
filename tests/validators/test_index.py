"""Tests for index validators: Datetime, Unique, MonoUp, MonoDown, Index."""
# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false
# pyright: reportCallIssue=false, reportAttributeAccessIssue=false

import pandas as pd
import pytest

from validated import Datetime, Index, MonoDown, MonoUp, Unique


class TestDatetime:
  """Tests for Datetime validator."""

  def test_valid_datetime_index(self):
    """Test Datetime validator with valid DatetimeIndex."""
    dates = pd.date_range("2024-01-01", periods=3)
    data = pd.Series([1, 2, 3], index=dates)
    validator = Index(Datetime)
    result = validator.validate(data)
    assert result.equals(data)

  def test_invalid_int_index(self):
    """Test Datetime validator rejects integer index."""
    data = pd.Series([1, 2, 3])
    validator = Index(Datetime)
    with pytest.raises(ValueError, match="Index must be DatetimeIndex"):
      validator.validate(data)

  def test_dataframe_with_datetime_index(self):
    """Test Datetime validator with DataFrame."""
    dates = pd.date_range("2024-01-01", periods=3)
    data = pd.DataFrame({"a": [1, 2, 3]}, index=dates)
    validator = Index(Datetime)
    result = validator.validate(data)
    assert result.equals(data)

  def test_non_pandas_type(self):
    """Test Datetime validator with non-pandas type."""
    validator = Index(Datetime)
    result = validator.validate([1, 2, 3])
    assert result == [1, 2, 3]


class TestUnique:
  """Tests for Unique validator."""

  def test_unique_series(self):
    validator = Unique()

    # Valid
    data = pd.Series([1, 2, 3])
    assert validator.validate(data) is data

    # Invalid
    data = pd.Series([1, 2, 1])
    with pytest.raises(ValueError, match="Values must be unique"):
      validator.validate(data)

  def test_unique_index(self):
    validator = Unique()

    # Valid
    index = pd.Index([1, 2, 3])
    assert validator.validate(index).equals(index)

    # Invalid
    index = pd.Index([1, 2, 1])
    with pytest.raises(ValueError, match="Values must be unique"):
      validator.validate(index)

  def test_index_unique_series(self):
    """Test Index[Unique] with Series."""
    validator = Index(Unique)

    # Valid
    data = pd.Series([1, 2, 3], index=[1, 2, 3])
    assert validator.validate(data) is data

    # Invalid
    data = pd.Series([1, 2, 3], index=[1, 1, 3])
    with pytest.raises(ValueError, match="Values must be unique"):
      validator.validate(data)

  def test_index_unique_dataframe(self):
    """Test Index[Unique] with DataFrame."""
    validator = Index(Unique)

    # Valid
    data = pd.DataFrame({"a": [1, 2]}, index=pd.Index([1, 2]))
    assert validator.validate(data) is data

    # Invalid
    data = pd.DataFrame({"a": [1, 2]}, index=pd.Index([1, 1]))
    with pytest.raises(ValueError, match="Values must be unique"):
      validator.validate(data)


class TestMonoUp:
  """Tests for MonoUp (monotonically increasing) validator."""

  def test_valid_series_increasing(self):
    """Test MonoUp validator with valid increasing Series."""
    data = pd.Series([1, 2, 3, 4, 5])
    validator = MonoUp()
    result = validator.validate(data)
    assert result.equals(data)

  def test_valid_series_equal(self):
    """Test MonoUp validator allows equal consecutive values."""
    data = pd.Series([1, 2, 2, 3, 3, 4])
    validator = MonoUp()
    result = validator.validate(data)
    assert result.equals(data)

  def test_invalid_series_decreasing(self):
    """Test MonoUp validator rejects decreasing values."""
    data = pd.Series([1, 2, 3, 2, 5])
    validator = MonoUp()
    with pytest.raises(ValueError, match="must be monotonically increasing"):
      validator.validate(data)

  def test_valid_index(self):
    """Test MonoUp validator with valid increasing Index."""
    data = pd.Index([1, 2, 3, 4, 5])
    validator = MonoUp()
    result = validator.validate(data)
    assert result.equals(data)

  def test_invalid_index(self):
    """Test MonoUp validator rejects non-monotonic Index."""
    data = pd.Index([1, 2, 3, 2, 5])
    validator = MonoUp()
    with pytest.raises(ValueError, match="must be monotonically increasing"):
      validator.validate(data)

  def test_index_monoup(self):
    """Test Index[MonoUp] validator with monotonic index."""
    data = pd.Series([1, 2, 3], index=[0, 1, 2])
    validator = Index(MonoUp)
    result = validator.validate(data)
    assert result.equals(data)

  def test_non_monotonic_index(self):
    """Test Index[MonoUp] validator rejects non-monotonic index."""
    data = pd.Series([1, 2, 3], index=[0, 2, 1])
    validator = Index(MonoUp)
    with pytest.raises(ValueError, match="must be monotonically increasing"):
      validator.validate(data)

  def test_datetime_monotonic(self):
    """Test Index[MonoUp] validator with datetime index."""
    dates = pd.date_range("2024-01-01", periods=3)
    data = pd.Series([1, 2, 3], index=dates)
    validator = Index(MonoUp)
    result = validator.validate(data)
    assert result.equals(data)

  def test_datetime_non_monotonic(self):
    """Test Index[MonoUp] validator rejects non-monotonic datetime."""
    dates = [
      pd.Timestamp("2024-01-01"),
      pd.Timestamp("2024-01-03"),
      pd.Timestamp("2024-01-02"),
    ]
    data = pd.Series([1, 2, 3], index=dates)
    validator = Index(MonoUp)
    with pytest.raises(ValueError, match="must be monotonically increasing"):
      validator.validate(data)


class TestMonoDown:
  """Tests for MonoDown (monotonically decreasing) validator."""

  def test_valid_series_decreasing(self):
    """Test MonoDown validator with valid decreasing Series."""
    data = pd.Series([5, 4, 3, 2, 1])
    validator = MonoDown()
    result = validator.validate(data)
    assert result.equals(data)

  def test_valid_series_equal(self):
    """Test MonoDown validator allows equal consecutive values."""
    data = pd.Series([5, 4, 4, 3, 3, 2])
    validator = MonoDown()
    result = validator.validate(data)
    assert result.equals(data)

  def test_invalid_series_increasing(self):
    """Test MonoDown validator rejects increasing values."""
    data = pd.Series([5, 4, 3, 4, 1])
    validator = MonoDown()
    with pytest.raises(ValueError, match="must be monotonically decreasing"):
      validator.validate(data)

  def test_valid_index(self):
    """Test MonoDown validator with valid decreasing Index."""
    data = pd.Index([5, 4, 3, 2, 1])
    validator = MonoDown()
    result = validator.validate(data)
    assert result.equals(data)

  def test_invalid_index(self):
    """Test MonoDown validator rejects non-monotonic Index."""
    data = pd.Index([5, 4, 3, 4, 1])
    validator = MonoDown()
    with pytest.raises(ValueError, match="must be monotonically decreasing"):
      validator.validate(data)


class TestIndexValidator:
  """Test Index validator edge cases."""

  def test_index_with_non_dataframe(self):
    """Test Index validator with Index object (no-op)."""
    index = pd.Index([1, 2, 3, 4, 5])
    validator = Index(Unique)
    result = validator.validate(index)
    assert result.equals(index)

  def test_index_with_validator_instance(self):
    """Test Index validator with validator instance."""
    df = pd.DataFrame({"a": [1, 2, 3]}, index=[0, 1, 2])
    validator = Index(Unique())
    result = validator.validate(df)
    assert result.equals(df)

  def test_index_with_multiple_validators(self):
    """Test Index validator with multiple validators."""
    index = pd.date_range("2023-01-01", periods=5, freq="D")
    df = pd.DataFrame({"a": [1, 2, 3, 4, 5]}, index=index)
    validator = Index(Datetime, MonoUp)
    result = validator.validate(df)
    assert result.equals(df)
