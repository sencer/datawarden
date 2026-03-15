"""Tests for index validators: Datetime, Unique, MonoUp, MonoDown, Index."""

import pandas as pd
import pytest

from datawarden import Validated, validate
from datawarden.exceptions import ValidationError
from datawarden.validators import Datetime, Index, MonoDown, MonoUp, Unique


class TestDatetime:
  """Tests for Datetime validator."""

  def test_valid_datetime_index(self):
    """Test Datetime validator with valid DatetimeIndex."""

    @validate
    def process(data: Validated[pd.Series, Index(Datetime)]) -> pd.Series:
      return data

    dates = pd.date_range("2024-01-01", periods=3)
    data = pd.Series([1, 2, 3], index=dates)
    result = process(data)
    assert result.equals(data)

  def test_invalid_int_index(self):
    """Test Datetime validator rejects integer index."""

    @validate
    def process(data: Validated[pd.Series, Index(Datetime)]) -> pd.Series:
      return data

    data = pd.Series([1, 2, 3])
    with pytest.raises(ValidationError):
      process(data)

  def test_dataframe_with_datetime_index(self):
    """Test Datetime validator with DataFrame."""

    @validate
    def process(data: Validated[pd.DataFrame, Index(Datetime)]) -> pd.DataFrame:
      return data

    dates = pd.date_range("2024-01-01", periods=3)
    data = pd.DataFrame({"a": [1, 2, 3]}, index=dates)
    result = process(data)
    assert result.equals(data)


class TestUnique:
  """Tests for Unique validator."""

  def test_unique_series(self):
    """Test Unique validator with Series."""

    @validate
    def process(data: Validated[pd.Series, Unique]) -> pd.Series:
      return data

    # Valid
    data = pd.Series([1, 2, 3])
    result = process(data)
    assert result.equals(data)

    # Invalid
    with pytest.raises(ValidationError):
      process(pd.Series([1, 2, 1]))

  def test_index_unique_series(self):
    """Test Index[Unique] with Series."""

    @validate
    def process(data: Validated[pd.Series, Index(Unique)]) -> pd.Series:
      return data

    # Valid
    data = pd.Series([1, 2, 3], index=[1, 2, 3])
    result = process(data)
    assert result.equals(data)

    # Invalid
    with pytest.raises(ValidationError):
      process(pd.Series([1, 2, 3], index=[1, 1, 3]))

  def test_index_unique_dataframe(self):
    """Test Index[Unique] with DataFrame."""

    @validate
    def process(data: Validated[pd.DataFrame, Index(Unique)]) -> pd.DataFrame:
      return data

    # Valid
    data = pd.DataFrame({"a": [1, 2]}, index=pd.Index([1, 2]))
    result = process(data)
    assert result.equals(data)

    # Invalid
    with pytest.raises(ValidationError):
      process(pd.DataFrame({"a": [1, 2]}, index=pd.Index([1, 1])))


class TestMonoUp:
  """Tests for MonoUp (monotonically increasing) validator."""

  def test_valid_series_increasing(self):
    """Test MonoUp validator with valid increasing Series."""

    @validate
    def process(data: Validated[pd.Series, MonoUp]) -> pd.Series:
      return data

    data = pd.Series([1, 2, 3, 4, 5])
    result = process(data)
    assert result.equals(data)

  def test_valid_series_equal(self):
    """Test MonoUp validator allows equal consecutive values."""

    @validate
    def process(data: Validated[pd.Series, MonoUp]) -> pd.Series:
      return data

    data = pd.Series([1, 2, 2, 3, 3, 4])
    result = process(data)
    assert result.equals(data)

  def test_invalid_series_decreasing(self):
    """Test MonoUp validator rejects decreasing values."""

    @validate
    def process(data: Validated[pd.Series, MonoUp]) -> pd.Series:
      return data

    data = pd.Series([1, 2, 3, 2, 5])
    with pytest.raises(ValidationError):
      process(data)

  def test_monoup_strict_fails_on_equal_values(self):
    """Test MonoUp(strict=True) rejects equal consecutive values."""

    @validate
    def process(data: Validated[pd.Series, MonoUp(strict=True)]) -> pd.Series:
      return data

    data = pd.Series([1, 2, 2, 3])
    with pytest.raises(ValidationError, match=r"Not monotonically increasing \(strict\)"):
      process(data)

  def test_monoup_strict_passes_on_strictly_increasing_values(self):
    """Test MonoUp(strict=True) passes on strictly increasing values."""

    @validate
    def process(data: Validated[pd.Series, MonoUp(strict=True)]) -> pd.Series:
      return data

    data = pd.Series([1, 2, 3, 4])
    result = process(data)
    assert result.equals(data)

  def test_monoup_explicit_non_strict_passes_on_equal_values(self):
    """Test MonoUp(strict=False) explicitly allows equal consecutive values."""

    @validate
    def process(data: Validated[pd.Series, MonoUp(strict=False)]) -> pd.Series:
      return data

    data = pd.Series([1, 2, 2, 3])
    result = process(data)
    assert result.equals(data)

  def test_index_monoup(self):
    """Test Index[MonoUp] validator with monotonic index."""

    @validate
    def process(data: Validated[pd.Series, Index(MonoUp)]) -> pd.Series:
      return data

    data = pd.Series([1, 2, 3], index=[0, 1, 2])
    result = process(data)
    assert result.equals(data)

  def test_non_monotonic_index(self):
    """Test Index[MonoUp] validator rejects non-monotonic index."""

    @validate
    def process(data: Validated[pd.Series, Index(MonoUp)]) -> pd.Series:
      return data

    data = pd.Series([1, 2, 3], index=[0, 2, 1])
    with pytest.raises(ValidationError):
      process(data)

  def test_datetime_monotonic(self):
    """Test Index[MonoUp] validator with datetime index."""

    @validate
    def process(data: Validated[pd.Series, Index(MonoUp)]) -> pd.Series:
      return data

    dates = pd.date_range("2024-01-01", periods=3)
    data = pd.Series([1, 2, 3], index=dates)
    result = process(data)
    assert result.equals(data)

  def test_datetime_non_monotonic(self):
    """Test Index[MonoUp] validator rejects non-monotonic datetime."""

    @validate
    def process(data: Validated[pd.Series, Index(MonoUp)]) -> pd.Series:
      return data

    dates = [
      pd.Timestamp("2024-01-01"),
      pd.Timestamp("2024-01-03"),
      pd.Timestamp("2024-01-02"),
    ]
    data = pd.Series([1, 2, 3], index=dates)
    with pytest.raises(ValidationError):
      process(data)


class TestMonoDown:
  """Tests for MonoDown (monotonically decreasing) validator."""

  def test_monodown_strict_fails_on_equal_values(self):
    """Test MonoDown(strict=True) rejects equal consecutive values."""

    @validate
    def process(data: Validated[pd.Series, MonoDown(strict=True)]) -> pd.Series:
      return data

    data = pd.Series([3, 2, 2, 1])
    with pytest.raises(ValidationError, match=r"Not monotonically decreasing \(strict\)"):
      process(data)

  def test_monodown_strict_passes_on_strictly_decreasing_values(self):
    """Test MonoDown(strict=True) passes on strictly decreasing values."""

    @validate
    def process(data: Validated[pd.Series, MonoDown(strict=True)]) -> pd.Series:
      return data

    data = pd.Series([4, 3, 2, 1])
    result = process(data)
    assert result.equals(data)

  def test_valid_series_decreasing(self):
    """Test MonoDown validator with valid decreasing Series."""

    @validate
    def process(
      data: Validated[
        pd.Series,
        MonoDown,
      ],
    ) -> pd.Series:
      return data

    data = pd.Series([5, 4, 3, 2, 1])
    result = process(data)
    assert result.equals(data)

  def test_valid_series_equal(self):
    """Test MonoDown validator allows equal consecutive values."""

    @validate
    def process(
      data: Validated[
        pd.Series,
        MonoDown,
      ],
    ) -> pd.Series:
      return data

    data = pd.Series([5, 4, 4, 3, 3, 2])
    result = process(data)
    assert result.equals(data)

  def test_invalid_series_increasing(self):
    """Test MonoDown validator rejects increasing values."""

    @validate
    def process(
      data: Validated[
        pd.Series,
        MonoDown,
      ],
    ) -> pd.Series:
      return data

    data = pd.Series([5, 4, 3, 4, 1])
    with pytest.raises(ValidationError):
      process(data)


class TestIndexValidator:
  """Test Index validator edge cases."""

  def test_index_with_multiple_validators(self):
    """Test Index validator with multiple validators."""

    @validate
    def process(df: Validated[pd.DataFrame, Index(Datetime, MonoUp)]) -> pd.DataFrame:
      return df

    index = pd.date_range("2023-01-01", periods=5, freq="D")
    df = pd.DataFrame({"a": [1, 2, 3, 4, 5]}, index=index)
    result = process(df)
    assert result.equals(df)
