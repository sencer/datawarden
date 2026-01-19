"""Tests for column validators: Column, Columns."""

import numpy as np
import pandas as pd
import pytest

from datawarden import Validated, validate
from datawarden.exceptions import ValidationError
from datawarden.validators import Column, Columns, Finite, Positive


class TestColumn:
  """Tests for Column wrapper validator."""

  def test_single_validator(self) -> None:
    """Test Column with single validator."""

    @validate
    def process(df: Validated[pd.DataFrame, Column("a", Finite)]) -> pd.DataFrame:
      return df

    data = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]})
    result = process(data)
    assert result.equals(data)

  def test_single_validator_fails(self) -> None:
    """Test Column validator fails when column violates constraint."""

    @validate
    def process(df: Validated[pd.DataFrame, Column("a", Finite)]) -> pd.DataFrame:
      return df

    data = pd.DataFrame({"a": [1.0, np.inf, 3.0], "b": [4.0, 5.0, 6.0]})
    with pytest.raises(ValidationError):
      process(data)

  def test_multiple_validators(self) -> None:
    """Test Column with multiple validators."""

    @validate
    def process(
      df: Validated[pd.DataFrame, Column("a", Finite, Positive)],
    ) -> pd.DataFrame:
      return df

    data = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]})
    result = process(data)
    assert result.equals(data)

  def test_multiple_validators_fails(self) -> None:
    """Test Column with multiple validators where one fails."""

    @validate
    def process(
      df: Validated[pd.DataFrame, Column("a", Finite, Positive)],
    ) -> pd.DataFrame:
      return df

    data = pd.DataFrame({"a": [1.0, 0.0, 3.0], "b": [4.0, 5.0, 6.0]})
    with pytest.raises(ValidationError):
      process(data)

  def test_missing_column(self) -> None:
    """Test Column with missing column."""

    @validate
    def process(df: Validated[pd.DataFrame, Column("a", Finite)]) -> pd.DataFrame:
      return df

    data = pd.DataFrame({"b": [1.0, 2.0, 3.0]})
    with pytest.raises(ValidationError):
      process(data)

  def test_column_presence_only(self) -> None:
    """Test Column just checks column presence when no validators."""

    @validate
    def process(df: Validated[pd.DataFrame, Column("a")]) -> pd.DataFrame:
      return df

    data = pd.DataFrame({"a": [1.0, np.inf, -5.0], "b": [4.0, 5.0, 6.0]})
    result = process(data)  # Should pass - column exists
    assert result.equals(data)

    # Missing column should fail
    with pytest.raises(ValidationError):
      process(pd.DataFrame({"b": [1.0, 2.0]}))

  def test_column_allows_nan(self) -> None:
    """Test Column allows NaN by default."""

    @validate
    def process(df: Validated[pd.DataFrame, Column("a")]) -> pd.DataFrame:
      return df

    df = pd.DataFrame({"a": [1, np.nan, 3], "b": [4, 5, 6]})
    result = process(df)
    assert result.equals(df)


class TestColumns:
  """Tests for Columns wrapper validator."""

  def test_valid_multiple_columns(self) -> None:
    """Test Columns validator with multiple columns."""

    @validate
    def process(df: Validated[pd.DataFrame, Columns(["a", "b"])]) -> pd.DataFrame:
      return df

    data = pd.DataFrame({"a": [1, 2], "b": [3, 4], "c": [5, 6]})
    result = process(data)
    assert result.equals(data)

  def test_missing_single_column(self) -> None:
    """Test Columns validator with missing column."""

    @validate
    def process(df: Validated[pd.DataFrame, Columns(["b"])]) -> pd.DataFrame:
      return df

    data = pd.DataFrame({"a": [1, 2]})
    with pytest.raises(ValidationError):
      process(data)

  def test_missing_multiple_columns(self) -> None:
    """Test Columns validator with missing columns."""

    @validate
    def process(df: Validated[pd.DataFrame, Columns(["b", "c"])]) -> pd.DataFrame:
      return df

    data = pd.DataFrame({"a": [1, 2]})
    with pytest.raises(ValidationError):
      process(data)

  def test_columns_with_validator(self) -> None:
    """Test Columns applies validators to columns."""

    @validate
    def process(
      df: Validated[pd.DataFrame, Columns(["a", "b"], Positive)],
    ) -> pd.DataFrame:
      return df

    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    result = process(df)
    assert result.equals(df)

  def test_columns_validator_fails(self) -> None:
    """Test Columns fails when column validator fails."""

    @validate
    def process(
      df: Validated[pd.DataFrame, Columns(["a", "b"], Positive)],
    ) -> pd.DataFrame:
      return df

    df = pd.DataFrame({"a": [1, 2, 3], "b": [0, 5, 6]})
    with pytest.raises(ValidationError):
      process(df)
