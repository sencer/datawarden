"""Tests for Finite validator."""

import numpy as np
import pandas as pd
import pytest

from datawarden import Validated, validate
from datawarden.exceptions import ValidationError
from datawarden.validators import Finite


class TestFinite:
  """Tests for Finite validator (rejects Inf and NaN)."""

  def test_validate_with_valid_series_passes(self):
    """Test Finite validator with valid Series."""

    @validate
    def process(data: Validated[pd.Series, Finite]) -> pd.Series:
      return data

    data = pd.Series([1.0, 2.0, 3.0])
    result = process(data)
    assert result.equals(data)

  def test_validate_with_valid_dataframe_passes(self):
    """Test Finite validator with valid DataFrame."""

    @validate
    def process(data: Validated[pd.DataFrame, Finite]) -> pd.DataFrame:
      return data

    data = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
    result = process(data)
    assert result.equals(data)

  def test_validate_with_inf_values_raises_error(self):
    """Test Finite validator rejects Inf."""

    @validate
    def process(data: Validated[pd.Series, Finite]) -> pd.Series:
      return data

    data = pd.Series([1.0, np.inf, 3.0])
    with pytest.raises(ValidationError):
      process(data)

  def test_validate_with_neg_inf_values_raises_error(self):
    """Test Finite validator rejects -Inf."""

    @validate
    def process(data: Validated[pd.Series, Finite]) -> pd.Series:
      return data

    data = pd.Series([1.0, -np.inf, 3.0])
    with pytest.raises(ValidationError):
      process(data)

  def test_validate_with_nan_values_raises_error(self):
    """Test Finite validator rejects NaN."""

    @validate
    def process(data: Validated[pd.Series, Finite]) -> pd.Series:
      return data

    data = pd.Series([1.0, np.nan, 3.0])
    with pytest.raises(ValidationError):
      process(data)

  def test_validate_dataframe_with_inf_values_raises_error(self):
    """Test Finite validator rejects DataFrame with Inf."""

    @validate
    def process(data: Validated[pd.DataFrame, Finite]) -> pd.DataFrame:
      return data

    data = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, np.inf]})
    with pytest.raises(ValidationError):
      process(data)

  def test_validate_with_empty_series_passes(self):
    """Test Finite validator with empty Series."""

    @validate
    def process(data: Validated[pd.Series, Finite]) -> pd.Series:
      return data

    data = pd.Series([], dtype=float)
    result = process(data)
    assert result.equals(data)
