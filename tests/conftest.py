"""Pytest configuration and fixtures."""

import numpy as np
import pandas as pd

# pyright: reportUnknownMemberType=false
import pytest


@pytest.fixture
def valid_series():
  """Valid pandas Series with finite values."""
  return pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])


@pytest.fixture
def series_with_nan():
  """Series containing NaN values."""
  return pd.Series([1.0, np.nan, 3.0, 4.0, 5.0])


@pytest.fixture
def series_with_inf():
  """Series containing Inf values."""
  return pd.Series([1.0, 2.0, np.inf, 4.0, 5.0])


@pytest.fixture
def series_with_negative():
  """Series containing negative values."""
  return pd.Series([1.0, -2.0, 3.0, 4.0, 5.0])


@pytest.fixture
def valid_dataframe():
  """Valid DataFrame with common OHLC-like structure."""
  return pd.DataFrame({
    "high": [102, 105, 104, 107, 106],
    "low": [100, 103, 101, 104, 103],
    "close": [101, 104, 102, 105, 104],
    "volume": [1000, 1200, 1100, 1300, 1150],
  })


@pytest.fixture
def datetime_series():
  """Series with DatetimeIndex."""
  dates = pd.date_range("2024-01-01", periods=10, freq="1D")
  return pd.Series(range(10), index=dates)


@pytest.fixture
def non_monotonic_series():
  """Series with non-monotonic index."""
  dates = [
    pd.Timestamp("2024-01-01"),
    pd.Timestamp("2024-01-03"),
    pd.Timestamp("2024-01-02"),
  ]
  return pd.Series([1, 2, 3], index=dates)
