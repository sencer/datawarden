from typing import Annotated

import pandas as pd
import pytest

from datawarden import (
  Finite,
  IgnoringNaNs,
  MonoUp,
  NonNaN,
  Rows,
  StrictFinite,
  validate,
)


def test_finite_on_string_series():
  """Finite should raise TypeError when applied to non-numeric Series."""
  v = Finite()
  s = pd.Series(["a", "b"])
  with pytest.raises(TypeError, match="numeric"):
    v.validate(s)


def test_strict_finite_on_string_series():
  """StrictFinite should raise TypeError when applied to non-numeric Series."""
  v = StrictFinite()
  s = pd.Series(["a", "b"])
  with pytest.raises(TypeError, match="numeric"):
    v.validate(s)


def test_finite_on_mixed_df():
  """Finite should ignore non-numeric columns in DataFrames."""
  v = Finite()
  df = pd.DataFrame({"a": [1.0, 2.0], "b": ["x", "y"]})
  v.validate(df)


def test_strict_finite_on_mixed_df():
  """StrictFinite should ignore non-numeric columns in DataFrames."""
  v = StrictFinite()
  df = pd.DataFrame({"a": [1.0, 2.0], "b": ["x", "y"]})
  v.validate(df)


def test_ignoring_nans_wrapping_rows():
  """IgnoringNaNs should handle wrapping holistic validators like Rows without crashing."""
  # Rows validator requires a DataFrame.
  # Currently IgnoringNaNs iterates columns and passes Series to wrapped validator.
  v = IgnoringNaNs(Rows(lambda row: row.sum() > 0))
  df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
  # This should pass without TypeError: Rows validator requires a pandas DataFrame
  v.validate(df)


def test_non_nan_on_non_numeric():
  """NonNaN should work correctly on non-numeric data."""
  v = NonNaN()
  s = pd.Series(["a", None, "c"])
  with pytest.raises(ValueError, match="must not contain NaN"):
    v.validate(s)

  s_valid = pd.Series(["a", "b", "c"])
  assert v.validate(s_valid) is None


def test_validator_reset_between_calls():
  """Stateful validators like MonoUp must be reset between decorator calls."""

  @validate
  def sort_check(data: Annotated[pd.Series, MonoUp]):
    return data

  # First call: sets internal state _last_val to 3
  sort_check(pd.Series([1, 2, 3]))

  # Second call: if NOT reset, would compare 1 with 3 and raise ValueError.
  # It should pass because the validator is reset.
  try:
    sort_check(pd.Series([1, 2, 3]))
  except ValueError as e:
    pytest.fail(f"Validator was not reset between calls: {e}")
