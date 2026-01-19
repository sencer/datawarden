import numpy as np
import pandas as pd
import pytest

from datawarden.context import ValidationContext
from datawarden.validators.numeric import (
  Eq,
  Finite,
  Ge,
  Gt,
  Infinite,
  IsNaN,
  Le,
  Lt,
  Ne,
  Negative,
  NonNegative,
  NonPositive,
  NotNaN,
  Positive,
)
from datawarden.validators.value import Between, NotOneOf, OneOf

CTX = ValidationContext(root_data=None)


@pytest.mark.parametrize(
  "validator, data, expected",
  [
    (Gt(0), pd.Series([1, 0, -1, np.nan]), [True, False, False, False]),
    (Ge(0), pd.Series([1, 0, -1, np.nan]), [True, True, False, False]),
    (Lt(0), pd.Series([1, 0, -1, np.nan]), [False, False, True, False]),
    (Le(0), pd.Series([1, 0, -1, np.nan]), [False, True, True, False]),
    (Eq(5), pd.Series([5, 4, 5.0, np.nan]), [True, False, True, False]),
    (Ne(5), pd.Series([5, 4, 5.0, np.nan]), [False, True, False, True]),
    (IsNaN(), pd.Series([1, np.nan, None]), [False, True, True]),
    (NotNaN(), pd.Series([1, np.nan, None]), [True, False, False]),
    (Positive(), pd.Series([1, 0, -1, np.nan]), [True, False, False, False]),
    (NonPositive(), pd.Series([1, 0, -1, np.nan]), [False, True, True, False]),
    (Negative(), pd.Series([1, 0, -1, np.nan]), [False, False, True, False]),
    (NonNegative(), pd.Series([1, 0, -1, np.nan]), [True, True, False, False]),
    (
      Finite(),
      pd.Series([1, np.inf, -np.inf, np.nan]),
      [True, False, False, False],
    ),
    (
      Infinite(),
      pd.Series([1, np.inf, -np.inf, np.nan]),
      [False, True, True, False],
    ),
  ],
)
def test_numeric_validators_series(
  validator: Gt, data: pd.Series, expected: list[bool]
) -> None:
  res = validator.validate(data, CTX)
  mask = res.mask
  if isinstance(mask, pd.Series):
    # Reset index to match expected list
    pd.testing.assert_series_equal(
      mask.reset_index(drop=True), pd.Series(expected), check_names=False
    )
  else:
    assert mask == expected


def test_between() -> None:
  s = pd.Series([0, 5, 10, 15])
  # Checks between five and ten inclusive
  res = Between(5, 10).validate(s, CTX)
  pd.testing.assert_series_equal(res.mask, pd.Series([False, True, True, False]))

  # left
  res = Between(5, 10, inclusive="left").validate(s, CTX)
  pd.testing.assert_series_equal(res.mask, pd.Series([False, True, False, False]))

  # right
  res = Between(5, 10, inclusive="right").validate(s, CTX)
  pd.testing.assert_series_equal(res.mask, pd.Series([False, False, True, False]))

  # neither
  res = Between(5, 10, inclusive="neither").validate(s, CTX)
  pd.testing.assert_series_equal(res.mask, pd.Series([False, False, False, False]))


def test_one_of():
  s = pd.Series([1, 2, 3, np.nan])
  res = OneOf(1, 3).validate(s, CTX)
  pd.testing.assert_series_equal(res.mask, pd.Series([True, False, True, False]))

  # Negation
  res = OneOf(1, 3).negate().validate(s, CTX)
  pd.testing.assert_series_equal(res.mask, pd.Series([False, True, False, True]))


def test_not_one_of():
  s = pd.Series([1, 2, 3, np.nan])
  res = NotOneOf(1, 3).validate(s, CTX)
  pd.testing.assert_series_equal(res.mask, pd.Series([False, True, False, True]))


def test_numeric_validators_dataframe():
  df = pd.DataFrame({"a": [1, -1], "b": [2, -2]})
  v = Gt(0)
  res = v.validate(df, CTX)
  assert not res.success
  pd.testing.assert_frame_equal(
    res.mask, pd.DataFrame({"a": [True, False], "b": [True, False]})
  )
