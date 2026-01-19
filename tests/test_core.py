import pandas as pd
import pytest

from datawarden import Validated, validate
from datawarden.exceptions import ValidationError
from datawarden.validators.numeric import Ge, Gt, IsNaN, Positive
from datawarden.validators.structural import Column


def test_basic_validation() -> None:
  @validate
  def process_series(s: Validated[pd.Series, Positive]) -> float:
    return s.sum()

  # Should pass
  process_series(pd.Series([1, 2, 3]))

  # Should fail
  with pytest.raises(ValidationError):
    process_series(pd.Series([-1, 2, 3]))


def test_dataframe_scoping() -> None:
  @validate
  def process_df(
    df: Validated[pd.DataFrame, Ge(5), Column("narrow", Ge(8) | IsNaN)],
  ) -> float:
    return df.sum()

  # Should pass: all >= 5, narrow >= 8
  df1 = pd.DataFrame({"A": [5, 6, 7], "narrow": [8, 9, 10]})
  process_df(df1)

  # Should pass: all >= 5, narrow has NaN
  df2 = pd.DataFrame({"A": [5, 6, 7], "narrow": [8, None, 10]})
  process_df(df2)

  # Should fail: 'A' has 4 (which is < 5)
  df3 = pd.DataFrame({"A": [4, 6, 7], "narrow": [8, 9, 10]})
  with pytest.raises(ValidationError) as excinfo:
    process_df(df3)
  assert "Global check failed" in str(excinfo.value)

  # Should fail: 'narrow' has 7 (which is < 8 and not NaN)
  df4 = pd.DataFrame({"A": [5, 6, 7], "narrow": [7, 9, 10]})
  with pytest.raises(ValidationError) as excinfo:
    process_df(df4)
  assert "Column 'narrow' failed" in str(excinfo.value)


def test_rich_error_message() -> None:
  @validate
  def func(df: Validated[pd.DataFrame, Gt(10)]) -> bool:
    del df
    return True

  df = pd.DataFrame({"a": [5, 15, 5, 20]})
  with pytest.raises(ValidationError) as excinfo:
    func(df)

  msg = str(excinfo.value)
  assert "2/4 rows failed" in msg
  assert ">10" in msg
