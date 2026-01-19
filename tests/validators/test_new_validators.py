import pandas as pd

from datawarden.context import ValidationContext
from datawarden.validators import (
  Columns,
  Datetime,
  Empty,
  Gt,
  Index,
  Is,
  MaxDiff,
  MaxGap,
  MonoDown,
  MonoUp,
  NotEmpty,
  NoTimeGaps,
  Outside,
  Positive,
  Rows,
  Unique,
)


def test_outside():
  ctx = ValidationContext(root_data=None)
  v = Outside(0, 10)
  assert v.validate(pd.Series([-1, 11]), ctx).success
  assert not v.validate(pd.Series([5]), ctx).success


def test_columns():
  ctx = ValidationContext(root_data=None)
  v = Columns(["a", "b"], Positive)
  df = pd.DataFrame({"a": [1], "b": [2], "c": [-1]})
  assert v.validate(df, ctx).success

  df2 = pd.DataFrame({"a": [-1], "b": [2]})
  assert not v.validate(df2, ctx).success


def test_datetime_shorthand():
  ctx = ValidationContext(root_data=None)
  v = Datetime()
  s = pd.to_datetime(["2020-01-01"])
  assert v.validate(s, ctx).success
  assert not v.validate(pd.Series([1]), ctx).success


def test_unique_mono():
  ctx = ValidationContext(root_data=None)

  s_unique = pd.Series([1, 2, 3])
  s_dup = pd.Series([1, 2, 1])

  assert Unique().validate(s_unique, ctx).success
  assert not Unique().validate(s_dup, ctx).success

  assert MonoUp().validate(pd.Series([1, 2, 3]), ctx).success
  assert not MonoUp().validate(pd.Series([3, 2, 1]), ctx).success

  assert MonoDown().validate(pd.Series([3, 2, 1]), ctx).success
  assert not MonoDown().validate(pd.Series([1, 2, 3]), ctx).success


def test_is_validator():
  ctx = ValidationContext(root_data=None)
  v = Is(lambda x: x.sum() > 10, name="sum_gt_10")

  df = pd.Series([5, 6])
  assert v.validate(df, ctx).success

  df2 = pd.Series([1, 2])
  assert not v.validate(df2, ctx).success


def test_rows_validator():
  ctx = ValidationContext(root_data=None)
  v = Rows(lambda row: row["a"] + row["b"] == 10)

  df = pd.DataFrame({"a": [1, 5, 9], "b": [9, 5, 1]})
  assert v.validate(df, ctx).success

  df2 = pd.DataFrame({"a": [1, 2], "b": [9, 5]})
  res = v.validate(df2, ctx)
  assert not res.success
  assert not res.mask.iloc[1]
  assert res.mask.iloc[0]


def test_index_validator() -> None:

  ctx = ValidationContext(root_data=None)
  v = Index(Gt(0))

  df = pd.Series([1, 2], index=[1, 2])
  assert v.validate(df, ctx).success

  df2 = pd.Series([1, 2], index=[0, 1])
  assert not v.validate(df2, ctx).success


def test_empty_not_empty():
  ctx = ValidationContext(root_data=None)
  df_empty = pd.DataFrame()
  df_full = pd.DataFrame({"a": [1]})

  assert NotEmpty().validate(df_full, ctx).success
  assert not NotEmpty().validate(df_empty, ctx).success

  assert Empty().validate(df_empty, ctx).success
  assert not Empty().validate(df_full, ctx).success


def test_max_diff():
  ctx = ValidationContext(root_data=None)
  v = MaxDiff(5)

  s = pd.Series([1, 4, 9, 12])
  assert v.validate(s, ctx).success

  s2 = pd.Series([1, 10])
  res = v.validate(s2, ctx)
  assert not res.success
  assert not res.mask.iloc[1]


def test_no_time_gaps():
  ctx = ValidationContext(root_data=None)
  v = NoTimeGaps("1D")

  idx = pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03"])
  assert v.validate(idx, ctx).success

  idx2 = pd.to_datetime(["2020-01-01", "2020-01-03"])
  assert not v.validate(idx2, ctx).success


def test_max_gap():
  ctx = ValidationContext(root_data=None)
  v = MaxGap("1D")

  idx = pd.to_datetime([
    "2020-01-01T00:00:00",
    "2020-01-01T12:00:00",
    "2020-01-02T00:00:00",
  ])
  assert v.validate(idx, ctx).success

  idx2 = pd.to_datetime(["2020-01-01T00:00:00", "2020-01-02T01:00:00"])
  assert not v.validate(idx2, ctx).success


def test_stateful_chunking():
  ctx = ValidationContext(root_data=None)
  v = NoTimeGaps("1D")

  chunk1 = pd.to_datetime(["2020-01-01", "2020-01-02"])
  chunk2 = pd.to_datetime(["2020-01-03", "2020-01-04"])

  assert v.validate(chunk1, ctx).success
  assert v.validate(chunk2, ctx).success

  # Reset context for a failure case
  ctx2 = ValidationContext(root_data=None)
  v2 = NoTimeGaps("1D")
  chunk3 = pd.to_datetime(["2020-01-01", "2020-01-02"])
  chunk4 = pd.to_datetime(["2020-01-04", "2020-01-05"])  # GAP!

  assert v2.validate(chunk3, ctx2).success
  assert not v2.validate(chunk4, ctx2).success
