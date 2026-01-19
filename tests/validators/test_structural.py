import pandas as pd

from datawarden.context import ValidationContext
from datawarden.validators.numeric import Gt
from datawarden.validators.structural import Column, Dtype, IsInstance, Shape

CTX = ValidationContext(root_data=None)


def test_is_instance() -> None:
  v = IsInstance(int)
  assert v.validate(5, CTX).success
  assert not v.validate("5", CTX).success

  v = IsInstance(pd.DataFrame)
  assert v.validate(pd.DataFrame(), CTX).success
  assert not v.validate(pd.Series(), CTX).success


def test_dtype() -> None:
  s_int = pd.Series([1, 2], dtype="int64")
  s_float = pd.Series([1.0, 2.0], dtype="float64")

  v = Dtype("int64")
  assert v.validate(s_int, CTX).success
  assert not v.validate(s_float, CTX).success

  # DataFrame
  df = pd.DataFrame({"a": [1], "b": [2]})
  assert v.validate(df, CTX).success

  df_mixed = pd.DataFrame({"a": [1], "b": [2.0]})
  assert not v.validate(df_mixed, CTX).success


def test_shape() -> None:
  df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})  # (2, 2)

  assert Shape(rows=2).validate(df, CTX).success
  assert not Shape(rows=3).validate(df, CTX).success

  assert Shape(cols=2).validate(df, CTX).success
  assert not Shape(cols=1).validate(df, CTX).success

  assert Shape(rows=2, cols=2).validate(df, CTX).success

  # Series
  s = pd.Series([1, 2])
  assert Shape(rows=2).validate(s, CTX).success
  # Cols on Series -> Fail/Error message
  res = Shape(cols=1).validate(s, CTX)
  assert not res.success
  assert res.message and "1D" in res.message


def test_column() -> None:
  df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})

  # Exists and passes
  v = Column("a", Gt(0))
  res = v.validate(df, CTX)
  assert res.success

  # Exists and fails
  v = Column("a", Gt(5))
  res = v.validate(df, CTX)
  assert not res.success
  assert "Column 'a' failed" in res.message

  # Does not exist
  v = Column("c", Gt(0))
  res = v.validate(df, CTX)
  assert not res.success
  assert "not found" in res.message
