from typing import Annotated
from unittest.mock import patch

import pandas as pd
import pytest

from datawarden import validate
from datawarden.config import get_config
from datawarden.validators import Ge, HasColumn, HasColumns


def test_warn_only_type_mismatch():
  with patch("datawarden.decorator.logger") as mock_logger:

    @validate(warn_only_by_default=True)
    def func(x: Annotated[int, Ge(0)]):
      return x

    assert func("not an int") is None
    assert mock_logger.error.called
    args, _ = mock_logger.error.call_args
    assert "Type mismatch for parameter 'x'" in args[0]


def test_warn_only_validator_failure():
  with patch("datawarden.decorator.logger") as mock_logger:

    @validate(warn_only_by_default=True)
    def func(x: Annotated[int, Ge(10)]):
      return x

    assert func(5) is None
    assert mock_logger.error.called
    args, _ = mock_logger.error.call_args
    assert "Validation failed for parameter 'x'" in args[0]


def test_warn_only_pandas_missing_columns():
  with patch("datawarden.decorator.logger") as mock_logger:

    @validate(warn_only_by_default=True)
    def func(df: Annotated[pd.DataFrame, HasColumns(["a", "b"])]):
      return df

    df = pd.DataFrame({"a": [1]})
    assert func(df) is None
    assert mock_logger.error.called
    args, _ = mock_logger.error.call_args
    assert "Missing columns: ['b']" in args[0]


def test_warn_only_pandas_column_failure():
  with patch("datawarden.decorator.logger") as mock_logger:

    @validate(warn_only_by_default=True)
    def func(df: Annotated[pd.DataFrame, HasColumn("a", Ge(10))]):
      return df

    df = pd.DataFrame({"a": [5]})
    assert func(df) is None
    assert mock_logger.error.called
    args, _ = mock_logger.error.call_args
    assert "Validation failed for parameter 'df'" in args[0]


def test_estimate_data_size_non_pandas():
  # This is internal but testing it indirectly via validation logic or directly if possible.
  # The decorator uses it.
  @validate
  def func(x: list):
    return x

  # Should not crash
  assert func([1, 2, 3]) == [1, 2, 3]


def test_parallel_execution():
  # Force parallel execution
  config = get_config()
  orig_threshold = config.parallel_threshold_rows
  config.parallel_threshold_rows = 1  # Low threshold

  try:

    @validate
    def func(df1: Annotated[pd.DataFrame, Ge(0)], df2: Annotated[pd.DataFrame, Ge(0)]):
      _ = (df1, df2)
      return True

    df = pd.DataFrame({"a": [1, 2, 3]})
    # We need >1 items to validate to trigger use_parallel
    assert func(df, df) is True
  finally:
    config.parallel_threshold_rows = orig_threshold


def test_parallel_execution_failure_warn_only():
  config = get_config()
  orig_threshold = config.parallel_threshold_rows
  config.parallel_threshold_rows = 1

  try:
    # We need to mock logger here too if we want to suppress output or check it
    # But for this test, just ensuring it returns None is enough?
    # The previous attempt failed because I commented out the assertion? No.
    # I didn't fail this one.

    @validate(warn_only_by_default=True)
    def func(df1: Annotated[pd.DataFrame, Ge(10)], df2: Annotated[pd.DataFrame, Ge(0)]):
      _ = (df1, df2)
      return True

    df_fail = pd.DataFrame({"a": [1, 2, 3]})  # Fail
    df_pass = pd.DataFrame({"a": [1, 2, 3]})  # Pass

    assert func(df_fail, df_pass) is None
  finally:
    config.parallel_threshold_rows = orig_threshold


def test_chunking_execution():
  config = get_config()
  orig_chunk = config.chunk_size_rows
  config.chunk_size_rows = 2

  try:
    # Ge is chunkable.
    @validate
    def func(df: Annotated[pd.DataFrame, Ge(0)]):
      _ = df
      return True

    df = pd.DataFrame({"a": [1, 2, 3, 4, 5]})
    assert func(df) is True
  finally:
    config.chunk_size_rows = orig_chunk


def test_var_args_kwargs_error():
  # To hit the fast path error "unexpected keyword argument", I need a function WITHOUT **kwargs.
  @validate
  def func_strict(a: int):
    pass

  # Pass 'b' which is not in signature.
  # Standard python would raise TypeError.
  # Decorator checks BEFORE calling function (in fast path).
  with pytest.raises(TypeError, match="unexpected keyword argument"):
    func_strict(1, b=2)

  # To hit "multiple values for argument"
  with pytest.raises(TypeError, match="multiple values for argument"):
    func_strict(1, a=2)


def test_type_mismatch_raises():
  @validate
  def func(x: Annotated[int, Ge(0)]):
    return x

  with pytest.raises(TypeError, match="Type mismatch"):
    func("s")


def test_validator_failure_raises():
  @validate
  def func(x: Annotated[int, Ge(0)]):
    return x

  with pytest.raises(ValueError, match="Data must be >= 0"):
    func(-1)


def test_pandas_column_failure_raises():
  @validate
  def func(df: Annotated[pd.DataFrame, HasColumn("a", Ge(0))]):
    _ = df
    return True

  with pytest.raises(ValueError, match="Data must be >= 0"):
    func(pd.DataFrame({"a": [-1]}))
