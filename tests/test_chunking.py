from typing import Annotated

import pandas as pd
import pytest

from datawarden import Overrides, Validated, validate
from datawarden.exceptions import ValidationError
from datawarden.validators import Gt, Lt
from datawarden.validators.sequence import MonoUp, NoTimeGaps
from datawarden.validators.structural import Column


def test_chunking_execution() -> None:
  df = pd.DataFrame({"a": range(20)})

  @validate
  def process(df: Validated[pd.DataFrame, Gt(-1)]) -> bool:
    del df
    return True

  # Trigger chunking
  with Overrides(chunk_size_rows=5):
    process(df)

  # Trigger failure in a later chunk
  df_fail = pd.DataFrame({"a": range(20)})
  df_fail.iloc[18, 0] = -5  # Fail in last chunk

  with Overrides(chunk_size_rows=5), pytest.raises(ValidationError):
    process(df_fail)


def test_stateful_monoup_chunking() -> None:
  # 1, 2, 3, 4, 5 | 4, 5, 6, 7, 8
  # MonoUp within chunks, but NOT across.
  df = pd.DataFrame({"a": [1, 2, 3, 4, 5, 4, 5, 6, 7, 8]})

  @validate
  def process(df: Validated[pd.DataFrame, Column("a", MonoUp)]) -> bool:
    del df
    return True

  # This should FAIL because 5 -> 4 is not monotonic.
  with Overrides(chunk_size_rows=5), pytest.raises(ValidationError):
    process(df)


def test_chunking_notimegaps() -> None:
  # Create valid hourly data
  dates = pd.date_range(start="2023-01-01", periods=100, freq="h")
  s = pd.Series(dates)

  @validate
  def process_time(s: Annotated[pd.Series, NoTimeGaps("1h")]) -> bool:
    del s
    return True

  with Overrides(chunk_size_rows=24):
    assert process_time(s)

  # Break across boundary
  # Chunk 1: 0-23. Last is 23:00.
  # Chunk 2 starts at 24.
  # Let's make index 24 (chunk 2 start) be 2 hours later than index 23
  dates_broken = list(dates)
  dates_broken[24] = dates_broken[23] + pd.Timedelta("2h")
  s_broken = pd.Series(dates_broken)

  with Overrides(chunk_size_rows=24):
    with pytest.raises(ValidationError) as exc:
      process_time(s_broken)
    # Verify it catches the gap between chunks
    assert "Time gap found between chunks" in str(exc.value)


def test_chunking_stateless_metric() -> None:
  # Ensure simple stateless checks still work with chunking and aggregate correctly

  @validate
  def process_gt(s: Annotated[pd.Series, Gt(0)]) -> bool:
    del s
    return True

  s = pd.Series(range(1, 101))  # All > 0

  with Overrides(chunk_size_rows=10):
    assert process_gt(s)

  s_bad = s.copy()
  s_bad.iloc[55] = -1  # Error in middle chunk

  with Overrides(chunk_size_rows=10), pytest.raises(ValidationError):
    process_gt(s_bad)


def test_chunking_with_parallelism() -> None:
  # Verify that enabling both chunking and parallelism works correctly
  # and actually detects errors.

  @validate
  def process_parallel(
    df: Annotated[pd.DataFrame, Column("a", Gt(0)), Column("b", Lt(10))],
  ) -> bool:
    del df
    return True

  df = pd.DataFrame({
    "a": range(1, 101),  # 1..100 (All > 0)
    "b": range(1, 101),  # 1..100 (Failed after 10)
  })

  # Chunk size 10. Parallel threshold 5.
  # Each chunk (size 10) > threshold (5), so parallel logic triggers.
  with Overrides(chunk_size_rows=10, parallel_threshold_rows=5):
    # Should fail on 'b'
    with pytest.raises(ValidationError) as exc:
      process_parallel(df)
    assert "Column 'b' failed: <10" in str(exc.value)
