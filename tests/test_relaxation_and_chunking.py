import pandas as pd
import pytest

from datawarden import IsNaN, Not, Validated, Validator, validate
from datawarden.config import overrides
from datawarden.validators import (
  AllowInf,
  AllowNaN,
  Ge,
  HasColumn,
  Index,
  MaxGap,
  MonoUp,
  NotNaN,
  Shape,
  StrictFinite,
  Unique,
)


def test_relaxation_allow_nan():
  """Test that AllowNaN overrides global Not(IsNaN)."""

  # Global Not(IsNaN), but Local AllowNaN for column "A"
  @validate
  def process_data(
    df: Validated[pd.DataFrame, NotNaN, HasColumn("A", AllowNaN)],
  ) -> None:
    pass

  df = pd.DataFrame({"A": [1.0, float("nan")], "B": [1.0, 2.0]})
  # Should pass because A allows NaN locally, and B (covered by global Not(IsNaN)) has no NaNs.
  process_data(df)

  df_fail = pd.DataFrame({"A": [1.0, 2.0], "B": [1.0, float("nan")]})
  # Should fail because B has NaNs and no local override
  with pytest.raises(ValueError, match="Data must not contain NaN"):
    process_data(df_fail)


def test_relaxation_allow_inf():
  """Test that AllowInf overrides global StrictFinite."""

  @validate
  def process_data(
    df: Validated[pd.DataFrame, StrictFinite, HasColumn("A", AllowInf)],
  ) -> None:
    pass

  df = pd.DataFrame({"A": [1.0, float("inf")], "B": [1.0, 2.0]})
  # Should pass because A allows Inf locally
  process_data(df)

  df_fail = pd.DataFrame({"A": [1.0, 2.0], "B": [1.0, float("inf")]})
  # Should fail because B has Inf and no local override
  with pytest.raises(ValueError, match="must be finite"):
    process_data(df_fail)


def test_index_chunkability_propagation():
  """Test that Index validator correctly sets is_chunkable."""

  # Ge(0) is chunkable (True by default)
  idx_chunkable = Index(Ge(0))
  assert idx_chunkable.is_chunkable is True

  # MonoUp is now chunkable (stateful)
  idx_chunkable_stateful = Index(MonoUp)
  assert idx_chunkable_stateful.is_chunkable is True

  # Unique is NOT chunkable
  idx_not_chunkable = Index(Unique)
  assert idx_not_chunkable.is_chunkable is False

  # Mixed: Ge(0) and Unique -> Not chunkable
  idx_mixed = Index(Ge(0), Unique)
  assert idx_mixed.is_chunkable is False


def test_chunked_validation():
  """Test that chunked validation works correctly."""

  @validate
  def process(_df: Validated[pd.DataFrame, Not(IsNaN), HasColumn("A", Ge(0))]):
    return True

  df = pd.DataFrame({"A": [1.0, 2.0, 3.0, 4.0], "B": [10.0, 11.0, 12.0, 13.0]})

  # Valid data, small chunks
  with overrides(chunk_size_rows=2):
    assert process(df) is True

  # Invalid data in second chunk
  df_invalid = pd.DataFrame({"A": [1.0, 2.0, -3.0], "B": [1.0, 2.0, 3.0]})
  with (
    overrides(chunk_size_rows=2),
    pytest.raises(ValueError, match=r"(Ge|CompoundValidator)"),
  ):
    process(df_invalid)


def test_mixed_chunkable_non_chunkable():
  """Test that non-chunkable validators still work when chunking is enabled."""

  @validate
  def process(_df: Validated[pd.DataFrame, Shape(4, 2), Not(IsNaN)]):
    return True

  df = pd.DataFrame({"A": [1.0, 2.0, 3.0, 4.0], "B": [10.0, 11.0, 12.0, 13.0]})

  # Valid data, small chunks. Shape(4, 2) should run on full data once.
  with overrides(chunk_size_rows=2):
    assert process(df) is True

  # Invalid shape
  df_wrong_shape = pd.DataFrame({"A": [1.0, 2.0], "B": [10.0, 11.0]})
  with overrides(chunk_size_rows=1), pytest.raises(ValueError, match="rows"):
    process(df_wrong_shape)


def test_stateful_chunked_validation():
  """Test that stateful validators (MonoUp) work correctly with chunking."""

  @validate
  def process(_df: Validated[pd.DataFrame, Index(MonoUp)]):
    return True

  # Valid monotonic index
  df = pd.DataFrame({"A": [1, 2, 3, 4]}, index=[10, 20, 30, 40])
  with overrides(chunk_size_rows=2):
    assert process(df) is True

  # Monotonic within chunks, but NOT across chunks
  # Chunk 1: [10, 20], Chunk 2: [15, 25] -> Broken at 20 -> 15
  df_broken = pd.DataFrame({"A": [1, 2, 3, 4]}, index=[10, 20, 15, 25])
  with (
    overrides(chunk_size_rows=2),
    pytest.raises(ValueError, match="Monotonicity broken"),
  ):
    process(df_broken)

  # Reset works: calling it again with valid data should pass
  with overrides(chunk_size_rows=2):
    assert process(df) is True


def test_stateful_gaps_chunking():
  """Test that stateful gap validators (MaxGap) work correctly with chunking."""

  @validate
  def process(_s: Validated[pd.Series, Index(MaxGap("2min"))]):
    return True

  # Valid 1-min data
  s = pd.Series(
    [1, 2, 3, 4],
    index=pd.to_datetime([
      "2023-01-01 00:00",
      "2023-01-01 00:01",
      "2023-01-01 00:02",
      "2023-01-01 00:03",
    ]),
  )
  with overrides(chunk_size_rows=2):
    assert process(s) is True

  # Gap across chunks: 00:01 to 00:04 (3 min gap)
  s_broken = pd.Series(
    [1, 2, 3, 4],
    index=pd.to_datetime([
      "2023-01-01 00:00",
      "2023-01-01 00:01",
      "2023-01-01 00:04",
      "2023-01-01 00:05",
    ]),
  )
  with (
    overrides(chunk_size_rows=2),
    pytest.raises(ValueError, match="Time gap exceeds maximum"),
  ):
    process(s_broken)


# -----------------------------------------------------------------------------
# Coverage Gap Tests (Fast vs Slow vs Chunkable)
# -----------------------------------------------------------------------------


class FastChunkable(Validator[pd.DataFrame]):
  @property
  def priority(self) -> int:
    return 10

  @property
  def is_chunkable(self) -> bool:
    return True

  def validate(self, data):
    pass


class FastNonChunkable(Validator[pd.DataFrame]):
  @property
  def priority(self) -> int:
    return 10

  @property
  def is_chunkable(self) -> bool:
    return False

  def validate(self, data):
    pass


class SlowChunkable(Validator[pd.DataFrame]):
  @property
  def priority(self) -> int:
    return 100

  @property
  def is_chunkable(self) -> bool:
    return True

  def validate(self, data):
    pass


class SlowNonChunkable(Validator[pd.DataFrame]):
  @property
  def priority(self) -> int:
    return 100

  @property
  def is_chunkable(self) -> bool:
    return False

  def validate(self, data):
    pass


def test_coverage_fast_slow_chunking():
  """Test fast/slow x chunkable/non-chunkable validator execution paths."""

  @validate
  def process(
    df: Validated[
      pd.DataFrame, FastChunkable, FastNonChunkable, SlowChunkable, SlowNonChunkable
    ],
  ):
    pass

  df = pd.DataFrame({"a": range(20)})

  # This will trigger 4 distinct execution paths in decorator.py:
  # 1. Fast loop -> mode="non-chunkable" -> FastNonChunkable runs
  # 2. Fast loop -> mode="chunkable" -> FastChunkable runs
  # 3. Slow loop -> mode="non-chunkable" -> SlowNonChunkable runs
  # 4. Slow loop -> mode="chunkable" -> SlowChunkable runs
  with overrides(chunk_size_rows=5):
    process(df)
