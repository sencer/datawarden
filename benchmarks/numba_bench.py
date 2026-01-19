from __future__ import annotations

from typing import TYPE_CHECKING, Any

from loguru import logger
import numpy as np
import pandas as pd

from benchmarks.utils import compare_benchmarks
from datawarden import Ge, Index, IsNaN, Le, MonoUp, Overrides
from datawarden.context import ValidationContext

if TYPE_CHECKING:
  from collections.abc import Callable

  from datawarden.validators.base import BaseValidator


def _validate(
  validator: BaseValidator[Any],
  data: pd.Series | pd.DataFrame | np.ndarray[Any, Any],
  use_numba: bool,
) -> None:
  with Overrides(use_numba=use_numba, fail_fast=True):
    ctx = ValidationContext(root_data=data)
    validator.validate(data, ctx)


class DataProvider:
  def __init__(self, factory: Callable[[], Any], count: int = 200) -> None:
    self.data = [factory() for _ in range(count)]
    self.idx = 0

  def next(self) -> Any:  # noqa: ANN401
    if self.idx < len(self.data):
      res = self.data[self.idx]
      self.idx += 1
      return res
    # Fallback if we run out (shouldn't happen with correct config)
    return self.data[0]  # Reuse to avoid crash, though cache might affect result


def run_numba_benchmarks() -> None:  # noqa: PLR0914
  logger.info("=" * 60)
  logger.info("NUMBA SPECIFIC BENCHMARKS")
  logger.info("=" * 60)

  # Shared data arrays to keep memory usage low while creating distinct objects
  n_rows = 10_000_000
  range_data_arr = np.arange(n_rows)  # Shared backing for RangeIndex case
  idx_arr_float = np.arange(n_rows, dtype=np.float64)  # Shared backing for FloatIndex
  col_arr_float = np.random.rand(n_rows)  # Shared backing for FloatIndex
  complex_arr = np.random.uniform(-5, 150, n_rows)  # Shared backing for Complex

  # Configuration
  iterations = 5
  repeats = 10
  warmup = 2
  total_needed = (repeats + warmup) * iterations + 20

  # 1. Index Optimization (RangeIndex)
  def make_range_df() -> pd.DataFrame:
    # Distinct DataFrame, shared data
    return pd.DataFrame({"a": range_data_arr}, copy=False)

  provider_range_base = DataProvider(make_range_df, count=total_needed)
  provider_range_opt = DataProvider(make_range_df, count=total_needed)
  v_index = Index(MonoUp)

  compare_benchmarks(
    f"Index(MonoUp) on RangeIndex (N={n_rows:,})",
    base_func=lambda: _validate(v_index, provider_range_base.next(), False),
    opt_func=lambda: _validate(v_index, provider_range_opt.next(), True),
    iterations=iterations,
    repeats=repeats,
    warmup=warmup,
  )

  # 2. Index Optimization (Standard Index)
  def make_float_df() -> pd.DataFrame:
    # Distinct DataFrame & Index, shared data
    df = pd.DataFrame({"a": col_arr_float}, copy=False)
    df.index = pd.Index(idx_arr_float, copy=False)
    return df

  provider_float_base = DataProvider(make_float_df, count=total_needed)
  provider_float_opt = DataProvider(make_float_df, count=total_needed)

  compare_benchmarks(
    f"Index(MonoUp) on FloatIndex (N={n_rows:,})",
    base_func=lambda: _validate(v_index, provider_float_base.next(), False),
    opt_func=lambda: _validate(v_index, provider_float_opt.next(), True),
    iterations=iterations,
    repeats=repeats,
    warmup=warmup,
  )

  # 2.5 Index + Condition
  v_index_cond = Index(MonoUp) & Ge(0)
  provider_cond_base = DataProvider(make_float_df, count=total_needed)
  provider_cond_opt = DataProvider(make_float_df, count=total_needed)

  compare_benchmarks(
    f"Index(MonoUp) & Ge(0) (N={n_rows:,})",
    base_func=lambda: _validate(v_index_cond, provider_cond_base.next(), False),
    opt_func=lambda: _validate(v_index_cond, provider_cond_opt.next(), True),
    iterations=iterations,
    repeats=repeats,
    warmup=warmup,
  )

  # 3. Complex Logic Fusion
  v_complex = ((Ge(0) & Le(10)) | (Ge(20) & Le(30)) | Ge(100)) & ~IsNaN()

  def make_complex_series() -> pd.Series[Any]:
    # Distinct Series, shared data
    return pd.Series(complex_arr, copy=False)

  provider_complex_base = DataProvider(make_complex_series, count=total_needed)
  provider_complex_opt = DataProvider(make_complex_series, count=total_needed)

  compare_benchmarks(
    f"Complex Logic Fusion (N={n_rows:,})",
    base_func=lambda: _validate(v_complex, provider_complex_base.next(), False),
    opt_func=lambda: _validate(v_complex, provider_complex_opt.next(), True),
    iterations=iterations,
    repeats=repeats,
    warmup=warmup,
  )


if __name__ == "__main__":
  run_numba_benchmarks()
