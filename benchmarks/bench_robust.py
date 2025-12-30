"""Consolidated benchmark script for datawarden performance."""

# pyright: reportCallIssue=false, reportArgumentType=false
import time
import timeit
from typing import Any

import numpy as np
import pandas as pd

from datawarden import (
  Datetime,
  Finite,
  HasColumn,
  Index,
  MonoUp,
  NonNaN,
  Positive,
  Validated,
  validate,
)

# =============================================================================
# Setup Test Data
# =============================================================================
small_data = pd.Series(np.random.rand(100))
large_data = pd.Series(np.random.rand(10000))
datetime_data = pd.Series(
  np.random.rand(1000), index=pd.date_range("2024-01-01", periods=1000)
)

# Dataframes for scaling and parallel benchmarks
df_small = pd.DataFrame({"a": np.random.rand(100), "b": np.random.rand(100)})
df_parallel1 = pd.DataFrame(np.random.uniform(0, 10, size=(500_000, 5)))
df_parallel2 = pd.DataFrame(np.random.uniform(0, 10, size=(500_000, 5)))


# =============================================================================
# Test Functions
# =============================================================================
def plain(data: pd.Series) -> float:
  return data.sum()


@validate
def decorated_simple(data: Validated[pd.Series, Finite]) -> float:
  return data.sum()


@validate
def decorated_multiple(data: Validated[pd.Series, Finite, Positive]) -> float:
  return data.sum()


@validate
def decorated_index(
  data: Validated[pd.Series, Index(Datetime, MonoUp), Finite],
) -> float:
  return data.sum()


@validate
def parallel_func(
  a: Validated[pd.DataFrame, Finite], b: Validated[pd.DataFrame, Finite]
) -> bool:
  _ = a
  _ = b
  return True


# =============================================================================
# Helpers
# =============================================================================
def run_overhead_bench(
  func: Any,
  data: Any,
  skip_val: bool | None,
  iterations: int,
) -> float:
  if skip_val is None:
    return timeit.timeit(lambda: func(data), number=iterations)
  return timeit.timeit(lambda: func(data, skip_validation=skip_val), number=iterations)


def format_us(seconds: float, iterations: int) -> str:
  return f"{(seconds / iterations) * 1_000_000:6.2f} us"


# =============================================================================
# Main Benchmark Logic
# =============================================================================
def main() -> None:
  iterations = 10000
  print("=" * 80)
  print("DataWarden Robust Benchmark".center(80))
  print("=" * 80)

  # 1. Decorator Overhead
  print(f"\n1. Decorator Overhead ({iterations:,} iterations)")
  print("-" * 80)
  header = (
    f"{'Scenario':<40} | {'Baseline':<12} | {'Skip=True':<12} | {'Skip=False':<12}"
  )
  print(header)
  print("-" * 80)

  scenarios = [
    ("Small Series (100 elements)", decorated_simple, small_data),
    ("Large Series (10,000 elements)", decorated_simple, large_data),
    ("Multiple Validators (Finite+Positive)", decorated_multiple, small_data),
    ("Index Validators (Datetime+MonoUp)", decorated_index, datetime_data),
  ]

  for name, func, data in scenarios:
    t_plain = run_overhead_bench(plain, data, None, iterations)
    t_skip = run_overhead_bench(func, data, True, iterations)
    t_val = run_overhead_bench(func, data, False, iterations)

    print(
      f"{name:<40} | "
      f"{format_us(t_plain, iterations):<12} | "
      f"{format_us(t_skip, iterations):<12} | "
      f"{format_us(t_val, iterations):<12}"
    )

  # 2. Parallel Validation
  print("\n2. Parallel Validation (Multiple Large DataFrames)")
  print("-" * 80)
  n_parallel = 10
  # Warmup
  parallel_func(df_parallel1, df_parallel2)
  t_parallel = timeit.timeit(
    lambda: parallel_func(df_parallel1, df_parallel2), number=n_parallel
  )
  print(f"Parallel @validate (2x 500k rows, 5 cols) x {n_parallel}: {t_parallel:.4f}s")
  print(f"Average time per call: {(t_parallel / n_parallel) * 1000:.2f}ms")

  # 3. Data Scaling
  print("\n3. Data Scaling (Global vs Local)")
  print("-" * 80)

  @validate
  def validate_series(data: Validated[pd.Series, Finite, NonNaN]):
    return data.sum()

  @validate
  def validate_df_global(data: Validated[pd.DataFrame, Finite, NonNaN]):
    return data.sum()

  @validate
  def validate_df_local(data: Validated[pd.DataFrame, HasColumn("a", Finite, NonNaN)]):
    return data["a"].sum()

  sizes = [10_000, 100_000, 1_000_000]
  print(
    f"{'Rows':>12} | {'Series (ms)':>15} | {'DF Global (ms)':>18} | {'DF Local (ms)':>18}"
  )
  print("-" * 80)

  for n in sizes:
    s = pd.Series(np.random.randn(n))
    df = pd.DataFrame({"a": np.random.rand(n), "b": np.random.randn(n)})

    # Warmup
    validate_series(s[:100])
    validate_df_global(df[:100])
    validate_df_local(df[:100])

    # Series
    start = time.perf_counter()
    validate_series(s)
    t_series = (time.perf_counter() - start) * 1000

    # DF Global (Promoted)
    start = time.perf_counter()
    validate_df_global(df)
    t_global = (time.perf_counter() - start) * 1000

    # DF Local (Column-wise)
    start = time.perf_counter()
    validate_df_local(df)
    t_local = (time.perf_counter() - start) * 1000

    print(f"{n:12,d} | {t_series:15.2f} | {t_global:18.2f} | {t_local:18.2f}")

  print("\n" + "=" * 80)


if __name__ == "__main__":
  main()
