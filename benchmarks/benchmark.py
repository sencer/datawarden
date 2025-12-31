"""Robust micro-benchmark suite and CI regression gatekeeper for datawarden.

Usage:
  python benchmarks/benchmark.py          # Run full suite
  python benchmarks/benchmark.py --ci     # Run only regression check for CI
"""

# ruff: noqa: PLR2004, RUF001, ARG001, F841
# pyright: reportCallIssue=false, reportArgumentType=false

import argparse
from collections.abc import Callable
import statistics
import sys
import time
import timeit
from typing import Any

import numpy as np
import pandas as pd

from datawarden import (
  Datetime,
  Finite,
  Ge,
  HasColumn,
  IgnoringNaNs,
  Index,
  MonoUp,
  Negative,
  Not,
  NotNaN,
  Positive,
  Validated,
  validate,
)

# --- Configuration ---
TRIALS = 10
ITERATIONS = 5_000
MAX_OVERHEAD_US = 3.0  # Threshold for CI gatekeeper


def time_ns(func: Callable[[], Any], iterations: int) -> float:
  """Run iterations and return total time in nanoseconds."""
  start = time.perf_counter_ns()
  for _ in range(iterations):
    func()
  return time.perf_counter_ns() - start


def benchmark(
  name: str,
  func: Callable[[], Any],
  trials: int = TRIALS,
  iterations: int = ITERATIONS,
) -> tuple[float, float]:
  """Run benchmark with warmup, return (min_ns_per_call, stdev_ns_per_call)."""
  # Warmup
  time_ns(func, iterations // 5)

  results_ns = []
  for _ in range(trials):
    total_ns = time_ns(func, iterations)
    results_ns.append(total_ns / iterations)
    print(".", end="", flush=True)

  min_ns = min(results_ns)
  stdev_ns = statistics.stdev(results_ns) if len(results_ns) > 1 else 0.0

  # Format appropriately (ns, us, or ms)
  if min_ns >= 1_000_000:
    min_fmt = f"{min_ns / 1_000_000:7.2f}ms"
    std_fmt = f"{stdev_ns / 1_000_000:5.2f}ms"
  elif min_ns >= 1_000:
    min_fmt = f"{min_ns / 1_000:7.2f}µs"
    std_fmt = f"{stdev_ns / 1_000:5.2f}µs"
  else:
    min_fmt = f"{min_ns:7.1f}ns"
    std_fmt = f"{stdev_ns:5.1f}ns"

  print(f"\r{name:<45} | Min: {min_fmt} | Stdev: {std_fmt}")

  return min_ns, stdev_ns


def compare(
  name: str,
  baseline: Callable[[], Any],
  optimized: Callable[[], Any],
  trials: int = TRIALS,
  iterations: int = ITERATIONS,
) -> None:
  """Compare two implementations side-by-side, report speedup."""
  # Interleaved warmup
  time_ns(baseline, iterations // 5)
  time_ns(optimized, iterations // 5)

  baseline_results = []
  optimized_results = []

  # Interleave trials to reduce systematic bias
  for _ in range(trials):
    baseline_results.append(time_ns(baseline, iterations) / iterations)
    optimized_results.append(time_ns(optimized, iterations) / iterations)
    print(".", end="", flush=True)

  base_min = min(baseline_results)
  opt_min = min(optimized_results)
  speedup = (base_min - opt_min) / base_min * 100 if base_min > 0 else 0

  indicator = "✓" if speedup > 5 else ("✗" if speedup < -5 else "=")

  # Format appropriately
  if base_min >= 1_000_000:
    base_fmt = f"{base_min / 1_000_000:6.2f}ms"
    opt_fmt = f"{opt_min / 1_000_000:6.2f}ms"
  elif base_min >= 1_000:
    base_fmt = f"{base_min / 1_000:6.2f}µs"
    opt_fmt = f"{opt_min / 1_000:6.2f}µs"
  else:
    base_fmt = f"{base_min:6.1f}ns"
    opt_fmt = f"{opt_min:6.1f}ns"

  print(
    f"\r{name:<45} | Base: {base_fmt} | Opt: {opt_fmt} | {speedup:+5.1f}% {indicator}"
  )


def get_stats(times, iterations):
  # Convert total times to microseconds per call
  us_per_call = (np.array(times) / iterations) * 1_000_000
  mean = np.mean(us_per_call)
  std_dev = np.std(us_per_call, ddof=1)
  sem = std_dev / np.sqrt(len(times))
  # 95% Confidence Interval (approx 1.96 for n=30)
  margin_of_error = 1.96 * sem
  return mean, margin_of_error


def check_regression():
  """Automated performance regression check for CI."""
  print("-" * 85)
  print("CI REGRESSION GATEKEEPER".center(85))
  print("-" * 85)

  iterations = 50_000
  repeats = 30
  data = pd.Series(np.random.rand(100))

  def plain(data: pd.Series) -> float:
    return data.sum()

  @validate
  def decorated(data: Validated[pd.Series, Finite]) -> float:
    return data.sum()

  print(f"Running benchmarks ({repeats} repeats, {iterations} iterations each)...")

  # Warmup
  timeit.timeit(lambda: plain(data), number=1000)
  timeit.timeit(lambda: decorated(data, skip_validation=True), number=1000)

  # Benchmark
  t_plain_list = timeit.repeat(lambda: plain(data), number=iterations, repeat=repeats)
  t_skip_list = timeit.repeat(
    lambda: decorated(data, skip_validation=True), number=iterations, repeat=repeats
  )

  mean_plain, ci_plain = get_stats(t_plain_list, iterations)
  mean_skip, ci_skip = get_stats(t_skip_list, iterations)

  overhead = mean_skip - mean_plain
  combined_ci = np.sqrt(ci_plain**2 + ci_skip**2)

  print("-" * 60)
  print(f"Baseline:         {mean_plain:6.4f}   {ci_plain:6.4f}  /call")
  print(f"Decorated (skip): {mean_skip:6.4f}   {ci_skip:6.4f}  /call")
  print(f"Measured Overhead: {overhead:6.4f}   {combined_ci:6.4f}  /call")
  print("-" * 60)

  stat_overhead_min = overhead - combined_ci

  if stat_overhead_min > MAX_OVERHEAD_US:
    print("ERROR: Statistically significant performance regression detected!")
    print(
      f"Lower bound of overhead ({stat_overhead_min:.2f} ) exceeds threshold ({MAX_OVERHEAD_US} )."
    )
    sys.exit(1)
  elif overhead > MAX_OVERHEAD_US:
    print(
      f"WARNING: Mean overhead ({overhead:.2f} ) exceeds threshold, but is not statistically significant."
    )
    print(f"This might be due to extreme CI noise (Margin:  {combined_ci:.2f} ).")
  else:
    print("Performance check passed!")


def run_all_benchmarks():
  print("=" * 85)
  print("DATAWARDEN MICRO-BENCHMARK SUITE".center(85))
  print(f"Method: MIN of {TRIALS} trials × {ITERATIONS:,} iterations".center(85))
  print("=" * 85)

  # =============================================================================
  # Setup Test Data
  # =============================================================================
  small_data = pd.Series(np.random.rand(100))
  large_data = pd.Series(np.random.rand(10_000))
  datetime_data = pd.Series(
    np.random.rand(1000), index=pd.date_range("2024-01-01", periods=1000)
  )

  # DataFrames for various tests
  df_small = pd.DataFrame({"a": np.random.rand(100), "b": np.random.rand(100)})
  df_medium = pd.DataFrame({"a": np.random.rand(100_000), "b": np.random.rand(100_000)})
  df_large = pd.DataFrame({
    f"c{i}": np.random.uniform(0, 10, 500_000) for i in range(5)
  })

  # =============================================================================
  # Define Test Functions
  # =============================================================================
  def plain_func(data: pd.Series) -> float:
    return data.sum()

  @validate
  def validated_simple(data: Validated[pd.Series, Finite]) -> float:
    return data.sum()

  @validate
  def validated_multiple(data: Validated[pd.Series, Finite, Positive]) -> float:
    return data.sum()

  @validate
  def validated_index(
    data: Validated[pd.Series, Index(Datetime, MonoUp), Finite],
  ) -> float:
    return data.sum()

  @validate
  def validated_df_global(data: Validated[pd.DataFrame, Finite, NotNaN]):
    return True

  @validate
  def validated_df_local(
    data: Validated[pd.DataFrame, Finite, NotNaN, HasColumn("a", Ge(0))],
  ):
    return True

  @validate
  def parallel_func(
    a: Validated[pd.DataFrame, Finite], b: Validated[pd.DataFrame, Finite]
  ) -> bool:
    return True

  @validate
  def validated_not_negative(data: Validated[pd.Series, Not(Negative)]) -> float:
    return data.sum()

  @validate
  def validated_ignoring_nans(
    data: Validated[pd.Series, IgnoringNaNs(Positive)],
  ) -> float:
    return data.sum()

  @validate
  def validated_not_positive(data: Validated[pd.Series, Not(Positive)]) -> float:
    return data.sum()

  # Data with NaN for IgnoringNaNs tests
  data_with_nan = pd.Series(np.random.rand(1000))
  data_with_nan.iloc[::10] = np.nan  # Every 10th element is NaN

  # Data for Not(Negative) - all positive
  data_positive = pd.Series(np.abs(np.random.rand(1000)) + 0.1)

  # Data for Not(Positive) - all non-positive
  data_non_positive = pd.Series(-np.abs(np.random.rand(1000)))

  # --- SECTION 1: DECORATOR OVERHEAD ---
  print("\n[1/6] DECORATOR OVERHEAD (Small Data)")

  benchmark("Plain Function Call", lambda: plain_func(small_data))
  benchmark(
    "@validate (Skip=True)", lambda: validated_simple(small_data, skip_validation=True)
  )
  benchmark(
    "@validate (Skip=False)",
    lambda: validated_simple(small_data, skip_validation=False),
  )

  # --- SECTION 2: VALIDATOR STACKING ---
  print("\n[2/6] VALIDATOR STACKING")

  benchmark("Single Validator (Finite)", lambda: validated_simple(small_data))
  benchmark(
    "Multiple Validators (Finite+Positive)", lambda: validated_multiple(small_data)
  )
  benchmark("Index + Data Validators", lambda: validated_index(datetime_data))

  # --- SECTION 3: DATA SIZE SCALING ---
  print("\n[3/6] DATA SIZE SCALING (Series)")

  benchmark("Series: 100 rows", lambda: validated_simple(small_data))
  benchmark("Series: 10k rows", lambda: validated_simple(large_data))

  # --- SECTION 4: DATAFRAME PATTERNS ---
  print("\n[4/6] DATAFRAME PATTERNS")

  benchmark("DF Global (100k×2, all numeric)", lambda: validated_df_global(df_medium))
  benchmark("DF Local (100k×2, column override)", lambda: validated_df_local(df_medium))

  # For parallel, use fewer iterations (heavy operation)
  benchmark(
    "Parallel (2× 500k×5)",
    lambda: parallel_func(df_large, df_large),
    iterations=50,
  )

  # --- SECTION 5: NOT & IGNORINGNANS WRAPPERS ---
  print("\n[5/6] NOT & IGNORINGNANS WRAPPERS (1k rows)")

  benchmark(
    "Not(Negative) on positive data", lambda: validated_not_negative(data_positive)
  )
  benchmark(
    "Not(Positive) on non-positive data",
    lambda: validated_not_positive(data_non_positive),
  )
  benchmark(
    "IgnoringNaNs(Positive) with 10% NaN",
    lambda: validated_ignoring_nans(data_with_nan),
  )

  compare(
    "Positive vs Not(Negative)",
    lambda: validated_multiple(data_positive),
    lambda: validated_not_negative(data_positive),
  )

  # --- SECTION 6: OVERHEAD COMPARISON ---
  print("\n[6/6] OVERHEAD MEASUREMENTS (A vs B)")

  compare(
    "Decorator overhead: Raw → @validate",
    lambda: plain_func(small_data),
    lambda: validated_simple(small_data),
  )

  compare(
    "Skip flag overhead: Skip=True → False",
    lambda: validated_simple(small_data, skip_validation=True),
    lambda: validated_simple(small_data, skip_validation=False),
  )

  compare(
    "Data scaling: Series 100 → 10k rows",
    lambda: validated_simple(small_data),
    lambda: validated_simple(large_data),
  )

  compare(
    "DF pattern: Global → Local override",
    lambda: validated_df_global(df_medium),
    lambda: validated_df_local(df_medium),
  )

  print("\n" + "=" * 85)
  print(
    "Overhead = (B - A) / A × 100%  |  ✓ B faster  |  = similar  |  ✗ B slower".center(
      85
    )
  )
  print("=" * 85)


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="datawarden benchmark suite")
  parser.add_argument("--ci", action="store_true", help="Run only CI regression check")
  args = parser.parse_args()

  if args.ci:
    check_regression()
  else:
    run_all_benchmarks()
    print("\n")
    check_regression()
