"""Automated performance regression check for CI with statistical significance."""

import sys
import timeit

import numpy as np
import pandas as pd

from datawarden import Finite, Validated, validate

# Threshold for skip_validation=True overhead in microseconds
# Typically 0.3 - 0.7us on most systems. 2.0us is a safe upper bound
# to catch significant regressions (e.g. accidentally binding args).
MAX_OVERHEAD_US = 3.0


def plain(data: pd.Series) -> float:
  return data.sum()


@validate
def decorated(data: Validated[pd.Series, Finite]) -> float:
  return data.sum()


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
  iterations = 50_000  # Increased for stability
  repeats = 30
  data = pd.Series(np.random.rand(100))

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
  # Combined margin of error for the difference of means
  combined_ci = np.sqrt(ci_plain**2 + ci_skip**2)

  print("-" * 60)
  print(f"Baseline:         {mean_plain:6.4f} \u00b1 {ci_plain:6.4f} \u00b5s/call")
  print(f"Decorated (skip): {mean_skip:6.4f} \u00b1 {ci_skip:6.4f} \u00b5s/call")
  print(f"Measured Overhead: {overhead:6.4f} \u00b1 {combined_ci:6.4f} \u00b5s/call")
  print("-" * 60)

  # We fail if the lower bound of the overhead estimate still exceeds our threshold.
  # This means we are 95% confident the overhead is at least (overhead - combined_ci).
  stat_overhead_min = overhead - combined_ci

  if stat_overhead_min > MAX_OVERHEAD_US:
    print("ERROR: Statistically significant performance regression detected!")
    print(
      f"Lower bound of overhead ({stat_overhead_min:.2f}\u00b5s) exceeds threshold ({MAX_OVERHEAD_US}\u00b5s)."
    )
    sys.exit(1)
  elif overhead > MAX_OVERHEAD_US:
    print(
      f"WARNING: Mean overhead ({overhead:.2f}\u00b5s) exceeds threshold, but is not statistically significant."
    )
    print(
      f"This might be due to extreme CI noise (Margin: \u00b1{combined_ci:.2f}\u00b5s)."
    )
  else:
    print("Performance check passed!")


if __name__ == "__main__":
  check_regression()
