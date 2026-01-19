"""CI performance regression gatekeeper for datawarden."""

import sys
import timeit

from loguru import logger
import numpy as np
import pandas as pd

from datawarden import Finite, Overrides, Validated, validate

# Threshold for CI gatekeeper (microseconds)
MAX_OVERHEAD_US = 3.0


def get_stats(times: list[float], iterations: int) -> tuple[float, float]:
  us_per_call = (np.array(times) / iterations) * 1_000_000
  mean = np.mean(us_per_call)
  std_dev = np.std(us_per_call, ddof=1)
  sem = std_dev / np.sqrt(len(times))
  margin_of_error = 1.96 * sem
  return float(mean), float(margin_of_error)


def check_regression() -> None:
  logger.info("-" * 60)
  logger.info("CI REGRESSION GATEKEEPER".center(60))
  logger.info("-" * 60)

  iterations = 50_000
  repeats = 30
  data = pd.Series(np.random.rand(100))

  def plain(data: pd.Series) -> float:
    return data.sum()

  @validate
  def decorated(data: Validated[pd.Series, Finite]) -> float:
    return data.sum()

  # Warmup
  timeit.timeit(lambda: plain(data), number=1000)
  with Overrides(skip_validation=True):
    timeit.timeit(lambda: decorated(data), number=1000)

  # Benchmark
  t_plain_list = timeit.repeat(lambda: plain(data), number=iterations, repeat=repeats)
  with Overrides(skip_validation=True):
    t_skip_list = timeit.repeat(
      lambda: decorated(data), number=iterations, repeat=repeats
    )

  mean_plain, ci_plain = get_stats(t_plain_list, iterations)
  mean_skip, ci_skip = get_stats(t_skip_list, iterations)

  overhead = mean_skip - mean_plain
  combined_ci = np.sqrt(ci_plain**2 + ci_skip**2)

  logger.info(
    f"Baseline:         {mean_plain:6.4f} \u00b1 {ci_plain:6.4f} \u03bc s/call"
  )
  logger.info(f"Decorated (skip): {mean_skip:6.4f} \u00b1 {ci_skip:6.4f} \u03bc s/call")
  logger.info(
    f"Measured Overhead: {overhead:6.4f} \u00b1 {combined_ci:6.4f} \u03bc s/call"
  )
  logger.info("-" * 60)

  stat_overhead_min = overhead - combined_ci

  if stat_overhead_min > MAX_OVERHEAD_US:
    logger.error(
      f"ERROR: Performance regression detected! Min overhead {stat_overhead_min:.2f}\u03bcs > {MAX_OVERHEAD_US}\u03bcs"
    )
    sys.exit(1)
  else:
    logger.success("Performance check passed!")


if __name__ == "__main__":
  check_regression()
