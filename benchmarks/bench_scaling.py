import logging
import time

import numpy as np
import pandas as pd

from datawarden import Finite, Ge, HasColumn, NonNaN, Validated, validate

# Disable logging for benchmarks
logging.getLogger("datawarden").setLevel(logging.ERROR)


def benchmark_scaling():
  sizes = [10_000, 100_000, 1_000_000]

  @validate
  def validate_series(data: Validated[pd.Series, Finite, NonNaN]):
    return data.sum()

  @validate
  def validate_df(data: Validated[pd.DataFrame, HasColumn("a", Finite, NonNaN, Ge(0))]):
    return data["a"].sum()

  print(f"{'Rows':>12} | {'Series (ms)':>15} | {'DataFrame (ms)':>18}")
  print("-" * 50)

  for n in sizes:
    # Prepare data
    s = pd.Series(np.random.randn(n))
    df = pd.DataFrame({"a": np.random.rand(n), "b": np.random.randn(n)})

    # Warmup
    validate_series(s[:100])
    validate_df(df[:100])

    # Benchmark Series
    start = time.perf_counter()
    validate_series(s)
    series_time = (time.perf_counter() - start) * 1000

    # Benchmark DataFrame
    start = time.perf_counter()
    validate_df(df)
    df_time = (time.perf_counter() - start) * 1000

    print(f"{n:12,d} | {series_time:15.2f} | {df_time:18.2f}")


if __name__ == "__main__":
  benchmark_scaling()
