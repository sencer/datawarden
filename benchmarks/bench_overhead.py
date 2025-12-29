# pyright: reportCallIssue=false, reportArgumentType=false
import timeit

import numpy as np
import pandas as pd

from datawarden import Finite, Validated, validate

# Setup data
data = pd.Series(np.random.randn(1000))
# Large dataframes for parallel benchmark
df_large1 = pd.DataFrame(np.random.uniform(0, 10, size=(500_000, 5)))
df_large2 = pd.DataFrame(np.random.uniform(0, 10, size=(500_000, 5)))


# 1. Raw function
def raw_func(data: pd.Series) -> float:
  return data.sum()


# 2. Decorated function
@validate
def decorated_func(data: "Validated[pd.Series, Finite]") -> float:
  return data.sum()


# 3. Parallel validation function
@validate
def parallel_func(
  a: "Validated[pd.DataFrame, Finite]", b: "Validated[pd.DataFrame, Finite]"
) -> bool:
  _ = a
  _ = b
  return True


# 4. Sequential manual validation for comparison
def manual_func(a: pd.DataFrame, b: pd.DataFrame) -> bool:
  # Emulate Finite check
  has_nan = a.isna().any(axis=None) or b.isna().any(axis=None)
  if has_nan:
    return False

  return not (
    np.any(np.isinf(a.select_dtypes(include=[np.number]).values))
    or np.any(np.isinf(b.select_dtypes(include=[np.number]).values))
  )


def run_benchmarks() -> None:
  n = 10000

  print("--- Single Argument Overhead ---")
  t_raw = timeit.timeit(lambda: raw_func(data), number=n)
  t_dec = timeit.timeit(lambda: decorated_func(data), number=n)
  t_skip = timeit.timeit(lambda: decorated_func(data, skip_validation=True), number=n)

  print(f"Raw function: {t_raw:.4f}s")
  print(f"Decorated (validate=True): {t_dec:.4f}s")
  print(f"Decorated (skip_validation=True): {t_skip:.4f}s")

  overhead = (t_dec - t_raw) / t_raw * 100
  print(f"Overhead (validate=True): {overhead:.1f}%")

  overhead_skip = (t_skip - t_raw) / t_raw * 100
  print(f"Overhead (skip_validation=True): {overhead_skip:.1f}%")

  print("\n--- Parallel Validation (Large DataFrames) ---")
  n_parallel = 10

  # Warmup
  parallel_func(df_large1, df_large2)
  manual_func(df_large1, df_large2)

  t_parallel = timeit.timeit(
    lambda: parallel_func(df_large1, df_large2), number=n_parallel
  )
  t_manual = timeit.timeit(lambda: manual_func(df_large1, df_large2), number=n_parallel)

  print(f"Parallel @validate (n={n_parallel}): {t_parallel:.4f}s")
  print(f"Sequential Manual (n={n_parallel}): {t_manual:.4f}s")

  if t_manual > 0:
    speedup = (t_manual - t_parallel) / t_manual * 100
    print(f"Parallel Speedup: {speedup:.1f}%")


if __name__ == "__main__":
  run_benchmarks()
