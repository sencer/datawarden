from __future__ import annotations

import timeit
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
  from collections.abc import Callable

from loguru import logger
import numpy as np
import pandas as pd

from datawarden import Ge, Index, MonoUp, Overrides
from datawarden.context import ValidationContext
from datawarden.validators.sequence import MonoUp as _MonoUp

# 10M rows
N = 10_000_000

# Global Arrays (Read-only)
# Monotonic increasing data
DATA_ARR_MONO = np.arange(N, dtype=np.float64)
# Float Index array (monotonic)
INDEX_ARR_FLOAT = np.arange(N, dtype=np.float64)


# Define Fusable MonoUp by subclassing and overriding properties
class FusableMonoUp(_MonoUp):
  @property
  def numba_supported(self) -> bool:
    return True

  @property
  def numba_fusable(self) -> bool:
    return True

  def __str__(self) -> str:
    return "MonoUp(Fused)"


fusable_monoup = FusableMonoUp()


def bench(
  name: str,
  validator_factory: Callable[[], Any],
  data_factory: Callable[[], Any],
  iterations: int = 5,
  force_numba: bool = True,
) -> float:
  times = []
  # Warmup once
  data = data_factory()
  ctx = ValidationContext(root_data=data)
  validator = validator_factory()
  with Overrides(use_numba=force_numba):
    validator.validate(data, ctx)

  for _ in range(iterations):
    # Setup data (destroy cache)
    data = data_factory()
    ctx = ValidationContext(root_data=data)
    validator = validator_factory()

    start = timeit.default_timer()
    with Overrides(use_numba=force_numba):
      validator.validate(data, ctx)
    end = timeit.default_timer()
    times.append(end - start)

  min_t = np.min(times)
  logger.info(f"  {name:<40} Min: {min_t * 1000:.2f} ms")
  return float(min_t)


# Data Factories
def make_range_series() -> pd.Series[Any]:
  # Fresh Series, default RangeIndex
  return pd.Series(DATA_ARR_MONO)


def make_float_index_series() -> pd.Series[Any]:
  # Recreate Index object to clear cache
  # copy=False to be fast
  idx = pd.Index(INDEX_ARR_FLOAT, copy=False)
  return pd.Series(DATA_ARR_MONO, index=idx, copy=False)


# Validators mapping
validators_to_test = [
  # 1. MonoUp (Values)
  ("MonoUp", MonoUp, lambda: fusable_monoup),
  # 2. MonoUp & Ge(0)
  ("MonoUp & Ge(0)", lambda: MonoUp() & Ge(0), lambda: fusable_monoup & Ge(0)),
  # 3. MonoUp | Ge(0)
  ("MonoUp | Ge(0)", lambda: MonoUp() | Ge(0), lambda: fusable_monoup | Ge(0)),
  # 4. Index(MonoUp)
  ("Index(MonoUp)", lambda: Index(MonoUp()), lambda: Index(fusable_monoup)),
  # 5. Index(MonoUp) & Ge(0) (Index Mono + Values >= 0)
  (
    "Index(MonoUp) & Ge(0)",
    lambda: Index(MonoUp()) & Ge(0),
    lambda: Index(fusable_monoup) & Ge(0),
  ),
  # 6. Index(MonoUp) | Ge(0)
  (
    "Index(MonoUp) | Ge(0)",
    lambda: Index(MonoUp()) | Ge(0),
    lambda: Index(fusable_monoup) & Ge(0),
  ),
]


def run() -> None:
  logger.info(
    f"Benchmarking N={N:,} rows. Comparing Pandas (Optimized C) vs Numba (Fused Loop)."
  )
  results = []

  for name, v_def, v_fused in validators_to_test:
    # RangeIndex
    t_pandas = bench(
      f"{name} (Pandas) [Range]", v_def, make_range_series, force_numba=False
    )
    t_numba = bench(
      f"{name} (Numba)  [Range]", v_fused, make_range_series, force_numba=True
    )
    results.append({
      "Test": name,
      "Index": "Range",
      "Pandas (ms)": t_pandas * 1000,
      "Numba (ms)": t_numba * 1000,
      "Speedup (Pd/Nu)": t_pandas / t_numba,
    })

    # FloatIndex
    t_pandas_f = bench(
      f"{name} (Pandas) [Float]", v_def, make_float_index_series, force_numba=False
    )
    t_numba_f = bench(
      f"{name} (Numba)  [Float]", v_fused, make_float_index_series, force_numba=True
    )
    results.append({
      "Test": name,
      "Index": "Float",
      "Pandas (ms)": t_pandas_f * 1000,
      "Numba (ms)": t_numba_f * 1000,
      "Speedup (Pd/Nu)": t_pandas_f / t_numba_f,
    })

  df = pd.DataFrame(results)
  # Format for readability
  pd.options.display.float_format = "{:.2f}".format
  logger.info("\n" + "=" * 80)
  logger.info("BENCHMARK RESULTS")
  logger.info("=" * 80)
  logger.info(df.to_string(index=False))


if __name__ == "__main__":
  run()
