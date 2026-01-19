from loguru import logger
import numpy as np
import pandas as pd

from benchmarks.utils import run_benchmark
from datawarden import Finite, Validated, validate


@validate
def validated_func(s: Validated[pd.Series, Finite]) -> None:
  pass


def run_scaling_bench() -> None:
  logger.info("=" * 60)
  logger.info("DATA SIZE SCALING BENCHMARKS")
  logger.info("=" * 60)

  sizes = [100, 1000, 10_000, 100_000, 1_000_000]

  for size in sizes:
    data = pd.Series(np.random.rand(size))
    # Adjust iterations to keep runtime reasonable
    iters = max(10, 100_000 // size)

    run_benchmark(
      f"Size: {size:10,d} rows",
      lambda data=data: validated_func(data),
      iterations=iters,
    )


if __name__ == "__main__":
  run_scaling_bench()
