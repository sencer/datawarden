from loguru import logger
import numpy as np
import pandas as pd

from benchmarks.utils import compare_benchmarks
from datawarden import Finite, Overrides, Validated, validate


def run_core_benchmarks() -> None:
  logger.info("=" * 60)
  logger.info("CORE / OVERHEAD BENCHMARKS (Small Data: N=100)")
  logger.info("=" * 60)

  data = pd.Series(np.random.rand(100))

  def plain(s: pd.Series) -> float:
    return s.sum()

  @validate
  def validated(s: Validated[pd.Series, Finite]) -> float:
    return s.sum()

  # 1. Decorator Overhead
  # Measures the cost of creating ValidationContext, plan lookup, and execution
  compare_benchmarks(
    "Decorator Overhead (Raw vs @validate)",
    base_func=lambda: plain(data),
    opt_func=lambda: validated(data),
    iterations=10_000,
    warmup=5,
  )

  # 2. Skip Overhead (Runtime)
  # Measures the cost of checking the context variable
  def run_skipped() -> None:
    with Overrides(skip_validation=True):
      validated(data)

  compare_benchmarks(
    "Skip Flag Overhead (Raw vs Skip=True)",
    base_func=lambda: plain(data),
    opt_func=run_skipped,
    iterations=10_000,
  )

  # 3. Skip Overhead (Import Time)
  # Measures if the decorator is truly stripped away
  with Overrides(skip_validation=True):

    @validate
    def validated_import_skipped(s: Validated[pd.Series, Finite]) -> float:
      return s.sum()

  compare_benchmarks(
    "Zero-Cost Import Skip (Raw vs ImportSkip)",
    base_func=lambda: plain(data),
    opt_func=lambda: validated_import_skipped(data),
    iterations=10_000,
  )


if __name__ == "__main__":
  run_core_benchmarks()
