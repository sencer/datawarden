from collections.abc import Callable
import gc
import statistics
import time
from typing import Any, NamedTuple

from loguru import logger


class BenchmarkResult(NamedTuple):
  min_time: float
  mean_time: float
  stdev: float
  loops: int
  repeats: int


def format_time(seconds: float) -> str:
  if seconds < 1e-6:  # noqa: PLR2004
    return f"{seconds * 1e9:.2f} ns"
  if seconds < 1e-3:  # noqa: PLR2004
    return f"{seconds * 1e6:.2f} μs"
  if seconds < 1:
    return f"{seconds * 1e3:.2f} ms"
  return f"{seconds:.2f} s"


def run_benchmark(  # noqa: PLR0913
  name: str,
  func: Callable[[], Any],
  *,
  setup: Callable[[], Any] | None = None,
  repeats: int = 7,
  iterations: int = 1000,
  warmup: int = 2,
) -> BenchmarkResult:
  """
  Run a benchmark robustly.

  1. Warmup (run 'warmup' loops of 'iterations').
  2. Run 'repeats' loops.
  3. Each loop runs 'iterations' times.
  4. Remove outliers (slowest 2 if repeats >= 5).
  5. Report min and stdev of the remaining.
  """
  if setup:
    setup()

  # Warmup
  for _ in range(warmup):
    for _ in range(iterations):
      func()

  times = []
  for _ in range(repeats):
    gc.collect()
    # Disable GC during measurement to avoid spikes
    gc_enabled = gc.isenabled()
    gc.disable()

    try:
      start = time.perf_counter()
      for _ in range(iterations):
        func()
      end = time.perf_counter()
      times.append((end - start) / iterations)
    finally:
      if gc_enabled:
        gc.enable()

  # Robustness: Remove outliers
  sorted_times = sorted(times)

  # If we have enough samples, drop the slowest ones (noise)
  valid_times = sorted_times[:-2] if repeats >= 5 else sorted_times  # noqa: PLR2004

  min_time = valid_times[0]
  mean_time = statistics.mean(valid_times)
  stdev = statistics.stdev(valid_times) if len(valid_times) > 1 else 0.0

  logger.info(f"{name:<50} | {format_time(min_time)} ± {format_time(stdev)}")

  return BenchmarkResult(min_time, mean_time, stdev, iterations, repeats)


def compare_benchmarks(
  name: str,
  base_func: Callable[[], Any],
  opt_func: Callable[[], Any],
  **kwargs: Any,  # noqa: ANN401
) -> None:
  logger.info(f"--- Comparing: {name} ---")
  res_base = run_benchmark("Baseline", base_func, **kwargs)
  res_opt = run_benchmark("Optimized", opt_func, **kwargs)

  if res_opt.min_time > 0:
    speedup = res_base.min_time / res_opt.min_time
    logger.info(f"Speedup: {speedup:.2f}x\n")
  else:
    logger.info("Speedup: inf\n")
