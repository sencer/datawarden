import contextlib
import gc
import pathlib
import resource
import tracemalloc
from typing import Annotated, Any

from loguru import logger
import numpy as np
import pandas as pd

from datawarden import Overrides, validate
from datawarden.context import ValidationContext
from datawarden.validators import Gt, Lt, MonoUp
from datawarden.validators.structural import Column


def get_current_rss_mb() -> float:
  """Get current Resident Set Size in MB using /proc/self/statm (Linux)."""
  try:
    page_size = resource.getpagesize()
    # /proc/self/statm: size resident shared text lib data dt
    with pathlib.Path("/proc/self/statm").open("r", encoding="utf-8") as f:
      resident_pages = int(f.read().split()[1])
      return (resident_pages * page_size) / 1024 / 1024
  except (FileNotFoundError, IndexError, ValueError):
    # Fallback for non-Linux or error
    return 0.0


def get_peak_rss_mb() -> float:
  """Get peak RSS in MB using getrusage."""
  ru = resource.getrusage(resource.RUSAGE_SELF)
  # ru_maxrss is in kilobytes on Linux
  return ru.ru_maxrss / 1024


def run_memory_benchmark() -> None:
  logger.info("=" * 60)
  logger.info("MEMORY PROFILING BENCHMARKS")
  logger.info("=" * 60)
  logger.info("Note: 'Overhead' measures peak memory increase during validation.")
  logger.info("      For Parallel, low overhead confirms shared-memory threading.")

  n = 5_000_000
  logger.info(f"Generating DataFrame with {n:,} rows (~80MB)...")
  df = pd.DataFrame({"a": np.arange(n, dtype=float), "b": np.random.randn(n)})

  # Force GC and stabilize memory
  gc.collect()
  base_rss = get_current_rss_mb()
  logger.info(f"Baseline Memory (Data Loaded): {base_rss:.2f} MB")

  @validate
  def process(
    df: Annotated[pd.DataFrame, Column("a", MonoUp), Column("b", Gt(-10.0))],
  ) -> None:
    pass

  def measure(label: str, use_numba: bool = False, **kwargs: Any) -> None:  # noqa: ANN401
    gc.collect()
    tracemalloc.start()

    # Snapshot memory before
    _ = get_current_rss_mb()
    start_peak_rss = get_peak_rss_mb()
    _, start_tm_peak = tracemalloc.get_traced_memory()

    try:
      with Overrides(use_numba=use_numba, **kwargs):
        process(df)
    except Exception:  # noqa: BLE001
      logger.exception(f"Failed {label}")
      return

    # Snapshot memory after
    _, end_tm_peak = tracemalloc.get_traced_memory()
    end_peak_rss = get_peak_rss_mb()
    tracemalloc.stop()

    # Calculate overheads
    # RSS Peak Increase: Did we push the process high-water mark?
    rss_peak_delta = max(0.0, end_peak_rss - start_peak_rss)

    # Tracemalloc Peak: Python object overhead
    tm_peak_mb = (end_tm_peak - start_tm_peak) / 1024 / 1024

    label_str = f"{label} (Numba={'On' if use_numba else 'Off'})"
    logger.info(
      f"{label_str:<45} | PyObj Overhead: {tm_peak_mb:6.2f} MB | System Peak Delta: {rss_peak_delta:6.2f} MB"
    )

  # Standard Benchmarks
  measure("No Chunking", chunk_size_rows=None)
  measure("Chunking (500k)", chunk_size_rows=500_000)
  measure(
    "Parallel (500k, 250k thresh)",
    chunk_size_rows=500_000,
    parallel_threshold_rows=250_000,
  )

  # Numba Benchmarks
  measure("No Chunking", use_numba=True, chunk_size_rows=None)
  measure("Chunking (500k)", use_numba=True, chunk_size_rows=500_000)

  # ---------------------------------------------------------
  # Complex Logic Fusion Memory Benchmark (Series)
  # Demonstrates Numba's advantage in avoiding intermediate arrays.
  # ---------------------------------------------------------
  logger.info("-" * 60)
  logger.info("COMPLEX LOGIC FUSION (Series Level)")
  logger.info("Constraint: (s > -1) | (s < 1)")
  logger.info("-" * 60)

  s_data = df["b"]  # Use Series to avoid whole-DF copy overhead

  # 1. Pandas Manual (Eager)
  gc.collect()
  tracemalloc.start()
  _, start_peak = tracemalloc.get_traced_memory()

  # (s > -1) | (s < 1) creates 2 intermediate bool arrays + 1 output
  _ = (s_data > -1) | (s_data < 1)

  _, end_peak = tracemalloc.get_traced_memory()
  tracemalloc.stop()
  peak_pandas = (end_peak - start_peak) / 1024 / 1024
  logger.info(f"Pandas Manual Eager              | PyObj Peak: {peak_pandas:6.2f} MB")

  # 2. DataWarden Fused (Numba)
  v_fused = Gt(-1) | Lt(1)
  ctx = ValidationContext(root_data=s_data)

  # Warmup with full data to trigger JIT and cache loading
  # This ensures we measure steady-state execution memory, not compilation overhead.
  with Overrides(use_numba=True), contextlib.suppress(Exception):
    v_fused.validate(s_data, ctx)

  gc.collect()
  tracemalloc.start()
  _, start_peak = tracemalloc.get_traced_memory()

  with Overrides(use_numba=True):
    v_fused.validate(s_data, ctx)

  _, end_peak = tracemalloc.get_traced_memory()
  tracemalloc.stop()
  peak_numba = (end_peak - start_peak) / 1024 / 1024
  logger.info(f"DataWarden Fused (Numba)         | PyObj Peak: {peak_numba:6.2f} MB")


if __name__ == "__main__":
  run_memory_benchmark()
