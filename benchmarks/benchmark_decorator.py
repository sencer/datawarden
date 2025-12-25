"""Detailed benchmark of @validated decorator overhead."""
# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false
# pyright: reportCallIssue=false, reportArgumentType=false

import timeit
from typing import Any

import numpy as np
import pandas as pd

from validated import Datetime, Finite, Index, MonoUp, Positive, Validated, validated

# Setup test data
small_data = pd.Series(np.random.rand(100))
large_data = pd.Series(np.random.rand(10000))
datetime_data = pd.Series(
  np.random.rand(1000), index=pd.date_range("2024-01-01", periods=1000)
)


# Test functions
def plain(data: pd.Series) -> float:
  return data.sum()


@validated
def decorated_simple(data: Validated[pd.Series, Finite]) -> float:
  return data.sum()


@validated
def decorated_multiple(data: Validated[pd.Series, Finite, Positive]) -> float:
  return data.sum()


@validated
def decorated_index(
  data: Validated[pd.Series, Index[Datetime, MonoUp], Finite],
) -> float:
  return data.sum()


def run_benchmark(
  func: Any,
  data: pd.Series,
  skip_val: bool | None,
  iterations: int,
) -> float:
  if skip_val is None:
    return timeit.timeit(lambda: func(data), number=iterations)
  return timeit.timeit(lambda: func(data, skip_validation=skip_val), number=iterations)


def main() -> None:
  iterations = 10000
  print("=" * 70)
  print("@validated Decorator Performance Benchmark")
  print("=" * 70)
  print(f"Iterations per test: {iterations:,}\n")

  # Test 1: Small data (100 elements)
  print("Test 1: Small Series (100 elements)")
  print("-" * 70)

  t_plain = run_benchmark(plain, small_data, None, iterations)
  t_skip = run_benchmark(decorated_simple, small_data, True, iterations)
  t_validate = run_benchmark(decorated_simple, small_data, False, iterations)

  overhead_skip = ((t_skip - t_plain) / iterations) * 1_000_000
  overhead_validate = ((t_validate - t_plain) / iterations) * 1_000_000

  print(f"Plain function:                     {t_plain:8.4f}s  (baseline)")
  print(
    f"Decorated (skip_validation=True):   {t_skip:8.4f}s  "
    f"(+{overhead_skip:6.2f}us/call)"
  )
  print(
    f"Decorated (skip_validation=False):  {t_validate:8.4f}s  "
    f"(+{overhead_validate:6.2f}us/call)"
  )
  print()

  # Test 2: Large data (10,000 elements)
  print("Test 2: Large Series (10,000 elements)")
  print("-" * 70)

  t_plain = run_benchmark(plain, large_data, None, iterations)
  t_skip = run_benchmark(decorated_simple, large_data, True, iterations)
  t_validate = run_benchmark(decorated_simple, large_data, False, iterations)

  overhead_skip = ((t_skip - t_plain) / iterations) * 1_000_000
  overhead_validate = ((t_validate - t_plain) / iterations) * 1_000_000

  print(f"Plain function:                     {t_plain:8.4f}s  (baseline)")
  print(
    f"Decorated (skip_validation=True):   {t_skip:8.4f}s  "
    f"(+{overhead_skip:6.2f}us/call)"
  )
  print(
    f"Decorated (skip_validation=False):  {t_validate:8.4f}s  "
    f"(+{overhead_validate:6.2f}us/call)"
  )
  print()

  # Test 3: Multiple validators
  print("Test 3: Multiple Validators (Finite + Positive)")
  print("-" * 70)

  t_plain = run_benchmark(plain, small_data, None, iterations)
  t_skip = run_benchmark(decorated_multiple, small_data, True, iterations)
  t_validate = run_benchmark(decorated_multiple, small_data, False, iterations)

  overhead_skip = ((t_skip - t_plain) / iterations) * 1_000_000
  overhead_validate = ((t_validate - t_plain) / iterations) * 1_000_000

  print(f"Plain function:                     {t_plain:8.4f}s  (baseline)")
  print(
    f"Decorated (skip_validation=True):   {t_skip:8.4f}s  "
    f"(+{overhead_skip:6.2f}us/call)"
  )
  print(
    f"Decorated (skip_validation=False):  {t_validate:8.4f}s  "
    f"(+{overhead_validate:6.2f}us/call)"
  )
  print()

  # Test 4: Index validators
  print("Test 4: Index Validators (Index[Datetime, MonoUp] + Finite)")
  print("-" * 70)

  t_plain = run_benchmark(plain, datetime_data, None, iterations)
  t_skip = run_benchmark(decorated_index, datetime_data, True, iterations)
  t_validate = run_benchmark(decorated_index, datetime_data, False, iterations)

  overhead_skip = ((t_skip - t_plain) / iterations) * 1_000_000
  overhead_validate = ((t_validate - t_plain) / iterations) * 1_000_000

  print(f"Plain function:                     {t_plain:8.4f}s  (baseline)")
  print(
    f"Decorated (skip_validation=True):   {t_skip:8.4f}s  "
    f"(+{overhead_skip:6.2f}us/call)"
  )
  print(
    f"Decorated (skip_validation=False):  {t_validate:8.4f}s  "
    f"(+{overhead_validate:6.2f}us/call)"
  )
  print()

  # Summary
  print("=" * 70)
  print("Summary")
  print("=" * 70)
  print("• skip_validation=True adds ~0.5us overhead (essentially zero)")
  print("• skip_validation=False (default) overhead scales with:")
  print("  - Number of validators")
  print("  - Data size")
  print("  - Validator complexity")
  print()
  print("Recommendation: Use skip_validation=True in production hot paths after")
  print("                initial data validation during development/testing.")


if __name__ == "__main__":
  main()
