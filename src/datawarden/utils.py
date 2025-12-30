"""Utility functions from datawarden."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast, get_origin

import numpy as np
import pandas as pd

from datawarden.base import Validator

if TYPE_CHECKING:
  from collections.abc import Iterator


def get_chunks(
  data: pd.Series | pd.DataFrame | pd.Index,
  chunk_size: int,
) -> Iterator[pd.Series | pd.DataFrame | pd.Index]:
  """Yield chunks of data for memory-efficient processing.

  Args:
    data: The pandas object to chunk.
    chunk_size: Number of rows per chunk.

  Yields:
    Slices of the original data.
  """
  n_rows = len(data)
  for i in range(0, n_rows, chunk_size):
    if isinstance(data, pd.Index):
      yield data[i : i + chunk_size]
    else:
      yield data.iloc[i : i + chunk_size]


def instantiate_validator(
  item: object,
  *,
  _check_syntax: bool = True,
) -> Validator[Any] | None:
  """Helper to instantiate a validator from a type or instance.

  Args:
    item: The object to check and potentially instantiate.
    _check_syntax: If True, enforces class syntax for 0-arg validators.

  Returns:
    The validator instance if item is a Validator or Validator subclass,
    otherwise None.
  """
  if isinstance(item, Validator):
    # Enforce class syntax for validators with no arguments
    if _check_syntax:
      try:
        # Try to create a default instance from the class
        default_instance = item.__class__()
        if item == default_instance:
          raise ValueError(
            f"Use validator class '{item.__class__.__name__}' instead of instance "
            + f"'{item.__class__.__name__}()' when no arguments are provided."
          )
      except TypeError:
        # __init__ requires arguments, so an instance is valid/required
        pass
    return item
  if isinstance(item, type) and issubclass(item, Validator):
    try:
      return item()  # type: ignore[return-value]
    except TypeError as e:
      raise TypeError(
        f"Validator class '{item.__name__}' could not be instantiated (missing arguments?). "
        + f"Did you forget to instantiate it like '{item.__name__}(...)'?"
      ) from e
  return None


# Fast numeric type detection constants
NUMERIC_KINDS = frozenset("iufc")


def is_numeric(data: pd.Series | pd.Index | pd.DataFrame) -> bool:
  """Check if data is numeric using a fast kind check.

  Args:
    data: The pandas object to check.

  Returns:
    True if numeric, False otherwise.
  """
  if isinstance(data, pd.DataFrame):
    return all(col.kind in NUMERIC_KINDS for col in data.dtypes)
  return data.dtype.kind in NUMERIC_KINDS


def is_pandas_type(annotated_type: object) -> bool:
  """Check if a type annotation represents a pandas type.

  Handles direct types, generic aliases, and module-based detection.

  Args:
    annotated_type: The type to check.

  Returns:
    True if the type is a pandas Series, DataFrame or Index, False otherwise.
  """
  # Fallback to module-based detection first - it's fast and handles aliases
  if hasattr(annotated_type, "__module__"):
    module = getattr(annotated_type, "__module__", "")
    # Only match if it's the actual pandas module (not third-party extensions)
    if module in {
      "pandas.core.frame",
      "pandas.core.series",
      "pandas.core.indexes.base",
      "pandas.core.indexes.datetimes",
    }:
      return True

  try:
    if isinstance(annotated_type, type) and issubclass(
      annotated_type, (pd.Series, pd.DataFrame, pd.Index)
    ):
      return True
  except TypeError:
    pass

  # Try getting the origin for generic types
  origin = get_origin(annotated_type)
  if origin is not None:
    try:
      if isinstance(origin, type) and issubclass(
        origin, (pd.Series, pd.DataFrame, pd.Index)
      ):
        return True
    except TypeError:
      pass

  return False


def scalar_any(
  data: pd.Series | pd.DataFrame | pd.Index | np.ndarray | bool | np.bool_,
) -> bool:
  """Check if any value is True across Series, DataFrame, Index, or Array.

  Handles axis=None for DataFrames and standard .any() for others.
  """
  if isinstance(data, pd.DataFrame):
    return bool(data.values.any())
  if isinstance(data, pd.Series):
    # Use pandas .any() for compatibility with ExtensionArrays (e.g. Categorical)
    return bool(data.any())
  if isinstance(data, pd.Index):
    return bool(data.to_numpy().any())
  if isinstance(data, np.ndarray):
    return bool(data.any())
  return bool(data)


def report_failures(
  data: pd.Series | pd.DataFrame | pd.Index,
  mask: pd.Series | pd.DataFrame | pd.Index | np.ndarray | bool | np.bool_,
  msg: str,
) -> None:
  """Report validation failures with context."""
  # Calculate number of failures
  if isinstance(mask, (bool, np.bool_)):
    n_failed = 1 if mask else 0
  elif isinstance(mask, pd.DataFrame):
    n_failed = mask.sum().sum()
  elif isinstance(mask, (pd.Index, np.ndarray)):
    n_failed = int(np.sum(mask))
  else:
    n_failed = mask.sum()

  if n_failed == 0:
    return

  # Extract failing indices (limit to 5)
  try:
    if isinstance(data, (pd.Series, pd.Index)):
      # Handle both Series/Index with either boolean Series/Index or numpy mask
      if isinstance(data, pd.Index):
        # Convert to Series to safely get index of failures if needed,
        # but for Index itself, we usually want the values.
        failures: list[Any] = data[cast("Any", mask)].tolist()[:5]
      # Series - mask can be Series, Index or ndarray
      # Use numpy-style indexing if it's a mask
      elif isinstance(mask, (pd.Series, pd.Index)):
        failures = data[cast("Any", mask)].index.tolist()[:5]
      else:
        failures = data.index[cast("Any", mask)].tolist()[:5]
    # DataFrame
    elif isinstance(data, pd.DataFrame):
      if isinstance(mask, pd.DataFrame):
        # Stack to get (index, column) pairs
        m = cast("Any", mask)
        failures = m.stack()[m.stack()].index.tolist()[:5]
      else:
        # Numpy mask on DataFrame - usually from vectorized checks
        # Try to find which rows/cols failed
        rows, cols = np.where(mask)
        failures = []
        for r, c in zip(rows[:5], cols[:5], strict=False):
          failures.append((data.index[r], data.columns[c]))
    else:
      failures = ["(unknown indices)"]
  except Exception as e:  # noqa: BLE001
    # Fallback if index extraction fails
    failures = [f"(error extracting indices: {e})"]

  raise ValueError(f"{msg} ({n_failed} violations. Examples: {failures})")
