"""Utility functions from datawarden."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, get_origin

import pandas as pd

from datawarden.base import Validator

if TYPE_CHECKING:
  from collections.abc import Iterator

  import numpy as np


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
) -> Validator[Any] | None:  # pyright: ignore[reportExplicitAny]
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


def is_pandas_type(annotated_type: object) -> bool:
  """Check if a type annotation represents a pandas type.

  Handles direct types, generic aliases, and module-based detection.

  Args:
    annotated_type: The type to check.

  Returns:
    True if the type is a pandas Series, DataFrame or Index, False otherwise.
  """
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

  # Fallback to module-based detection - check for exact pandas module
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

  return False


def report_failures(
  data: pd.Series | pd.DataFrame | pd.Index,
  mask: pd.Series | pd.DataFrame | np.ndarray,  # pyright: ignore[reportExplicitAny]
  msg: str,
) -> None:
  """Report validation failures with context."""
  # Calculate number of failures
  n_failed = mask.sum().sum() if isinstance(mask, pd.DataFrame) else mask.sum()

  if n_failed == 0:
    return

  # Extract failing indices (limit to 5)
  try:
    if isinstance(data, (pd.Series, pd.Index)):
      # For Index, we need to handle it carefully as it doesn't support boolean indexing same way always
      if isinstance(data, pd.Index):
        # Convert to Series to safely get index of failures
        # If data IS the index, the "index" of failures is the values themselves?
        # Or the integer position? Usually index validation implies checking index values.
        failures = data[mask].tolist()[:5]  # pyright: ignore[reportArgumentType]
      else:
        # Series
        failures = data[mask].index.tolist()[:5]  # pyright: ignore[reportAttributeAccessIssue]
    # DataFrame
    # Stack to get (index, column) pairs
    elif isinstance(mask, pd.DataFrame):
      failures = mask.stack()[mask.stack()].index.tolist()[:5]  # pyright: ignore
    else:
      failures = ["(unknown indices)"]
  except Exception:  # noqa: BLE001
    # Fallback if index extraction fails
    failures = ["(error extracting indices)"]

  raise ValueError(f"{msg} ({n_failed} violations. Examples: {failures})")
