"""Utility functions from datawarden."""

from __future__ import annotations

from typing import Any, get_origin

import pandas as pd

from datawarden.base import Validator


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
    True if the type is a pandas Series or DataFrame, False otherwise.
  """
  try:
    if isinstance(annotated_type, type) and issubclass(
      annotated_type, (pd.Series, pd.DataFrame)
    ):
      return True
  except TypeError:
    pass

  # Try getting the origin for generic types
  origin = get_origin(annotated_type)
  if origin is not None:
    try:
      if isinstance(origin, type) and issubclass(origin, (pd.Series, pd.DataFrame)):
        return True
    except TypeError:
      pass

  # Fallback to module-based detection - check for exact pandas module
  if hasattr(annotated_type, "__module__"):
    module = getattr(annotated_type, "__module__", "")
    # Only match if it's the actual pandas module (not third-party extensions)
    if module in {"pandas.core.frame", "pandas.core.series"}:
      return True

  return False
