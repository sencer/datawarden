"""The @validate decorator for automatic argument validation."""

from __future__ import annotations

import concurrent.futures
import contextlib
import functools
import inspect
import typing
from typing import (
  TYPE_CHECKING,
  Any,
  ParamSpec,
  get_origin,
  overload,
)

from loguru import logger
import pandas as pd

from datawarden.config import get_config
from datawarden.plan import ValidationPlanBuilder
from datawarden.utils import get_chunks

_shared_executor: concurrent.futures.ThreadPoolExecutor | None = None


def _get_executor() -> concurrent.futures.ThreadPoolExecutor:
  """Get or create the shared thread pool executor (lazy initialization)."""
  global _shared_executor
  if _shared_executor is None:
    _shared_executor = concurrent.futures.ThreadPoolExecutor(
      max_workers=get_config().max_workers
    )
  return _shared_executor


def _estimate_data_size(value: object) -> int:
  """Estimate data size in rows for parallel threshold check."""
  if isinstance(value, (pd.DataFrame, pd.Series, pd.Index)):
    return len(value)
  return 0


if TYPE_CHECKING:
  from collections.abc import Callable

P = ParamSpec("P")
R = typing.TypeVar("R")


@overload
def validate[**P, R](
  func: Callable[P, R],
) -> Callable[P, R]: ...


@overload
def validate[**P, R](
  *,
  skip_validation_by_default: bool | None = None,
  warn_only_by_default: bool | None = None,
) -> Callable[[Callable[P, R]], Callable[P, R | None]]: ...


def validate[**P, R](
  func: Callable[P, R] | None = None,
  *,
  skip_validation_by_default: bool | None = None,
  warn_only_by_default: bool | None = None,
) -> Callable[P, R | None] | Callable[[Callable[P, R]], Callable[P, R | None]]:
  """Decorator to validate function arguments based on Annotated types.

  The decorator automatically adds a `skip_validation` parameter to the function
  unless the function already defines it. When `skip_validation=False` (default),
  validation is performed. When `skip_validation=True`, validation is skipped
  for maximum performance.

  Argument Collision:
    If your function already has `skip_validation` or `warn_only` parameters,
    the decorator will respect them and pass the values through to your function,
    while still using them to control validation behavior.

  Validation features:
  - **Type Checking:** Runtime checks against Annotated base types.
  - **Constraint Validation:** Applies all validators in the Annotated metadata.
  - **Memory Efficiency:** Supports chunked validation for large datasets (DataFrame,
    Series, Index) via `config.chunk_size_rows`.
  - **Parallelism:** Validates multiple large arguments (DataFrame, Series, Index)
    in parallel threads (unless chunking is active).

  Args:
    func: The function to decorate.
    skip_validation_by_default: If True, `skip_validation` defaults to True.
      If None, defaults to the global configuration `skip_validation`.
    warn_only_by_default: If True, `warn_only` defaults to True. When `warn_only` is
      True, validation failures log an error and return None instead of raising.
      If None, defaults to the global configuration `warn_only`.

  Returns:
    The decorated function with automatic validation support.
  """

  def decorator(
    func: Callable[P, R],
  ) -> Callable[P, R | None]:
    # Inspect function signature
    sig = inspect.signature(func)

    # Build validation plan
    builder = ValidationPlanBuilder(func)
    arg_validators, arg_base_types = builder.build()

    def validate_arg(  # noqa: PLR0911
      arg_name: str,
      value: Any,  # noqa: ANN401
      warn_only: bool,
      mode: str = "all",
    ) -> bool:
      """Validate a single argument. Returns is_valid."""
      # Check base type first (skip if value is None - Optional types allow None)
      # Only check type in "all" or "non-chunkable" modes to avoid redundant checks
      if mode != "chunkable" and arg_name in arg_base_types and value is not None:
        expected_type = arg_base_types[arg_name]
        # Handle generic types by extracting origin
        check_type = get_origin(expected_type) or expected_type
        # Check if check_type is valid for isinstance
        is_valid_type = False
        with contextlib.suppress(TypeError):
          is_valid_type = isinstance(check_type, type)
        if is_valid_type and not isinstance(value, check_type):
          msg = (
            f"Type mismatch for parameter '{arg_name}' "
            f"in '{func.__name__}': expected {getattr(expected_type, '__name__', str(expected_type))}, "
            f"got {type(value).__name__}"
          )
          if warn_only:
            logger.error(msg)
            return False
          raise TypeError(msg)

      if arg_name in arg_validators:
        validators = arg_validators[arg_name]

        # Plan-based validation (Dict)
        if isinstance(validators, dict):
          plan = validators

          # Common: Holistic Validators
          for v in plan["holistic"]:
            is_chunkable = getattr(v, "is_chunkable", True)
            if mode == "chunkable" and not is_chunkable:
              continue
            if mode == "non-chunkable" and is_chunkable:
              continue

            try:
              v.validate(value)
            except Exception as e:
              msg = f"Validation failed for parameter '{arg_name}' in '{func.__name__}' ({type(v).__name__}): {e}"
              if warn_only:
                logger.error(msg)
                return False
              raise type(e)(msg) from e

          # DataFrame Specifics
          if isinstance(value, pd.DataFrame):
            # 2. Check Missing Columns (Only in non-chunkable or all)
            if mode != "chunkable" and plan["has_col_checks"]:
              specified_cols = plan["columns"].keys()
              missing = [c for c in specified_cols if c not in value.columns]
              if missing:
                msg = f"Missing columns: {missing}"
                if warn_only:
                  logger.error(msg)
                  return False
                raise ValueError(msg)

            # 3. Column-wise execution
            for col_name in value.columns:
              col_data = value[col_name]

              # Determine active validators
              if col_name in plan["columns"]:
                active_v = plan["columns"][col_name]
              else:
                active_v = plan["default"]

              # Execute
              for v in active_v:
                is_chunkable = getattr(v, "is_chunkable", True)
                if mode == "chunkable" and not is_chunkable:
                  continue
                if mode == "non-chunkable" and is_chunkable:
                  continue

                try:
                  v.validate(col_data)
                except Exception as e:
                  msg = f"Validation failed for parameter '{arg_name}' in '{func.__name__}' (Column '{col_name}', {type(v).__name__}): {e}"
                  if warn_only:
                    logger.error(msg)
                    return False
                  raise type(e)(msg) from e

          elif isinstance(value, (pd.Series, pd.Index)):
            # Series/Index uses 'default' validators (Globals) + Holistic (already ran)
            for v in plan["default"]:
              is_chunkable = getattr(v, "is_chunkable", True)
              if mode == "chunkable" and not is_chunkable:
                continue
              if mode == "non-chunkable" and is_chunkable:
                continue

              try:
                v.validate(value)
              except Exception as e:
                msg = f"Validation failed for parameter '{arg_name}' in '{func.__name__}' ({type(v).__name__}): {e}"
                if warn_only:
                  logger.error(msg)
                  return False
                raise type(e)(msg) from e

        # List-based validation (Legacy/Standard)
        elif isinstance(validators, list):
          for v in validators:
            is_chunkable = getattr(v, "is_chunkable", True)
            if mode == "chunkable" and not is_chunkable:
              continue
            if mode == "non-chunkable" and is_chunkable:
              continue

            try:
              v.validate(value)
            except Exception as e:
              validator_name = type(v).__name__
              msg = (
                f"Validation failed for parameter '{arg_name}' "
                f"in '{func.__name__}' ({validator_name}): {e}"
              )
              if warn_only:
                logger.error(msg)
                return False
              raise type(e)(msg) from e
      return True

    @functools.wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R | None:
      # Resolve defaults
      config = get_config()
      effective_skip_default = (
        skip_validation_by_default
        if skip_validation_by_default is not None
        else config.skip_validation
      )
      effective_warn_default = (
        warn_only_by_default if warn_only_by_default is not None else config.warn_only
      )

      # Check for skip_validation in kwargs
      if "skip_validation" in sig.parameters:
        skip = kwargs.get("skip_validation", effective_skip_default)
      else:
        skip = kwargs.pop("skip_validation", effective_skip_default)

      if skip:
        return func(*args, **kwargs)

      # Check for warn_only in kwargs (explicit cast to bool for type checkers)
      if "warn_only" in sig.parameters:
        warn_only = bool(kwargs.get("warn_only", effective_warn_default))
      else:
        warn_only = bool(kwargs.pop("warn_only", effective_warn_default))

      # Bind arguments
      bound_args = sig.bind(*args, **kwargs)
      bound_args.apply_defaults()

      # Reset all validators to clear any state from previous runs
      for plan_or_list in arg_validators.values():
        if isinstance(plan_or_list, dict):
          # Plan: holistic, columns (dict of lists), default (list)
          for v in plan_or_list["holistic"]:
            v.reset()
          for v_list in plan_or_list["columns"].values():
            for v in v_list:
              v.reset()
          for v in plan_or_list["default"]:
            v.reset()
        elif isinstance(plan_or_list, list):
          for v in plan_or_list:
            v.reset()

      # Filter items that need validation
      items_to_validate = [
        (name, value)
        for name, value in bound_args.arguments.items()
        if name in arg_base_types or name in arg_validators
      ]

      def _handle_result(_: str, is_valid: bool) -> bool:
        return not (not is_valid and warn_only)

      chunk_size = config.chunk_size_rows

      # Validate arguments - adaptive threading based on data size
      # Only use parallel for multiple args with large data, and when chunking is disabled
      total_rows = sum(_estimate_data_size(v) for _, v in items_to_validate)
      use_parallel = (
        len(items_to_validate) > 1
        and total_rows >= config.parallel_threshold_rows
        and chunk_size is None
      )

      if use_parallel:
        executor = _get_executor()
        future_to_name = {
          executor.submit(validate_arg, name, value, warn_only, "all"): name
          for name, value in items_to_validate
        }

        for future in concurrent.futures.as_completed(future_to_name):
          name = future_to_name[future]
          try:
            is_valid = future.result()
          except Exception:
            raise

          if not _handle_result(name, is_valid):
            return None
      else:
        # Sequential for single argument, small data, or when chunking is enabled
        for name, value in items_to_validate:
          if chunk_size and isinstance(value, (pd.DataFrame, pd.Series, pd.Index)):
            # 1. Non-chunkable (Full data) - includes type checks
            is_valid = validate_arg(name, value, warn_only, mode="non-chunkable")
            if not _handle_result(name, is_valid):
              return None

            # 2. Chunkable
            for chunk in get_chunks(value, chunk_size):
              is_valid = validate_arg(name, chunk, warn_only, mode="chunkable")
              if not _handle_result(name, is_valid):
                return None
          else:
            is_valid = validate_arg(name, value, warn_only, mode="all")
            if not _handle_result(name, is_valid):
              return None

      return func(*bound_args.args, **bound_args.kwargs)

    return wrapper

  if func is None:
    return decorator

  return decorator(func)
