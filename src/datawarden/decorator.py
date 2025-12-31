"""The @validate decorator for automatic argument validation."""

from __future__ import annotations

import concurrent.futures
import functools
import inspect
import typing
from typing import (
  TYPE_CHECKING,
  Any,
  ParamSpec,
  overload,
)

from loguru import logger
import pandas as pd

from datawarden.config import get_config
from datawarden.plan import ValidationPlanBuilder
from datawarden.utils import get_chunks, is_numeric

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

    **Reserved Keywords:** When calling a decorated function, `skip_validation`
    and `warn_only` are treated as control flags and removed from `kwargs` unless
    they appear in the function signature. Do not use these names for `**kwargs`
    parameters that your function logic depends on.

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

    def validate_arg(
      arg_name: str,
      value: Any,
      warn_only: bool,
      mode: str = "all",
    ) -> bool:
      """Validate a single argument.

      Returns is_valid.
      """
      # 1. Type Check (Pre-resolved in builder)
      if mode != "chunkable" and value is not None:
        type_info = arg_base_types.get(arg_name)
        if type_info:
          check_type, is_valid_type = type_info
          if is_valid_type and not isinstance(value, check_type):
            msg = (
              f"Type mismatch for parameter '{arg_name}' "
              f"in '{func.__name__}': expected {getattr(check_type, '__name__', str(check_type))}, "
              f"got {type(value).__name__}"
            )
            if warn_only:
              logger.error(msg)
              return False
            raise TypeError(msg)

      # 2. Validation
      if arg_name in arg_validators:
        if value is None:
          return True

        plan = arg_validators[arg_name]

        # Pandas Plan Path
        if plan.get("is_pandas"):
          fast_holistic = plan["fast_holistic"]
          slow_holistic = plan["slow_holistic"]
          columns = plan["columns"]
          has_col_checks = plan["has_col_checks"]
          default = plan["default"]

          # Cache type once
          value_is_numeric = None

          # [SECTION] Holistic (Fast)
          for v, is_v_chunkable, v_is_numeric_only in fast_holistic:
            if (mode == "chunkable" and not is_v_chunkable) or (
              mode == "non-chunkable" and is_v_chunkable
            ):
              continue

            if v_is_numeric_only:
              if value_is_numeric is None:
                value_is_numeric = is_numeric(value)
              if not value_is_numeric:
                continue

            try:
              v.validate(value)
            except Exception as e:
              msg = f"Validation failed for parameter '{arg_name}' in '{func.__name__}' ({type(v).__name__}): {e}"
              if warn_only:
                logger.error(msg)
                return False
              raise type(e)(msg) from e

          # DataFrame Column-wise
          if isinstance(value, pd.DataFrame):
            if mode != "chunkable" and has_col_checks:
              missing = [c for c in columns if c not in value.columns]
              if missing:
                msg = f"Missing columns: {missing}"
                if warn_only:
                  logger.error(msg)
                  return False
                raise ValueError(msg)

            for col_name in value.columns:
              col_data = value[col_name]
              active_v = columns.get(col_name, default)
              if not active_v:
                continue

              col_is_numeric = None
              for v, is_v_chunkable, v_is_numeric_only in active_v:
                if (mode == "chunkable" and not is_v_chunkable) or (
                  mode == "non-chunkable" and is_v_chunkable
                ):
                  continue

                if v_is_numeric_only:
                  if col_is_numeric is None:
                    col_is_numeric = is_numeric(col_data)
                  if not col_is_numeric:
                    continue

                try:
                  v.validate(col_data)
                except Exception as e:
                  msg = f"Validation failed for parameter '{arg_name}' in '{func.__name__}' (Column '{col_name}', {type(v).__name__}): {e}"
                  if warn_only:
                    logger.error(msg)
                    return False
                  raise type(e)(msg) from e

          # Series / Index Path
          elif isinstance(value, (pd.Series, pd.Index)):
            for v, is_v_chunkable, v_is_numeric_only in default:
              if (mode == "chunkable" and not is_v_chunkable) or (
                mode == "non-chunkable" and is_v_chunkable
              ):
                continue

              if v_is_numeric_only:
                if value_is_numeric is None:
                  value_is_numeric = is_numeric(value)
                if not value_is_numeric:
                  continue

              try:
                v.validate(value)
              except Exception as e:
                msg = f"Validation failed for parameter '{arg_name}' in '{func.__name__}' ({type(v).__name__}): {e}"
                if warn_only:
                  logger.error(msg)
                  return False
                raise type(e)(msg) from e

          # [SECTION] Holistic (Slow)
          for v, is_v_chunkable, _ in slow_holistic:
            if (mode == "chunkable" and not is_v_chunkable) or (
              mode == "non-chunkable" and is_v_chunkable
            ):
              continue

            try:
              v.validate(value)
            except Exception as e:
              msg = f"Validation failed for parameter '{arg_name}' in '{func.__name__}' ({type(v).__name__}): {e}"
              if warn_only:
                logger.error(msg)
                return False
              raise type(e)(msg) from e

        # Standard Validator List (Non-pandas)
        else:
          for v, is_v_chunkable, _ in plan["validators"]:
            if (mode == "chunkable" and not is_v_chunkable) or (
              mode == "non-chunkable" and is_v_chunkable
            ):
              continue

            try:
              v.validate(value)
            except Exception as e:
              msg = f"Validation failed for parameter '{arg_name}' in '{func.__name__}' ({type(v).__name__}): {e}"
              if warn_only:
                logger.error(msg)
                return False
              raise type(e)(msg) from e

      return True

    # Pre-compute signature details for fast binding
    parameters = list(sig.parameters.values())
    arg_names = [p.name for p in parameters]
    arg_set = set(arg_names)

    # Check if we have var-args (args/kwargs) which complicate fast path
    has_var_args = any(
      p.kind in {inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD}
      for p in parameters
    )

    # Pre-compute defaults mapping
    defaults = {
      p.name: p.default for p in parameters if p.default is not inspect.Parameter.empty
    }

    # Pre-calculate validation sets
    param_names_with_validation = set(arg_base_types.keys()) | set(
      arg_validators.keys()
    )
    # Sort for deterministic execution
    validation_order = [
      name for name in arg_names if name in param_names_with_validation
    ]

    # Pre-calculate if any stateful validators exist
    all_stateful = []
    for plan in arg_validators.values():
      if isinstance(plan, dict) and "stateful" in plan:
        all_stateful.extend(plan["stateful"])

    @functools.wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R | None:
      # 1. Quick Global Bypass
      config = get_config()
      skip_val = (
        kwargs.pop("skip_validation", skip_validation_by_default)
        if "skip_validation" in kwargs or skip_validation_by_default is not None
        else config.skip_validation
      )
      if skip_val:
        return func(*args, **kwargs)

      warn_only = bool(
        kwargs.pop("warn_only", warn_only_by_default)
        if "warn_only" in kwargs or warn_only_by_default is not None
        else config.warn_only
      )
      # 2. Fast Parameter Binding
      if not has_var_args and len(args) <= len(arg_names):
        bound_arguments = {}
        # Positional
        for i, val in enumerate(args):
          bound_arguments[arg_names[i]] = val
        # Kwargs
        if kwargs:
          for k, v in kwargs.items():
            if k not in arg_set:
              raise TypeError(
                f"{func.__name__} got an unexpected keyword argument '{k}'"
              )
            if k in bound_arguments:
              raise TypeError(f"{func.__name__} got multiple values for argument '{k}'")
            bound_arguments[k] = v
        # Defaults
        if len(bound_arguments) < len(arg_names):
          for name, def_val in defaults.items():
            if name not in bound_arguments:
              bound_arguments[name] = def_val
      else:
        # Fallback to slow bind
        bound = sig.bind(*args, **kwargs)
        bound.apply_defaults()
        bound_arguments = bound.arguments

      # 3. Reset Stateful
      for v in all_stateful:
        v.reset()

      # 4. Filter items that actually need validation
      items_to_validate = [
        (name, bound_arguments[name])
        for name in validation_order
        if name in bound_arguments
      ]

      if not items_to_validate:
        return func(*args, **kwargs)

      # 5. Fast Path: No Chunking
      chunk_size = config.chunk_size_rows
      if chunk_size is None and not config.parallel_threshold_rows:
        for name, value in items_to_validate:
          if not validate_arg(name, value, warn_only, mode="all"):
            return None
        return func(*args, **kwargs)

      # 6. Adaptive Strategy
      total_rows = sum(_estimate_data_size(v) for _, v in items_to_validate)

      if total_rows < (config.parallel_threshold_rows or 100000) and chunk_size is None:
        for name, value in items_to_validate:
          if not validate_arg(name, value, warn_only, mode="all"):
            return None
        return func(*args, **kwargs)

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
          name = future_to_name[future]  # Keep name for potential logging/debugging
          try:
            is_valid = future.result()
          except Exception:
            raise

          if not is_valid:
            # If any fail in parallel, return None (already logged in validate_arg)
            return None
      else:
        # Sequential (with potential chunking)
        for name, value in items_to_validate:
          if chunk_size and isinstance(value, (pd.DataFrame, pd.Series, pd.Index)):
            # 1. Non-chunkable
            if not validate_arg(name, value, warn_only, mode="non-chunkable"):
              return None

            # 2. Chunkable
            for chunk in get_chunks(value, chunk_size):
              if not validate_arg(name, chunk, warn_only, mode="chunkable"):
                return None
          elif not validate_arg(name, value, warn_only, mode="all"):
            return None
      return func(*args, **kwargs)

    return wrapper

  if func is None:
    return decorator

  return decorator(func)
