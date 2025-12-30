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

from datawarden.base import Priority
from datawarden.config import get_config
from datawarden.plan import ValidationPlanBuilder
from datawarden.utils import get_chunks, is_numeric

_shared_executor: concurrent.futures.ThreadPoolExecutor | None = None


def _reset_validators(item: Any) -> None:
  """Recursively reset all validators in a plan or list."""
  if isinstance(item, list):
    for v in item:
      _reset_validators(v)
  elif isinstance(item, dict):
    for val in item.values():
      _reset_validators(val)
  elif hasattr(item, "reset"):
    item.reset()


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

    def validate_arg(  # noqa: PLR0914
      arg_name: str,
      value: Any,
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
        if value is None:
          return True

        validators = arg_validators[arg_name]

        # Plan-based validation (Dict)
        if isinstance(validators, dict):
          plan = validators

          holistic = plan.get("holistic", [])
          columns = plan.get("columns", {})
          has_col_checks = plan.get("has_col_checks", False)

          # Split holistic validators into fast (priority <= COMPLEX) and slow (priority > COMPLEX)
          # Fast: Shape(STRUCTURAL), Vectorized(VECTORIZED), Complex(COMPLEX)
          # Slower checks: Rows(SLOW)
          priority_threshold = Priority.COMPLEX
          fast_holistic = [v for v in holistic if v.priority <= priority_threshold]
          slow_holistic = [v for v in holistic if v.priority > priority_threshold]

          # Lazy numeric check for holistic object
          value_is_numeric = None

          # 1. Fast Holistic Checks (e.g. Shape, properties)
          for v in fast_holistic:
            is_chunkable = v.is_chunkable
            if mode == "chunkable" and not is_chunkable:
              continue
            if mode == "non-chunkable" and is_chunkable:
              continue

            # Skip numeric-only validators if whole object is not numeric
            if v.is_numeric_only:
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

          # DataFrame Specifics
          if isinstance(value, pd.DataFrame):
            # 2. Check Missing Columns (Only in non-chunkable or all)
            if mode != "chunkable" and has_col_checks:
              specified_cols = columns.keys()
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
              active_v = columns[col_name] if col_name in columns else plan["default"]
              if not active_v:
                continue

              # Lazy check for numeric status of this column
              col_is_numeric = None

              # Execute
              for v in active_v:
                is_chunkable = getattr(v, "is_chunkable", True)
                if mode == "chunkable" and not is_chunkable:
                  continue
                if mode == "non-chunkable" and is_chunkable:
                  continue

                # Skip numeric-only validators for non-numeric columns
                if v.is_numeric_only:
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

          if isinstance(value, (pd.Series, pd.Index)):
            # Series/Index uses 'default' validators (Globals) + Holistic (already ran)
            for v in plan["default"]:
              is_chunkable = v.is_chunkable
              if mode == "chunkable" and not is_chunkable:
                continue
              if mode == "non-chunkable" and is_chunkable:
                continue

              # Skip numeric-only validators if object is not numeric
              if v.is_numeric_only:
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

          # 4. Slow Holistic Checks (e.g. Rows)
          for v in slow_holistic:
            is_chunkable = v.is_chunkable
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
            is_chunkable = v.is_chunkable
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

    # Pre-compute signature details for fast binding
    parameters = list(sig.parameters.values())
    arg_names = [p.name for p in parameters]

    # Check if we have var-args (args/kwargs) which complicate fast path
    has_var_args = any(
      p.kind in {inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD}
      for p in parameters
    )

    # Pre-compute defaults mapping
    defaults = {
      p.name: p.default for p in parameters if p.default is not inspect.Parameter.empty
    }

    @functools.wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R | None:  # noqa: PLR0914
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

      # Fast path variables
      skip = effective_skip_default
      warn_only = effective_warn_default
      bound_arguments = {}
      skip_in_sig = False
      warn_in_sig = False

      # Determine if we can use fast path
      if not has_var_args:
        # Fast path: Manual mapping
        if len(args) > len(arg_names):
          raise TypeError(
            f"{func.__name__} takes {len(arg_names)} positional arguments but {len(args)} were given"
          )

        # Map positional args
        for i, value in enumerate(args):
          bound_arguments[arg_names[i]] = value

        # Merge kwargs
        for name, value in kwargs.items():
          if name in bound_arguments:
            raise TypeError(
              f"{func.__name__} got multiple values for argument '{name}'"
            )

          # Enforce known arguments (excluding skip/warn which are allowed)
          if name not in arg_names and name not in {"skip_validation", "warn_only"}:
            raise TypeError(
              f"{func.__name__} got an unexpected keyword argument '{name}'"
            )

          bound_arguments[name] = value

        # Apply defaults for missing args
        for name, default_val in defaults.items():
          if name not in bound_arguments:
            bound_arguments[name] = default_val

        skip_in_sig = "skip_validation" in arg_names
        warn_in_sig = "warn_only" in arg_names

        # Determine skip
        if skip_in_sig:
          skip = bound_arguments.get("skip_validation", effective_skip_default)
        else:
          skip = kwargs.get("skip_validation", effective_skip_default)

          if skip:
            # If we need to strip skip_validation from kwargs before calling:
            if not skip_in_sig and "skip_validation" in kwargs:
              # Return func with cleaned kwargs
              return func(
                *args, **{k: v for k, v in kwargs.items() if k != "skip_validation"}
              )  # pyright: ignore[reportCallIssue]
            return func(*args, **kwargs)  # pyright: ignore[reportCallIssue]

        # Determine warn_only
        if warn_in_sig:
          warn_only = bool(bound_arguments.get("warn_only", effective_warn_default))
        else:
          warn_only = bool(kwargs.get("warn_only", effective_warn_default))

      else:
        # Slow path (fallback to bind)
        # We need to handle skip_validation/warn_only carefully if they are not in sig.

        # Check kwargs for magic params before bind (since bind would fail if not in sig)
        kwargs_copy = None

        if "skip_validation" not in sig.parameters and "skip_validation" in kwargs:
          if kwargs_copy is None:
            kwargs_copy = kwargs.copy()
          skip = kwargs_copy.pop("skip_validation")
        elif "skip_validation" in sig.parameters:
          # Will be extracted from bound args
          # Initialize with default for logic below (will be overwritten if present)
          skip = effective_skip_default
        else:
          skip = effective_skip_default

        if "warn_only" not in sig.parameters and "warn_only" in kwargs:
          if kwargs_copy is None:
            kwargs_copy = kwargs.copy()
          warn_only = bool(kwargs_copy.pop("warn_only"))
        else:
          # Will be extracted later or use default
          warn_only = effective_warn_default  # Placeholder

        # Use cleaned kwargs for binding
        final_kwargs_for_bind = kwargs_copy if kwargs_copy is not None else kwargs

        bound = sig.bind(*args, **final_kwargs_for_bind)
        bound.apply_defaults()
        bound_arguments = bound.arguments

        if "skip_validation" in sig.parameters:
          skip = bound_arguments.get("skip_validation", effective_skip_default)

        if skip:
          return func(*args, **final_kwargs_for_bind)  # pyright: ignore[reportCallIssue]

        if "warn_only" in sig.parameters:
          warn_only = bool(bound_arguments.get("warn_only", effective_warn_default))

      # Reset all validators to clear any state from previous runs
      _reset_validators(arg_validators)

      # Filter items that need validation
      items_to_validate = [
        (name, bound_arguments[name])
        for name in bound_arguments
        if name in arg_base_types or name in arg_validators
      ]

      def _handle_result(_: str, is_valid: bool) -> bool:
        return not (not is_valid and warn_only)

      chunk_size = config.chunk_size_rows

      # Validate arguments - adaptive threading based on data size
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
        # Sequential
        for name, value in items_to_validate:
          if chunk_size and isinstance(value, (pd.DataFrame, pd.Series, pd.Index)):
            # 1. Non-chunkable
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

      # Prepare final arguments for call
      if not has_var_args:
        # Fast path cleanup
        # Need to remove skip/warn if they were in kwargs and not in sig
        keys_to_remove = set()
        if not skip_in_sig and "skip_validation" in kwargs:
          keys_to_remove.add("skip_validation")
        if not warn_in_sig and "warn_only" in kwargs:
          keys_to_remove.add("warn_only")

        if keys_to_remove:
          return func(
            *args, **{k: v for k, v in kwargs.items() if k not in keys_to_remove}
          )  # pyright: ignore[reportCallIssue]

      return func(*args, **kwargs)

    return wrapper

  if func is None:
    return decorator

  return decorator(func)
