"""The @validated decorator for automatic argument validation."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import contextlib
import functools
import inspect
import typing
from typing import (
  TYPE_CHECKING,
  Annotated,
  Any,
  ParamSpec,
  get_args,
  get_origin,
  overload,
)

from loguru import logger
import pandas as pd

from validated.base import Validator
from validated.exceptions import LogicError
from validated.solver import resolve_domains
from validated.utils import apply_default_validators, instantiate_validator
from validated.validators.columns import HasColumn, HasColumns
from validated.validators.comparison import Ge, Gt, Le, Lt
from validated.validators.markers import MaybeEmpty, Nullable
from validated.validators.value import NonEmpty, NonNaN, Shape

if TYPE_CHECKING:
  from collections.abc import Callable

P = ParamSpec("P")
R = typing.TypeVar("R")


@overload
def validated[**P, R](
  func: Callable[P, R],
) -> Callable[P, R]: ...


@overload
def validated[**P, R](
  *, skip_validation_by_default: bool = False, warn_only_by_default: bool = False
) -> Callable[[Callable[P, R]], Callable[P, R | None]]: ...


def validated[**P, R](
  func: Callable[P, R] | None = None,
  *,
  skip_validation_by_default: bool = False,
  warn_only_by_default: bool = False,
) -> Callable[P, R | None] | Callable[[Callable[P, R]], Callable[P, R | None]]:
  """Decorator to validate function arguments based on Annotated types.

  The decorator automatically adds a `skip_validation` parameter to the function.
  When `skip_validation=False` (default), validation is performed. When
  `skip_validation=True`, validation is skipped for maximum performance.

  Args:
    func: The function to decorate.
    skip_validation_by_default: If True, `skip_validation` defaults to True.
    warn_only_by_default: If True, `warn_only` defaults to True. When `warn_only` is
      True, validation failures log an error and return None instead of raising.

  Returns:
    The decorated function with automatic validation support.

  Example:
    >>> from validated import validated, Validated, Finite
    >>> import pandas as pd
    >>>
    >>> @validated
    ... def process(data: Validated[pd.Series, Finite]):
    ...     return data.sum()
    >>>
    >>> # Validation enabled by default
    >>> result = process(valid_data)
    >>>
    >>> # Skip validation for performance
    >>> result = process(valid_data, skip_validation=True)
    >>>
    >>> # Change default behavior
    >>> @validated(skip_validation_by_default=True)
    >>> def fast_process(data: Validated[pd.Series, Finite]):
    ...     return data.sum()
    >>>
    >>> # Validation skipped by default
    >>> result = fast_process(valid_data)
    >>>
    >>> # Enable validation explicitly
    >>> result = fast_process(valid_data, skip_validation=False)
  """

  def decorator(  # noqa: PLR0914
    func: Callable[P, R],
  ) -> Callable[P, R | None]:
    # Inspect function signature
    sig = inspect.signature(func)
    type_hints = typing.get_type_hints(func, include_extras=True)

    # Pre-compute validators and base types for each argument
    arg_validators: dict[str, Any] = {}
    arg_base_types: dict[str, type] = {}

    for param_name in sig.parameters:
      name = str(param_name)
      if name in type_hints:
        hint = type_hints[name]

        # Handle Optional/Union types
        origin = get_origin(hint)
        if (
          origin is typing.Union
          or str(origin) == "typing.Union"
          or str(origin) == "<class 'types.UnionType'>"
        ):
          # Check args for Annotated
          for arg in get_args(hint):
            if get_origin(arg) is Annotated:
              hint = arg
              break

        if get_origin(hint) is Annotated:
          args = get_args(hint)
          # First arg is the type, rest are metadata (validators)
          raw_validators = []
          for item in args[1:]:
            v = instantiate_validator(item)
            if v:
              raw_validators.append(v)

          # Add defaults if not opted out
          # Only apply defaults if the type is pandas Series/DataFrame
          annotated_type = args[0]
          is_pandas = False
          try:
            if issubclass(annotated_type, (pd.Series, pd.DataFrame)):
              is_pandas = True
          except TypeError:
            # annotated_type might be a generic alias or something else
            origin = get_origin(annotated_type)
            if origin is not None:
              if isinstance(origin, type) and issubclass(
                origin, (pd.Series, pd.DataFrame)
              ):
                is_pandas = True
            elif (
              hasattr(annotated_type, "__module__")
              and "pandas" in annotated_type.__module__
            ):
              is_pandas = True

          # Filter and Process Validators
          if is_pandas:
            # 1. Separate into Holistic, Column-Specific, and Global
            holistic = []
            globals_list = []
            col_map: dict[str, list[Validator[Any]]] = {}

            has_column_validator = False

            for v in raw_validators:
              if isinstance(v, (HasColumn, HasColumns)):
                has_column_validator = True
                cols = v.columns if isinstance(v, HasColumns) else [v.column]

                # Instantiate internal validators
                specs = []
                for item in v.validators:
                  inst = instantiate_validator(item)
                  if inst:
                    specs.append(inst)

                for col in cols:
                  if str(col) not in col_map:
                    col_map[str(col)] = []
                  col_map[str(col)].extend(specs)

              elif isinstance(v, Shape) or (
                isinstance(v, (Ge, Le, Gt, Lt)) and len(v.targets) > 1
              ):
                holistic.append(v)
              else:
                globals_list.append(v)

            # Apply defaults to globals if applicable
            # If HasColumn is used, we add Nullable/MaybeEmpty to GLOBALS
            # so they can be overridden or intersected?
            # Actually, if HasColumn is used, user likely wants loose defs elsewhere
            # OR strict defaults elsewhere.
            # Current logic: If HasColumn is present, we add Nullable/MaybeEmpty
            # to the list before resolving defaults.

            if has_column_validator:
              # Implicitly allow defaults if specific columns are checked
              # We append markers. Markers are NOT Validators, so they pass through resolve_domains
              # But resolve_domains expects Validators.
              # apply_default_validators filters based on markers in the list.
              globals_list.append(Nullable())
              globals_list.append(MaybeEmpty())

            # Resolve Globals (apply defaults)
            resolved_globals = apply_default_validators(globals_list)
            try:
              # Check for global contradictions (pass empty locals)
              resolved_globals = resolve_domains(resolved_globals, [])
            except LogicError:
              raise
            except Exception as e:
              raise ValueError(
                f"Global validation conflict in parameter '{name}': {e}"
              ) from e

            # Resolve Columns
            resolved_columns = {}
            for col, local_specs in col_map.items():
              # Handle Markers for Locals
              # If local specs has Nullable, remove NonNaN from Global context for this resolution
              # If local specs has MaybeEmpty, remove NonEmpty

              effective_globals = list(resolved_globals)  # Copy

              # Optimization: Check markers without iterating full list?
              # apply_default_validators handles the logic of "If Nullable is present, don't add NonNaN".
              # But here resolved_globals ALREADY has NonNaN added.
              # So we need to strip it if local says so.

              if any(isinstance(s, Nullable) for s in local_specs):
                effective_globals = [
                  x for x in effective_globals if not isinstance(x, NonNaN)
                ]
              if any(isinstance(s, MaybeEmpty) for s in local_specs):
                effective_globals = [
                  x for x in effective_globals if not isinstance(x, NonEmpty)
                ]

              # Resolve locals (apply default validators to get standard ones if strictly typed?)
              # Usually locals are just constraints. apply_default_validators only adds defaults
              # if not present. If I say Gt(5), I still want NonNaN unless I say Nullable.
              # So let's apply defaults to locals too?
              # Re-think: "Locals merge with Globals".
              # The Solver logic takes "Global Validators" and "Local Validators".
              # We should pass `effective_globals` and `local_specs` to solver.

              # But we need basic validators (NonNaN) in local_specs if they aren't in globals?
              # No, globals provide the defaults.

              # Resolve
              try:
                resolved = resolve_domains(effective_globals, local_specs)
                resolved_columns[col] = resolved
              except LogicError:
                raise
              except Exception as e:
                raise ValueError(
                  f"Validation conflict for column '{col}' in parameter '{name}': {e}"
                ) from e

            # Prepare Plan
            plan = {
              "holistic": holistic,
              "columns": resolved_columns,
              "default": resolved_globals,  # For columns not in col_map
              "has_col_checks": bool(
                col_map
              ),  # To know if we need to check column existence based on keys
            }
            arg_validators[name] = plan

          else:
            # Non-pandas types
            validators = [v for v in raw_validators if isinstance(v, Validator)]
            if validators:
              arg_validators[name] = validators

          # Store the base type for runtime type checking
          arg_base_types[name] = annotated_type

    def validate_arg(  # noqa: PLR0911
      arg_name: str,
      value: Any,  # noqa: ANN401
      warn_only: bool,
    ) -> bool:
      """Validate a single argument. Returns True if valid, False if warn_only triggered."""
      # Check base type first (skip if value is None - Optional types allow None)
      if arg_name in arg_base_types and value is not None:
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
            f"in '{func.__name__}': expected {expected_type.__name__}, "
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
          v_name = "UnknownValidator"
          try:
            for v in plan["holistic"]:
              v_name = type(v).__name__
              v.validate(value)
          except Exception as e:
            msg = f"Validation failed for parameter '{arg_name}' in '{func.__name__}' ({v_name}): {e}"
            if warn_only:
              logger.error(msg)
              return False
            raise type(e)(msg) from e

          # DataFrame Specifics
          if isinstance(value, pd.DataFrame):
            # 2. Check Missing Columns
            if plan["has_col_checks"]:
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
                try:
                  v.validate(col_data)
                except Exception as e:
                  msg = f"Validation failed for parameter '{arg_name}' in '{func.__name__}' (Column '{col_name}', {type(v).__name__}): {e}"
                  if warn_only:
                    logger.error(msg)
                    return False
                  raise type(e)(msg) from e

          elif isinstance(value, pd.Series):
            # Series uses 'default' validators (Globals) + Holistic (already ran)
            for v in plan["default"]:
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
      # Check for skip_validation in kwargs
      skip = kwargs.pop("skip_validation", skip_validation_by_default)
      if skip:
        return func(*args, **kwargs)

      # Check for warn_only in kwargs (explicit cast to bool for type checkers)
      warn_only = bool(kwargs.pop("warn_only", warn_only_by_default))

      # Bind arguments
      bound_args = sig.bind(*args, **kwargs)
      bound_args.apply_defaults()

      # Filter items that need validation
      items_to_validate = [
        (name, value)
        for name, value in bound_args.arguments.items()
        if name in arg_base_types or name in arg_validators
      ]

      # Validate arguments
      if len(items_to_validate) > 1:
        # Parallel validation for multiple arguments
        with ThreadPoolExecutor(max_workers=len(items_to_validate)) as executor:
          results = list(
            executor.map(
              lambda x: validate_arg(x[0], x[1], warn_only),  # pyright: ignore[reportUnknownLambdaType]
              items_to_validate,
            )
          )
          if warn_only and not all(results):
            return None
      elif items_to_validate:
        # Sequential validation for single argument to avoid pool overhead
        name, value = items_to_validate[0]
        if not validate_arg(name, value, warn_only):
          return None

      return func(*args, **kwargs)

    return wrapper

  if func is None:
    return decorator

  return decorator(func)
