"""Validation plan builder for datawarden."""

from __future__ import annotations

import inspect
import typing
from typing import Annotated, Any, get_args, get_origin

from datawarden.base import Validator
from datawarden.exceptions import LogicError
from datawarden.protocols import MetaValidator
from datawarden.solver import resolve_domains
from datawarden.utils import instantiate_validator, is_pandas_type
from datawarden.validators.columns import HasColumn, HasColumns
from datawarden.validators.comparison import Ge, Gt, Le, Lt
from datawarden.validators.value import IgnoringNaNs, Is, Rows, Shape


class ValidationPlanBuilder:
  """Builds validation plans from function type hints."""

  def __init__(self, func: typing.Callable[..., Any]) -> None:
    super().__init__()
    self.func = func
    self.sig = inspect.signature(func)
    self.type_hints = typing.get_type_hints(func, include_extras=True)

  def build(self) -> tuple[dict[str, Any], dict[str, type]]:
    """Build validators and base types for all arguments.

    Returns:
      Tuple of (arg_validators, arg_base_types)
    """
    arg_validators: dict[str, Any] = {}
    arg_base_types: dict[str, type] = {}

    for param_name in self.sig.parameters:
      name = str(param_name)
      if name in self.type_hints:
        hint = self.type_hints[name]

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
          annotated_type = args[0]

          # First arg is the type, rest are metadata (validators)
          raw_validators = self._extract_validators(args[1:])

          if not raw_validators:
            # Store base type even if no validators, for type checking
            arg_base_types[name] = annotated_type
            continue

          is_pandas = is_pandas_type(annotated_type)

          if is_pandas:
            plan = self._build_pandas_plan(name, raw_validators)
            arg_validators[name] = plan
          else:
            # Non-pandas types
            validators = [v for v in raw_validators if isinstance(v, Validator)]
            if validators:
              arg_validators[name] = validators

          # Store the base type for runtime type checking
          arg_base_types[name] = annotated_type

    return arg_validators, arg_base_types

  def _extract_validators(self, metadata: tuple[Any, ...]) -> list[Validator[Any]]:
    """Instantiate and expand validators from Annotated metadata."""
    raw_validators = []
    for item in metadata:
      v = instantiate_validator(item)
      if v:
        raw_validators.append(v)

    # 1. Handle Markers (e.g. IgnoringNaNs())
    has_ignoring_nans_marker = any(
      isinstance(v, IgnoringNaNs) and v.is_marker() for v in raw_validators
    )
    if has_ignoring_nans_marker:
      wrapped_validators = []
      for v in raw_validators:
        if isinstance(v, IgnoringNaNs) and v.is_marker():
          continue  # Skip the marker itself
        if isinstance(v, IgnoringNaNs):
          # Already wrapped, keep as-is
          wrapped_validators.append(v)
        else:
          # Wrap with IgnoringNaNs
          wrapped_validators.append(IgnoringNaNs(v, _check_syntax=False))
      raw_validators = wrapped_validators

    # 2. Expand MetaValidators
    optimized_validators = []
    for v in raw_validators:
      if isinstance(v, MetaValidator):
        optimized_validators.extend(v.transform())
      else:
        optimized_validators.append(v)

    return optimized_validators

  def _build_pandas_plan(
    self, param_name: str, validators: list[Validator[Any]]
  ) -> dict[str, Any]:
    """Build structured validation plan for pandas objects."""
    holistic = []
    globals_list = []
    col_map: dict[str, list[Validator[Any]]] = {}

    for v in validators:
      if isinstance(v, (HasColumn, HasColumns)):
        cols = v.columns if isinstance(v, HasColumns) else [v.column]

        # Instantiate internal validators
        specs = []
        for item in v.validators:
          inst = instantiate_validator(item, _check_syntax=False)
          if inst:
            specs.append(inst)

        for col in cols:
          if str(col) not in col_map:
            col_map[str(col)] = []
          # Clone specs for each column to avoid state sharing
          cloned_specs = [v.clone() for v in specs]
          col_map[str(col)].extend(cloned_specs)

      elif isinstance(v, (Shape, Rows, Is)) or (
        isinstance(v, (Ge, Le, Gt, Lt)) and len(v.targets) > 1
      ):
        holistic.append(v)
      else:
        globals_list.append(v)

    # Resolve Globals
    resolved_globals = list(globals_list)
    try:
      # Check for global contradictions (pass empty locals)
      resolved_globals = resolve_domains(resolved_globals, [])
    except Exception as e:
      if isinstance(e, LogicError):
        raise
      raise ValueError(
        f"Global validation conflict in parameter '{param_name}': {e}"
      ) from e

    # Resolve Columns
    resolved_columns = {}
    for col, local_specs in col_map.items():
      # Resolve: globals apply unless they conflict with local specs
      try:
        resolved = resolve_domains(resolved_globals, local_specs)
        resolved_columns[col] = resolved
      except Exception as e:
        if isinstance(e, LogicError):
          raise
        raise ValueError(
          f"Validation conflict for column '{col}' in parameter '{param_name}': {e}"
        ) from e

    # Sort all validator lists by priority
    holistic.sort(key=lambda v: v.priority)  # pyright: ignore[reportUnknownLambdaType]
    resolved_globals.sort(key=lambda v: v.priority)  # pyright: ignore[reportUnknownLambdaType]
    for col_validators in resolved_columns.values():
      col_validators.sort(key=lambda v: v.priority)  # pyright: ignore[reportUnknownLambdaType]

    return {
      "holistic": holistic,
      "columns": resolved_columns,
      "default": resolved_globals,
      "has_col_checks": bool(col_map),
    }
