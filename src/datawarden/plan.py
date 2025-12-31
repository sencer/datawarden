"""Validation plan builder for datawarden."""

from __future__ import annotations

import inspect
import typing
from typing import Annotated, Any, get_args, get_origin

from datawarden.base import CompoundValidator, Priority, Validator
from datawarden.exceptions import LogicError
from datawarden.protocols import MetaValidator
from datawarden.solver import is_domain_validator, resolve_domains
from datawarden.utils import instantiate_validator, is_pandas_type
from datawarden.validators.columns import HasColumn, HasColumns
from datawarden.validators.value import AllowInf, AllowNaN, IgnoringNaNs


class ValidationPlanBuilder:
  """Builds validation plans from function type hints."""

  def __init__(self, func: typing.Callable[..., Any]) -> None:
    super().__init__()
    self.func = func
    self.sig = inspect.signature(func)
    self.type_hints = typing.get_type_hints(func, include_extras=True)

  def build(self) -> tuple[dict[str, Any], dict[str, tuple[type, bool]]]:
    """Build validators and base types for all arguments.

    Returns:
      Tuple of (arg_validators, arg_base_types)
    """
    arg_validators: dict[str, Any] = {}
    arg_base_types: dict[str, tuple[type, bool]] = {}

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
            is_pandas = is_pandas_type(annotated_type)
            check_type = get_origin(annotated_type) or annotated_type
            is_valid_type = isinstance(check_type, type)
            arg_base_types[name] = (check_type, is_valid_type)
            continue

          is_pandas = is_pandas_type(annotated_type)

          if is_pandas:
            plan = self._build_pandas_plan(name, raw_validators)
            arg_validators[name] = plan
          else:
            # Non-pandas types
            validators = [v for v in raw_validators if isinstance(v, Validator)]
            if validators:
              stateful = [v for v in validators if self._is_stateful(v)]
              arg_validators[name] = {
                "validators": self._prep(validators),
                "stateful": stateful,
                "is_pandas": False,
              }

          # Store the base type for runtime type checking
          check_type = get_origin(annotated_type) or annotated_type
          is_valid_type = isinstance(check_type, type)
          arg_base_types[name] = (check_type, is_valid_type)

    return arg_validators, arg_base_types

  def _is_stateful(self, v: Validator[Any]) -> bool:
    """Check if a validator has state that needs resetting."""
    if isinstance(v, CompoundValidator):
      return any(self._is_stateful(sv) for sv in v.validators)
    # Check if reset is overridden
    return type(v).reset is not Validator.reset

  def _prep(
    self, v_list: list[Validator[Any]]
  ) -> list[tuple[Validator[Any], bool, bool]]:
    """Pre-calculate flags for fast access in decorator.

    Format: (validator, is_chunkable, is_numeric_only)
    """
    return [
      (v, getattr(v, "is_chunkable", True), getattr(v, "is_numeric_only", False))
      for v in v_list
    ]

  def _fuse_validators(self, validators: list[Validator[Any]]) -> list[Validator[Any]]:
    """Fuse multiple vectorizable validators into a single CompoundValidator."""
    if len(validators) <= 1:
      return validators

    fused = []
    current_group = []

    for v in validators:
      # Fuse consecutive validators that support vectorized execution
      if hasattr(v, "validate_vectorized"):
        current_group.append(v)
      else:
        if current_group:
          if len(current_group) > 1:
            fused.append(CompoundValidator(current_group))
          else:
            fused.append(current_group[0])
          current_group = []
        fused.append(v)

    if current_group:
      if len(current_group) > 1:
        fused.append(CompoundValidator(current_group))
      else:
        fused.append(current_group[0])

    return fused

  def _extract_validators(self, metadata: tuple[Any, ...]) -> list[Validator[Any]]:
    """Instantiate and expand validators from Annotated metadata."""
    raw_validators = []
    for item in metadata:
      v = instantiate_validator(item)
      if v:
        raw_validators.append(v)

    # 1. Expand MetaValidators
    expanded_validators = []
    for v in raw_validators:
      if isinstance(v, MetaValidator):
        expanded_validators.extend(v.transform())
      else:
        expanded_validators.append(v)
    raw_validators = expanded_validators

    # 2. Handle Markers (e.g. IgnoringNaNs())
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

    return raw_validators

  def _build_pandas_plan(  # noqa: PLR0914
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

      elif v.is_holistic:
        holistic.append(v)
      else:
        globals_list.append(v)

    # Resolve Globals first to catch contradictions
    try:
      # Check for global contradictions (pass empty locals)
      resolved_globals = resolve_domains(list(globals_list), [])
    except Exception as e:
      if isinstance(e, LogicError):
        raise
      raise ValueError(
        f"Global validation conflict in parameter '{param_name}': {e}"
      ) from e

    # 1. Promote eligible global validators to holistic if not overridden.
    # A validator is promotable if is_promotable=True and it's not present in col_map.
    # We also check if any local marker (AllowNaN, AllowInf, IgnoringNaNs marker) exists
    # because they affect how global domain-based validators (Finite, StrictFinite) run.
    has_markers = any(
      (
        isinstance(v, (AllowNaN, AllowInf))
        or (isinstance(v, IgnoringNaNs) and v.is_marker())
      )
      for specs in col_map.values()
      for v in specs
    )

    # Check if any column has domain validators - if so, we can't promote
    # domain validators to holistic since they need per-column resolution
    cols_have_domain_validators = any(
      is_domain_validator(v) for specs in col_map.values() for v in specs
    )

    promoted_globals = []
    final_globals = []
    for v in resolved_globals:
      # If there are ANY markers in the plan, we don't promote to avoid bypassing marker logic
      # If any column has domain validators, don't promote domain globals (need resolution)
      if (
        not has_markers
        and v.is_promotable
        and type(v) not in {type(sv) for specs in col_map.values() for sv in specs}
        and not (cols_have_domain_validators and is_domain_validator(v))
      ):
        promoted_globals.append(v)
      else:
        final_globals.append(v)

    holistic.extend(promoted_globals)
    resolved_globals = final_globals

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

    # Fuse consecutive vectorizable validators
    fused_holistic = self._fuse_validators(holistic)
    fused_default = self._fuse_validators(resolved_globals)
    fused_columns = {
      col: self._fuse_validators(specs) for col, specs in resolved_columns.items()
    }

    # Pre-split holistic into fast/slow
    fast_h = [v for v in fused_holistic if v.priority <= Priority.COMPLEX]
    slow_h = [v for v in fused_holistic if v.priority > Priority.COMPLEX]

    # Collect ALL stateful validators for fast resetting
    stateful = [v for v in fused_holistic + fused_default if self._is_stateful(v)]
    for specs in fused_columns.values():
      stateful.extend([v for v in specs if self._is_stateful(v)])

    return {
      "fast_holistic": self._prep(fast_h),
      "slow_holistic": self._prep(slow_h),
      "columns": {col: self._prep(specs) for col, specs in fused_columns.items()},
      "default": self._prep(fused_default),
      "stateful": stateful,
      "has_col_checks": bool(col_map),
      "is_pandas": True,
    }
