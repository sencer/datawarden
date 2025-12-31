"""Domain constraint solver for validator resolution."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
  from collections.abc import Sequence

  from datawarden.base import Validator

from datawarden.domain import ValidationDomain
from datawarden.exceptions import LogicError
from datawarden.validators.comparison import Ge, Gt, Le, Lt
from datawarden.validators.logic import Not
from datawarden.validators.value import (
  AllowInf,
  AllowNaN,
  Between,
  Finite,
  IgnoringNaNs,
  IsNaN,
  Negative,
  OneOf,
  Positive,
  StrictFinite,
)


def is_domain_validator(
  v: Validator[Any],
) -> bool:
  """Check if validator can be mapped to a domain."""
  if isinstance(v, IgnoringNaNs):
    # Only treat as domain validator if it's a marker or wraps a domain validator
    return v.wrapped is None or is_domain_validator(v.wrapped)

  return isinstance(
    v,
    (
      Ge,
      Gt,
      Le,
      Lt,
      OneOf,
      Finite,
      StrictFinite,
      IsNaN,
      IgnoringNaNs,
      AllowNaN,
      AllowInf,
      Between,
      Positive,
      Negative,
    ),
  ) or (isinstance(v, Not) and isinstance(v.wrapped, (Negative, Positive, IsNaN)))


def resolve_domains(
  global_validators: Sequence[Validator[Any]],
  local_validators: Sequence[Validator[Any]],
) -> list[Validator[Any]]:
  """
  Resolve potential conflicts between global and local validators.

  Strategy:
  1. Filter out non-domain validators (return as is).
  2. Convert remaining to ValidationDomain.
  3. Check for internal contradictions (LogicError) within local scope.
  4. Intersect Local with Global.
     - If Intersection is Empty -> Local overrides Global (User Intent).
     - If Intersection is Valid -> Use Intersection.
  5. Apply explicit local relaxations (AllowNaN, AllowInf).
  """

  # 1. Separate domain vs other validators
  domain_globals = [v for v in global_validators if is_domain_validator(v)]
  global_others = [v for v in global_validators if not is_domain_validator(v)]

  domain_locals = [v for v in local_validators if is_domain_validator(v)]
  local_others = [v for v in local_validators if not is_domain_validator(v)]

  # Resolve "Other" validators (Type-based override)
  # Map type -> validator. Local overrides Global.
  other_map = {type(v): v for v in global_others}
  other_map.update({type(v): v for v in local_others})
  # Clone them to ensure each column gets its own stateful instance
  resolved_others = [v.clone() for v in other_map.values()]

  # 2. Build Domains
  # Combine all globals into one domain
  global_domain = ValidationDomain()
  for v in domain_globals:
    v_dom = ValidationDomain.from_validator(v)
    # Check global consistency? Assuming globals are consistent or we filter
    # But let's just intersect them
    global_domain = global_domain.intersect(v_dom)
    if global_domain.is_empty():
      # Globals contradict each other!
      raise LogicError(f"Global validators contradiction: {global_validators}")

  # Combine all locals into one domain
  if domain_locals:
    local_domain = ValidationDomain()
    for v in domain_locals:
      v_dom = ValidationDomain.from_validator(v)
      local_domain = local_domain.intersect(v_dom)
      if local_domain.is_empty():
        # Locals contradict each other! e.g. Gt(10), Lt(5) in same HasColumn
        raise LogicError(f"Local validators contradiction: {local_validators}")

    # 3. Intersect Global + Local
    # We start with the local domain as the base "intent"
    final_domain = local_domain

    if not global_domain.is_subset(local_domain):
      # Global adds constraints not present in local.
      # Try to intersection.
      intersection = local_domain.intersect(global_domain)

      if intersection.is_empty():
        # Contradiction!
        # Global says > 10, Local says < 5.
        # Strategy: Local Overrides Global.
        # We keep `local_domain` as is.
        pass
      else:
        # Compatible. Refine local domain.
        final_domain = intersection
  else:
    # No local domain constraints -> Inherit Global Domain
    final_domain = global_domain

  # 5. Apply explicit local relaxations (Task 3)
  # If local validators explicitly say "Allow", force it.
  for v in domain_locals:
    if isinstance(v, AllowNaN) or (isinstance(v, IgnoringNaNs) and v.is_marker()):
      final_domain.allows_nan = True

    if isinstance(v, AllowInf):
      final_domain.allows_inf = True

  # 4. Convert back to validators
  return resolved_others + final_domain.to_validators()
