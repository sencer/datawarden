"""Domain constraint solver for validator resolution."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
  from collections.abc import Sequence

  from datawarden.base import Validator

from datawarden.domain import ValidationDomain
from datawarden.exceptions import LogicError
from datawarden.validators.comparison import Ge, Gt, Le, Lt
from datawarden.validators.value import OneOf


def is_domain_validator(
  v: Validator[Any],  # pyright: ignore[reportExplicitAny]
) -> bool:
  """Check if validator can be mapped to a domain."""
  return isinstance(v, (Ge, Gt, Le, Lt, OneOf))


def resolve_domains(
  global_validators: Sequence[Validator[Any]],  # pyright: ignore[reportExplicitAny]
  local_validators: Sequence[Validator[Any]],  # pyright: ignore[reportExplicitAny]
) -> list[Validator[Any]]:  # pyright: ignore[reportExplicitAny]
  """
  Resolve potential conflicts between global and local validators.

  Strategy:
  1. Filter out non-domain validators (return as is).
  2. Convert remaining to ValidationDomain.
  3. Check for internal contradictions (LogicError) within local scope.
  4. Intersect Local with Global.
     - If Intersection is Empty -> Local overrides Global (User Intent).
     - If Intersection is Valid -> Use Intersection.
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
  resolved_others = list(other_map.values())

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

  # 4. Convert back to validators
  return resolved_others + final_domain.to_validators()
