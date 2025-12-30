"""Domain logic for validation constraints."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
  from datawarden.base import Validator

from datawarden.validators.comparison import Ge, Gt, Le, Lt
from datawarden.validators.value import (
  AllowInf,
  AllowNaN,
  Between,
  Finite,
  IgnoringNaNs,
  NonNaN,
  NonNegative,
  OneOf,
  Positive,
  StrictFinite,
)


@dataclass
class ValidationDomain:
  """Represents the domain of valid values for a column."""

  min_val: float | int | None = None
  max_val: float | int | None = None
  min_inclusive: bool = True
  max_inclusive: bool = True
  allowed_values: set[Any] | None = None
  # Disallowed values (like inf, NaN) are implicitly handled by the absence
  # from domain or explicit checks, but for intersection we focus on ranges and sets.
  # We might need to track "no NaN" separately if we want to merge that logic.
  allows_nan: bool = True
  allows_inf: bool = True

  def is_empty(self) -> bool:
    """Check if the domain is logically impossible."""
    if self.min_val is not None and self.max_val is not None:
      if self.min_val > self.max_val:
        return True
      if self.min_val == self.max_val and not (
        self.min_inclusive and self.max_inclusive
      ):
        return True

    return bool(self.allowed_values is not None and not self.allowed_values)

  def intersect(self, other: ValidationDomain) -> ValidationDomain:
    """Return the intersection of this domain with another."""
    new_domain = ValidationDomain()

    # Intersect allowable values (sets)
    if self.allowed_values is None and other.allowed_values is None:
      new_domain.allowed_values = None
    elif self.allowed_values is not None and other.allowed_values is None:
      new_domain.allowed_values = self.allowed_values.copy()
    elif self.allowed_values is None and other.allowed_values is not None:
      new_domain.allowed_values = other.allowed_values.copy()
    else:
      # Both have sets - strict intersection
      # ignoring type for set intersection of optional sets (handled by ifs)
      new_domain.allowed_values = self.allowed_values & other.allowed_values  # type: ignore

    # Intersect ranges
    # Min value: max(min1, min2)
    new_min = self.min_val
    new_min_inc = self.min_inclusive

    if other.min_val is not None:
      if new_min is None or other.min_val > new_min:
        new_min = other.min_val
        new_min_inc = other.min_inclusive
      elif other.min_val == new_min:
        # If values equal, be exclusive if either is exclusive
        new_min_inc = new_min_inc and other.min_inclusive

    new_domain.min_val = new_min
    new_domain.min_inclusive = new_min_inc

    # Max value: min(max1, max2)
    new_max = self.max_val
    new_max_inc = self.max_inclusive

    if other.max_val is not None:
      if new_max is None or other.max_val < new_max:
        new_max = other.max_val
        new_max_inc = other.max_inclusive
      elif other.max_val == new_max:
        new_max_inc = new_max_inc and other.max_inclusive

    new_domain.max_val = new_max
    new_domain.max_inclusive = new_max_inc

    # Flags
    new_domain.allows_nan = self.allows_nan and other.allows_nan
    new_domain.allows_inf = self.allows_inf and other.allows_inf

    # Apply Range constraints to AllowedSet if both exist
    if new_domain.allowed_values is not None and (
      new_domain.min_val is not None or new_domain.max_val is not None
    ):
      filtered = set()
      for x in new_domain.allowed_values:
        valid = True
        if new_domain.min_val is not None:
          if new_domain.min_inclusive:
            if x < new_domain.min_val:
              valid = False
          elif x <= new_domain.min_val:
            valid = False

        if valid and new_domain.max_val is not None:
          if new_domain.max_inclusive:
            if x > new_domain.max_val:
              valid = False
          elif x >= new_domain.max_val:
            valid = False

        if valid:
          filtered.add(x)
      new_domain.allowed_values = filtered

    return new_domain

  def is_subset(self, other: ValidationDomain) -> bool:
    """Check if self is a subset of other."""
    # Simple implementation for now
    # 1. Check flags
    if self.allows_nan and not other.allows_nan:
      return False
    if self.allows_inf and not other.allows_inf:
      return False

    # 2. Check sets
    if other.allowed_values is not None:
      if self.allowed_values is None:
        return False
      if not self.allowed_values.issubset(other.allowed_values):
        return False

    # 3. Check Ranges
    # If other has a min limit, self must respect it
    if other.min_val is not None:
      if self.min_val is None:
        return False
      if self.min_val < other.min_val:
        return False
      if (
        self.min_val == other.min_val and not other.min_inclusive and self.min_inclusive
      ):
        return False

    # If other has a max limit, self must respect it
    if other.max_val is not None:
      if self.max_val is None:
        return False
      if self.max_val > other.max_val:
        return False
      if (
        self.max_val == other.max_val and not other.max_inclusive and self.max_inclusive
      ):
        return False

    return True

  @classmethod
  def from_validator(
    cls,
    v: Validator[Any],
  ) -> ValidationDomain:
    """Convert a Validator instance to a Domain."""
    domain = cls()

    if isinstance(v, Ge):
      # Handle unary only for now, multi-target resolved elsewhere or ignored here
      if len(v.targets) == 1 and isinstance(v.targets[0], (int, float)):
        domain.min_val = v.targets[0]
        domain.min_inclusive = True
        domain.allows_nan = False
    elif isinstance(v, Gt):
      if len(v.targets) == 1 and isinstance(v.targets[0], (int, float)):
        domain.min_val = v.targets[0]
        domain.min_inclusive = False
        domain.allows_nan = False
    elif isinstance(v, Le):
      if len(v.targets) == 1 and isinstance(v.targets[0], (int, float)):
        domain.max_val = v.targets[0]
        domain.max_inclusive = True
        domain.allows_nan = False
    elif isinstance(v, Lt):
      if len(v.targets) == 1 and isinstance(v.targets[0], (int, float)):
        domain.max_val = v.targets[0]
        domain.max_inclusive = False
        domain.allows_nan = False
    elif isinstance(v, OneOf):
      domain.allowed_values = v.allowed
      domain.allows_nan = False
    elif isinstance(v, Between):
      domain.min_val = v.lower
      domain.max_val = v.upper
      domain.min_inclusive = v.lower_inclusive
      domain.max_inclusive = v.upper_inclusive
      domain.allows_nan = False
    elif isinstance(v, Positive):
      domain.min_val = 0
      domain.min_inclusive = False
      domain.allows_nan = False
    elif isinstance(v, NonNegative):
      domain.min_val = 0
      domain.min_inclusive = True
      domain.allows_nan = False
    elif isinstance(v, Finite):
      domain.allows_inf = False
    elif isinstance(v, StrictFinite):
      domain.allows_inf = False
      domain.allows_nan = False
    elif isinstance(v, NonNaN):
      domain.allows_nan = False
    elif isinstance(v, (IgnoringNaNs, AllowNaN)):
      if isinstance(v, IgnoringNaNs) and v.wrapped is not None:
        domain = cls.from_validator(v.wrapped)
      domain.allows_nan = True
    elif isinstance(v, AllowInf):
      domain.allows_inf = True

    # If validator has ignore_nan=True, allow NaNs
    if getattr(v, "ignore_nan", False):
      domain.allows_nan = True

    return domain

  def to_validators(self) -> list[Validator[Any]]:
    """Convert domain back to validators."""
    vals: list[Validator[Any]] = []

    if self.allowed_values is not None:
      vals.append(OneOf(*tuple(self.allowed_values)))

    if self.min_val is not None:
      if self.min_inclusive:
        vals.append(Ge(self.min_val))
      else:
        vals.append(Gt(self.min_val))

    if self.max_val is not None:
      if self.max_inclusive:
        vals.append(Le(self.max_val))
      else:
        vals.append(Lt(self.max_val))

    # Apply Flags Logic
    if self.allows_nan:
      # If NaNs allowed, wrap strict validators in IgnoringNaNs
      wrapped_vals: list[Validator[Any]] = [
        IgnoringNaNs(v, _check_syntax=False) for v in vals
      ]
      vals = wrapped_vals

      if not self.allows_inf:
        # Finite already allows NaN, no need to wrap in IgnoringNaNs
        vals.append(Finite())
    # Strict mode (NaNs not allowed)
    # vals are already strict (Ge/Le check for NaN and fail)

    elif not self.allows_inf:
      vals.append(StrictFinite())
    else:
      vals.append(NonNaN())

    return vals
