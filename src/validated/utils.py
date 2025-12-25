"""Utility functions from validated."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from validated.base import Validator, ValidatorMarker

if TYPE_CHECKING:
  from collections.abc import Sequence


def instantiate_validator(
  item: object,
) -> Validator[Any] | ValidatorMarker | None:  # pyright: ignore[reportExplicitAny]
  """Helper to instantiate a validator from a type or instance.

  Args:
    item: The object to check and potentially instantiate.

  Returns:
    The validator instance if item is a Validator or Validator subclass,
    otherwise None.
  """
  if isinstance(item, (Validator, ValidatorMarker)):
    return item
  if isinstance(item, type) and issubclass(item, (Validator, ValidatorMarker)):
    try:
      return item()  # type: ignore[return-value]
    except TypeError:
      pass
  return None


def apply_default_validators(
  validators: Sequence[Validator[Any] | ValidatorMarker],  # pyright: ignore[reportExplicitAny]
) -> list[Validator[Any]]:  # pyright: ignore[reportExplicitAny]
  """Filter markers and return explicit validators.

  Note: Defaults (NonNaN, NonEmpty) are no longer applied implicitly.
  Markers like Nullable/MaybeEmpty are ignored as strictness is now opt-in.

  Args:
    validators: Sequence of validators and markers.

  Returns:
    List of active validators.
  """
  return [v for v in validators if isinstance(v, Validator)]
