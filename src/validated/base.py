"""Base classes for validators."""

from __future__ import annotations

from typing import Annotated

# Validated alias for Annotated - the main type hint for validated parameters
Validated = Annotated


class Validator[T]:
  """Base class for validators."""

  def validate(self, data: T) -> T:
    """Validate the data and return it (potentially modified)."""
    return data


class ValidatorMarker:
  """Base class for validator markers that don't perform validation.

  Markers like Nullable and MaybeEmpty are used to opt-out of default
  validation behaviors and don't need a validate method.
  """
