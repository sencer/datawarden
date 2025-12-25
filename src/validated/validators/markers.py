"""Marker validators that opt-out of default validation behaviors."""

from __future__ import annotations

from validated.base import ValidatorMarker


class Nullable(ValidatorMarker):
  """Marker to allow NaN values.

  Deprecated: Strictness is now opt-in.
  This marker is a no-op as NaNs are allowed by default.
  Use NonNaN explicitly if you want to disallow NaNs.
  """


class MaybeEmpty(ValidatorMarker):
  """Marker to allow empty data.

  Deprecated: Strictness is now opt-in.
  This marker is a no-op as empty data is allowed by default.
  Use NonEmpty explicitly if you want to disallow empty data.
  """
