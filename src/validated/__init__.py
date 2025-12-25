"""validated - Pandas validation using Annotated types and decorators."""

__version__ = "0.1.0"

# Base classes
from validated.base import Validated, Validator, ValidatorMarker

# Decorator
from validated.decorator import validated

# All validators
from validated.validators import (
  Datetime,
  Finite,
  Ge,
  Gt,
  HasColumn,
  HasColumns,
  Index,
  IsDtype,
  Le,
  Lt,
  MaxDiff,
  MaxGap,
  MaybeEmpty,
  MonoDown,
  MonoUp,
  NonEmpty,
  NonNaN,
  NonNegative,
  NoTimeGaps,
  Nullable,
  OneOf,
  Positive,
  Shape,
  StrictFinite,
  Unique,
)

__all__ = [
  # Index validators
  "Datetime",
  # Value validators
  "Finite",
  # Comparison validators
  "Ge",
  "Gt",
  "HasColumn",
  "HasColumns",
  "Index",
  # Column validators
  "IsDtype",
  "Le",
  "Lt",
  "MaxDiff",
  "MaxGap",
  "MaybeEmpty",
  "MonoDown",
  "MonoUp",
  # Gap validators
  "NoTimeGaps",
  "NonEmpty",
  "NonNaN",
  "NonNegative",
  # Markers
  "Nullable",
  "OneOf",
  "Positive",
  "Shape",
  "StrictFinite",
  "Unique",
  # Base
  "Validated",
  "Validator",
  "ValidatorMarker",
  # Version
  "__version__",
  # Decorator
  "validated",
]
