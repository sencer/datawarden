"""datawarden - Pandas validation using Annotated types and decorators."""

__version__ = "0.1.0"

# Base classes
from datawarden.base import Validated, Validator

# Decorator
from datawarden.decorator import validate

# Exceptions
from datawarden.exceptions import LogicError

# All validators
from datawarden.validators import (
  Between,
  Datetime,
  Finite,
  Ge,
  Gt,
  HasColumn,
  HasColumns,
  IgnoringNaNs,
  Index,
  Is,
  IsDtype,
  Le,
  Lt,
  MaxDiff,
  MaxGap,
  MonoDown,
  MonoUp,
  NonEmpty,
  NonNaN,
  NonNegative,
  NoTimeGaps,
  OneOf,
  Positive,
  Rows,
  Shape,
  StrictFinite,
  Unique,
)

__all__ = [
  # Range validators
  "Between",
  # Index validators
  "Datetime",
  # Value validators
  "Finite",
  # Comparison validators
  "Ge",
  "Gt",
  "HasColumn",
  "HasColumns",
  "IgnoringNaNs",
  "Index",
  # Lambda validators
  "Is",
  # Column validators
  "IsDtype",
  "Le",
  # Exceptions
  "LogicError",
  "Lt",
  "MaxDiff",
  "MaxGap",
  "MonoDown",
  "MonoUp",
  # Gap validators
  "NoTimeGaps",
  "NonEmpty",
  "NonNaN",
  "NonNegative",
  "OneOf",
  "Positive",
  "Rows",
  "Shape",
  "StrictFinite",
  "Unique",
  # Base
  "Validated",
  "Validator",
  # Version
  "__version__",
  # Decorator
  "validate",
]
