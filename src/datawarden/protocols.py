"""Protocols for datawarden validators."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
  from datawarden.base import PandasData, Validator, VectorizedResult


@runtime_checkable
class ScalarConstraint(Protocol):
  """Protocol for objects that can constrain a scalar value."""

  def check(self, value: object) -> bool:
    """Check if the value satisfies the constraint."""
    ...

  def describe(self) -> str:
    """Return a string description of the constraint (e.g. '>= 10')."""
    ...


@runtime_checkable
class VectorizedValidator(Protocol):
  """Protocol for validators that support vectorized operations."""

  def validate_vectorized(self, data: PandasData) -> VectorizedResult:
    """Validate all values in vectorized data."""
    ...


@runtime_checkable
class MetaValidator(Protocol):
  """Protocol for validators that transform other validators.

  They can modify, unwrap, or replace validators in the chain.
  """

  def transform(self) -> list[Validator[Any]]:
    """Transform this validator into one or more concrete validators.

    Returns:
      A list of validators to replace this one in the chain.
    """
    ...
