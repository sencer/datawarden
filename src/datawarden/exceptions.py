from typing import TYPE_CHECKING, override

if TYPE_CHECKING:
  from .validators.base import ValidationResult


class ValidationError(Exception):
  __slots__ = ("details", "results")

  def __init__(
    self,
    message: str,
    details: list[str] | None = None,
    results: dict[str, list["ValidationResult"]] | None = None,
  ) -> None:
    super().__init__(message)
    self.details = details or []
    self.results = results or {}

  @override
  def __str__(self) -> str:
    msg = super().__str__()
    if self.details:
      return f"{msg}:\n" + "\n".join(self.details)
    return msg


class LogicError(Exception):
  """Raised when validation logic is inherently contradictory."""

  __slots__ = ()
  pass
