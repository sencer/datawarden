from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast, override

import numpy as np
import pandas as pd

from ..common import PandasLike
from .base import SUCCESS, BaseValidator, Priority, ValidationResult
from .numeric import NumericValidator

if TYPE_CHECKING:
  from collections.abc import Callable

  import numpy.typing as npt

  from ..context import ValidationContext

type PredicateResult = (
  bool
  | np.ndarray[tuple[int, ...], np.dtype[np.bool_]]
  | pd.Series[bool]
  | pd.DataFrame
)


class Is[T: PandasLike](BaseValidator[T]):
  __slots__ = ("predicate",)
  priority = Priority.DEFAULT

  def __init__(
    self,
    predicate: Callable[[T], PredicateResult],
    /,
    name: str | None = None,
  ) -> None:
    super().__init__(name or getattr(predicate, "__name__", str(predicate)))
    self.predicate = predicate

  @property
  @override
  def numba_supported(self) -> bool:
    return False

  @override
  def validate(self, data: T, context: ValidationContext) -> ValidationResult:
    del context  # Unused
    try:
      res = self.predicate(data)
    except Exception as e:  # noqa: BLE001 (User predicate can raise anything)
      return ValidationResult(
        success=False, message=f"Is({self.name}) raised exception: {e}"
      )

    # Handle numpy array / pandas object
    if hasattr(res, "all"):
      all_result = res.all()  # pyright: ignore[reportUnknownMemberType,reportAttributeAccessIssue]
      if all_result:  # pyright: ignore[reportUnknownVariableType,reportGeneralTypeIssues]
        return SUCCESS
      # res should be a mask here
      if isinstance(res, (pd.Series, pd.DataFrame)):
        return ValidationResult(
          success=False, message=f"Is({self.name}) failed", mask=res
        )
      return ValidationResult(success=False, message=f"Is({self.name}) failed")

    if res:  # pyright: ignore[reportGeneralTypeIssues] - Truthy check on pandas type
      return SUCCESS

    return ValidationResult(success=False, message=f"Is({self.name}) failed")

  @override
  def __str__(self) -> str:
    return f"Is({self.name})"


class Rows(BaseValidator[pd.DataFrame]):
  __slots__ = ("predicate",)
  priority = Priority.SLOW

  def __init__(
    self, predicate: Callable[[pd.Series[float]], bool], /, name: str | None = None
  ) -> None:
    super().__init__(name or getattr(predicate, "__name__", str(predicate)))
    self.predicate = predicate

  @property
  @override
  def numba_supported(self) -> bool:
    return False

  @override
  def validate(
    self, data: pd.DataFrame, context: ValidationContext
  ) -> ValidationResult:
    del context  # Unused
    try:
      # Result of apply along axis 1 is a Series of results (usually bools)
      mask: pd.Series[bool] = data.apply(self.predicate, axis=1)
    except Exception as e:  # noqa: BLE001 (User predicate can raise anything)
      return ValidationResult(
        success=False, message=f"Rows({self.name}) raised exception: {e}"
      )

    if bool(mask.all()):
      return SUCCESS
    return ValidationResult(
      success=False, message=f"Rows({self.name}) failed", mask=mask
    )

  @override
  def __str__(self) -> str:
    return f"Rows({self.name})"


class Between[T: (pd.Series[float], pd.DataFrame)](NumericValidator[T]):
  __slots__ = ("inclusive", "lower", "upper")

  def __init__(self, lower: float, upper: float, /, inclusive: str = "both") -> None:
    super().__init__()
    self.lower = lower
    self.upper = upper
    self.inclusive = inclusive

  @override
  def _validate_range(self, start: int, stop: int, step: int, length: int) -> bool:
    if length == 0:
      return True
    last = start + (length - 1) * step

    # Check both ends of the range
    if self.inclusive == "both":
      return (start >= self.lower and start <= self.upper) and (
        last >= self.lower and last <= self.upper
      )
    if self.inclusive == "left":
      return (start >= self.lower and start < self.upper) and (
        last >= self.lower and last < self.upper
      )
    if self.inclusive == "right":
      return (start > self.lower and start <= self.upper) and (
        last > self.lower and last <= self.upper
      )
    if self.inclusive == "neither":
      return (start > self.lower and start < self.upper) and (
        last > self.lower and last < self.upper
      )
    return False

  @property
  @override
  def numba_supported(self) -> bool:
    return True

  @override
  def _get_mask_numpy(self, data: npt.NDArray[np.floating]) -> npt.NDArray[np.bool_]:
    if self.inclusive == "both":
      return (data >= self.lower) & (data <= self.upper)
    if self.inclusive == "left":
      return (data >= self.lower) & (data < self.upper)
    if self.inclusive == "right":
      return (data > self.lower) & (data <= self.upper)
    if self.inclusive == "neither":
      return (data > self.lower) & (data < self.upper)
    raise ValueError(f"Unknown inclusive mode: {self.inclusive}")

  @override
  def build_numba_expr(
    self,
    target: str,
    parts: list[str],
    arr_name: str = "arr",
    is_range_index: bool = False,
  ) -> None:
    if self.inclusive == "both":
      parts.append(f"(({target} >= {self.lower}) and ({target} <= {self.upper}))")
    elif self.inclusive == "left":
      parts.append(f"(({target} >= {self.lower}) and ({target} < {self.upper}))")
    elif self.inclusive == "right":
      parts.append(f"(({target} > {self.lower}) and ({target} <= {self.upper}))")
    elif self.inclusive == "neither":
      parts.append(f"(({target} > {self.lower}) and ({target} < {self.upper}))")

  @override
  def __str__(self) -> str:
    return f"Between({self.lower}, {self.upper}, {self.inclusive})"


class Outside[T: (pd.Series[float], pd.DataFrame)](NumericValidator[T]):
  __slots__ = ("inclusive", "lower", "upper")

  def __init__(self, lower: float, upper: float, /, inclusive: str = "neither") -> None:
    super().__init__()
    self.lower = lower
    self.upper = upper
    self.inclusive = inclusive

  @override
  def _validate_range(self, start: int, stop: int, step: int, length: int) -> bool:
    if length == 0:
      return True
    last = start + (length - 1) * step

    # Outside means EVERY value must be outside.
    # For a monotonic range, this means the WHOLE range must be either < lower or > upper.
    if self.inclusive == "neither":
      return (start <= self.lower and last <= self.lower) or (
        start >= self.upper and last >= self.upper
      )
    if self.inclusive == "both":
      return (start < self.lower and last < self.lower) or (
        start > self.upper and last > self.upper
      )
    if self.inclusive == "left":
      return (start < self.lower and last < self.lower) or (
        start >= self.upper and last >= self.upper
      )
    if self.inclusive == "right":
      return (start <= self.lower and last <= self.lower) or (
        start > self.upper and last > self.upper
      )
    return False

  @property
  @override
  def numba_supported(self) -> bool:
    return True

  @override
  def _get_mask_numpy(self, data: npt.NDArray[np.floating]) -> npt.NDArray[np.bool_]:
    if self.inclusive == "neither":
      return (data <= self.lower) | (data >= self.upper)
    if self.inclusive == "both":
      return (data < self.lower) | (data > self.upper)
    if self.inclusive == "left":
      return (data < self.lower) | (data >= self.upper)
    if self.inclusive == "right":
      return (data <= self.lower) | (data > self.upper)
    raise ValueError(f"Unknown inclusive mode: {self.inclusive}")

  @override
  def build_numba_expr(
    self,
    target: str,
    parts: list[str],
    arr_name: str = "arr",
    is_range_index: bool = False,
  ) -> None:
    if self.inclusive == "neither":
      parts.append(f"(({target} <= {self.lower}) or ({target} >= {self.upper}))")
    elif self.inclusive == "both":
      parts.append(f"(({target} < {self.lower}) or ({target} > {self.upper}))")
    elif self.inclusive == "left":
      parts.append(f"(({target} < {self.lower}) or ({target} >= {self.upper}))")
    elif self.inclusive == "right":
      parts.append(f"(({target} <= {self.lower}) or ({target} > {self.upper}))")

  @override
  def __str__(self) -> str:
    return f"Outside({self.lower}, {self.upper}, {self.inclusive})"


class OneOf[T: PandasLike](BaseValidator[T]):
  """Validates that all values are in the given set."""

  __slots__ = ("values",)

  def __init__(self, *values: object) -> None:
    super().__init__()
    self.values: tuple[object, ...] = values

  @override
  def validate(self, data: T, context: ValidationContext) -> ValidationResult:
    del context  # Unused
    vals = data.values
    mask: npt.NDArray[np.bool_] = np.isin(vals, cast("Any", list(self.values)))  # pyright: ignore[reportExplicitAny]

    if mask.all():
      return SUCCESS

    if isinstance(data, pd.Series):
      pd_mask: pd.Series[bool] | pd.DataFrame = pd.Series(
        mask, index=data.index, copy=False
      )
    elif isinstance(data, pd.DataFrame):
      pd_mask = pd.DataFrame(mask, index=data.index, columns=data.columns, copy=False)
    else:
      return ValidationResult(success=False, message=str(self))
    return ValidationResult(success=False, message=str(self), mask=pd_mask)

  @override
  def negate(self) -> NotOneOf[T]:
    return NotOneOf(*self.values)

  @override
  def __str__(self) -> str:
    return f"OneOf({self.values})"


class NotOneOf[T: PandasLike](BaseValidator[T]):
  """Validates that no values are in the given set."""

  __slots__ = ("values",)

  def __init__(self, *values: object) -> None:
    super().__init__()
    self.values: tuple[object, ...] = values

  @override
  def validate(self, data: T, context: ValidationContext) -> ValidationResult:
    del context  # Unused
    vals = data.values
    mask: npt.NDArray[np.bool_] = ~np.isin(vals, cast("Any", list(self.values)))  # pyright: ignore[reportExplicitAny]

    if mask.all():
      return SUCCESS

    if isinstance(data, pd.Series):
      pd_mask: pd.Series[bool] | pd.DataFrame = pd.Series(
        mask, index=data.index, copy=False
      )
    elif isinstance(data, pd.DataFrame):
      pd_mask = pd.DataFrame(mask, index=data.index, columns=data.columns, copy=False)
    else:
      return ValidationResult(success=False, message=str(self))
    return ValidationResult(success=False, message=str(self), mask=pd_mask)

  @override
  def negate(self) -> OneOf[T]:
    return OneOf(*self.values)

  @override
  def __str__(self) -> str:
    return f"NotOneOf({self.values})"
