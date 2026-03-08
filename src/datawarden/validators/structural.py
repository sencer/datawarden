from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast, override

import numpy as np
import pandas as pd

from ..common import PandasLike
from .base import (
  SUCCESS,
  BaseValidator,
  Priority,
  ValidationResult,
  ensure_instance,
)
from .numeric import Eq

if TYPE_CHECKING:
  import numpy.typing as npt

  from ..context import ValidationContext

DIM_2D = 2


class IsInstance[T](BaseValidator[PandasLike]):
  """Validates that the data is an instance of a specific type."""

  __slots__ = ("type_",)
  priority = Priority.STRUCTURAL

  def __init__(self, type_: type[T], /) -> None:
    super().__init__()
    self.type_ = type_

  @override
  def validate(self, data: PandasLike, context: ValidationContext) -> ValidationResult:
    del context  # Unused
    if isinstance(data, self.type_):
      return SUCCESS
    return ValidationResult(
      success=False, message=f"Expected {self.type_}, got {type(data)}"
    )

  @override
  def __str__(self) -> str:
    try:
      name = self.type_.__name__
    except AttributeError:
      name = str(self.type_)
    return f"IsInstance({name})"


class Dtype(BaseValidator[PandasLike]):
  """Validates the dtype of a container or its columns."""

  __slots__ = ("dtype",)

  def __init__(self, dtype: object, /) -> None:
    super().__init__()
    self.dtype: np.dtype[Any] = np.dtype(cast("npt.DTypeLike", dtype))

  @property
  @override
  def numba_supported(self) -> bool:
    return True

  @property
  @override
  def numba_fusable(self) -> bool:
    return True

  @override
  def build_numba_expr(
    self,
    target: str,
    parts: list[str],
    arr_name: str = "arr",
    is_range_index: bool = False,
  ) -> None:
    # Dtype check is normally done ahead of time; loop expr is a no-op
    parts.append("True")

  @override
  def validate(self, data: PandasLike, context: ValidationContext) -> ValidationResult:
    del context  # Unused
    if isinstance(data, (pd.Series, pd.Index)):
      if data.dtype == self.dtype:
        return SUCCESS
      return ValidationResult(
        success=False, message=f"Expected dtype {self.dtype}, got {data.dtype}"
      )
    if isinstance(data, pd.DataFrame):
      failed_cols = [col for col in data.columns if data[col].dtype != self.dtype]
      if failed_cols:
        return ValidationResult(
          success=False,
          message=f"Expected columns to be {self.dtype}, failed: {failed_cols}",
        )
      return SUCCESS
    return ValidationResult(
      success=False, message="Unsupported data type for Dtype check"
    )

  @override
  def __str__(self) -> str:
    return f"Dtype({self.dtype})"


class Datetime(Dtype):
  """Ensures data has datetime64[ns] dtype."""

  __slots__ = ()

  def __init__(self) -> None:
    super().__init__("datetime64[ns]")

  @override
  def __str__(self) -> str:
    return "Datetime"


class StructuralValidator[T: PandasLike](BaseValidator[T]):
  """Base class for validators checking container metadata."""

  __slots__ = ()
  priority = Priority.STRUCTURAL


class Index[T: PandasLike](StructuralValidator[T]):
  """Validates the data index.

  Example:
    >>> Index(MonoUp())
  """

  __slots__ = ("validator",)

  def __init__(
    self,
    validator: BaseValidator[pd.Index] | type[BaseValidator[pd.Index]],
    /,
  ) -> None:
    super().__init__()
    self.validator = ensure_instance(validator).clone()

  @property
  @override
  def numba_supported(self) -> bool:
    return self.validator.numba_supported

  @property
  @override
  def uses_index(self) -> bool:
    return True

  @override
  def build_numba_expr(
    self,
    target: str,
    parts: list[str],
    arr_name: str = "arr",
    is_range_index: bool = False,
  ) -> None:
    self.validator.build_numba_expr(
      "idx_val", parts, arr_name="index_arr", is_range_index=is_range_index
    )

  @override
  def validate(self, data: T, context: ValidationContext) -> ValidationResult:
    idx = data.index if hasattr(data, "index") else data
    if not isinstance(idx, pd.Index):
      return ValidationResult(success=False, message="Data has no index")

    res = self.validator.validate(idx, context)
    if res.success:
      return SUCCESS
    return ValidationResult(
      success=False, message=f"Index: {res.message}", mask=res.mask
    )

  @override
  def __str__(self) -> str:
    return f"Index({self.validator})"


class Column(StructuralValidator[pd.DataFrame]):
  """Validates a specific column in a DataFrame."""

  __slots__ = ("column", "validator")

  def __init__(
    self,
    column: str,
    validator: BaseValidator[pd.Series] | type[BaseValidator[pd.Series]],
    /,
  ) -> None:
    super().__init__()
    self.column = column
    self.validator = ensure_instance(validator).clone()

  @override
  def validate(
    self, data: pd.DataFrame, context: ValidationContext
  ) -> ValidationResult:
    if self.column not in data.columns:
      return ValidationResult(success=False, message=f"Column '{self.column}' missing")

    res = self.validator.validate(data[self.column], context)
    if res.success:
      return SUCCESS
    return ValidationResult(
      success=False, message=f"Column('{self.column}'): {res.message}", mask=res.mask
    )

  @override
  def __str__(self) -> str:
    return f"Column({self.column!r}, {self.validator})"


class Columns(StructuralValidator[pd.DataFrame]):
  """Validates multiple DataFrame columns simultaneously."""

  __slots__ = ("validator",)

  def __init__(
    self, validator: BaseValidator[pd.Index] | type[BaseValidator[pd.Index]], /
  ) -> None:
    super().__init__()
    self.validator = ensure_instance(validator).clone()

  @override
  def validate(
    self, data: pd.DataFrame, context: ValidationContext
  ) -> ValidationResult:
    res = self.validator.validate(data.columns, context)
    if res.success:
      return SUCCESS
    return ValidationResult(
      success=False, message=f"Columns: {res.message}", mask=res.mask
    )

  @override
  def __str__(self) -> str:
    return f"Columns({self.validator})"


class Shape[T: (pd.Series, pd.DataFrame, pd.Index)](StructuralValidator[T]):
  """Validates container dimensions (rows, cols)."""

  __slots__ = ("cols", "rows")

  def __init__(
    self,
    rows: int | BaseValidator[int] | None = None,
    cols: int | BaseValidator[int] | None = None,
  ) -> None:
    super().__init__()
    self.rows = self._ensure_validator(rows)
    self.cols = self._ensure_validator(cols)

  @staticmethod
  def _ensure_validator(val: object) -> BaseValidator[int] | None:
    if val is None:
      return None
    if isinstance(val, (int, float)):
      return Eq(cast("float", val))
    return ensure_instance(cast("Any", val))

  @override
  def validate(self, data: T, context: ValidationContext) -> ValidationResult:
    n_rows = len(data)
    n_cols = len(data.columns) if hasattr(data, "columns") else 1

    if self.rows:
      res = self.rows.validate(n_rows, context)  # pyright: ignore[reportAttributeAccessIssue]
      if not res.success:
        return ValidationResult(success=False, message=f"Row count: {res.message}")

    if self.cols:
      res = self.cols.validate(n_cols, context)  # pyright: ignore[reportAttributeAccessIssue]
      if not res.success:
        return ValidationResult(success=False, message=f"Col count: {res.message}")

    return SUCCESS

  @override
  def __str__(self) -> str:
    parts = []
    if self.rows:
      parts.append(f"rows={self.rows}")
    if self.cols:
      parts.append(f"cols={self.cols}")
    return f"Shape({', '.join(parts)})"


class Empty[T: (pd.Series, pd.DataFrame, pd.Index)](StructuralValidator[T]):
  """Validates that a container is empty."""

  __slots__ = ()

  @override
  def validate(self, data: T, context: ValidationContext) -> ValidationResult:
    if data.empty:
      return SUCCESS
    return ValidationResult(success=False, message="Not empty")

  @override
  def __str__(self) -> str:
    return "Empty"


class NotEmpty[T: (pd.Series, pd.DataFrame, pd.Index)](StructuralValidator[T]):
  """Validates that a container is not empty."""

  __slots__ = ()

  @override
  def validate(self, data: T, context: ValidationContext) -> ValidationResult:
    if not data.empty:
      return SUCCESS
    return ValidationResult(success=False, message="Empty")

  @override
  def __str__(self) -> str:
    return "NotEmpty"
