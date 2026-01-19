from __future__ import annotations

from typing import TYPE_CHECKING, cast, override

import numpy as np
import pandas as pd

from ..common import PandasLike
from .base import (
  SUCCESS,
  And,
  BaseValidator,
  Pass,
  Priority,
  ValidationResult,
  ensure_instance,
)

if TYPE_CHECKING:
  import numpy.typing as npt

  from ..context import ValidationContext

DIM_2D = 2


class IsInstance[T](BaseValidator[PandasLike]):
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


class Column(BaseValidator[pd.DataFrame]):
  __slots__ = ("column_name", "validator")
  priority = Priority.STRUCTURAL

  def __init__(
    self,
    name: str,
    /,
    *validators: BaseValidator[pd.Series[float]],
  ) -> None:
    super().__init__(name)
    self.column_name = name
    # Clone validators to ensure state isolation (Prototype Pattern)
    cloned: list[BaseValidator[pd.Series[float]]] = [
      ensure_instance(v).clone() for v in validators
    ]
    if not cloned:
      self.validator: BaseValidator[pd.Series[float]] = Pass()
    elif len(cloned) == 1:
      self.validator = cloned[0]
    else:
      self.validator = And(*cloned)

  @override
  def validate(
    self, data: pd.DataFrame, context: ValidationContext
  ) -> ValidationResult:
    # Optimized path: avoid Series creation
    # Optimization: Direct access by name is much faster than iloc
    try:
      vals: npt.NDArray[np.floating] = data[self.name].values
    except KeyError:
      return ValidationResult(success=False, message=f"Column '{self.name}' not found")
    try:
      mask = self.validator._get_mask_numpy(vals)
      if mask.all():
        return SUCCESS
      pd_mask: pd.Series[bool] = pd.Series(mask, index=data.index, copy=False)
      return ValidationResult(
        success=False,
        message=f"Column '{self.name}' failed: {self.validator}",
        mask=pd_mask,
      )
    except NotImplementedError:
      # Fallback to standard validation
      res = self.validator.validate(data[self.name], context)
      if not res.success:
        return ValidationResult(
          success=False,
          message=f"Column '{self.name}' failed: {res.message}",
          mask=res.mask,
        )
      return SUCCESS

  @override
  def decompose(self) -> list[BaseValidator[pd.DataFrame]]:
    inner_atoms = self.validator.decompose()
    res: list[BaseValidator[pd.DataFrame]] = [
      Column(cast("str", self.name), atom) for atom in inner_atoms
    ]
    return res

  @override
  def __str__(self) -> str:
    return f"Column('{self.name}', {self.validator})"


class Columns(And[pd.DataFrame]):
  __slots__ = ()

  def __init__(
    self,
    names: list[str],
    /,
    *validators: BaseValidator[pd.Series[float]],
  ) -> None:
    cols: list[Column] = [Column(name, *validators) for name in names]
    super().__init__(*cols)


type DtypeSupported = pd.Series[float] | pd.DataFrame | pd.Index[float]


class Dtype(BaseValidator[DtypeSupported]):
  __slots__ = ("dtype",)

  def __init__(self, dtype: object, /) -> None:
    super().__init__()
    self.dtype: np.dtype[np.floating[object]] = np.dtype(dtype)

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
    # Dtype is usually checked upfront, inside Numba loop it's a no-op
    parts.append("True")

  @override
  def validate(
    self, data: DtypeSupported, context: ValidationContext
  ) -> ValidationResult:
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
          message=f"Expected all columns to be {self.dtype}, failed: {failed_cols}",
        )
      return SUCCESS
    # Unreachable for properly typed Validated[]
    return ValidationResult(success=False, message="Not a Series or DataFrame")

  @override
  def __str__(self) -> str:
    return f"Dtype({self.dtype})"


class Datetime(Dtype):
  __slots__ = ()

  def __init__(self) -> None:
    super().__init__("datetime64[ns]")

  @override
  def __str__(self) -> str:
    return "Datetime"


class Shape(BaseValidator["pd.Series[float] | pd.DataFrame"]):
  __slots__ = ("cols", "rows")
  priority = Priority.STRUCTURAL

  def __init__(self, rows: int | None = None, cols: int | None = None) -> None:
    super().__init__()
    self.rows = rows
    self.cols = cols

  @override
  def validate(
    self, data: pd.Series[float] | pd.DataFrame, context: ValidationContext
  ) -> ValidationResult:
    del context  # Unused
    shape = data.shape
    current_rows = shape[0]
    if self.rows is not None and current_rows != self.rows:
      return ValidationResult(
        success=False, message=f"Expected {self.rows} rows, got {current_rows}"
      )

    if self.cols is not None:
      if len(shape) < DIM_2D:
        return ValidationResult(
          success=False,
          message=f"Expected {self.cols} cols, but object is 1D",
        )
      current_cols = shape[1]
      if current_cols != self.cols:
        return ValidationResult(
          success=False,
          message=f"Expected {self.cols} cols, got {current_cols}",
        )

    return SUCCESS

  @override
  def __str__(self) -> str:
    return f"Shape(rows={self.rows}, cols={self.cols})"


class NotEmpty(BaseValidator["pd.Series[float] | pd.DataFrame"]):
  __slots__ = ()
  priority = Priority.STRUCTURAL

  @override
  def validate(
    self, data: pd.Series[float] | pd.DataFrame, context: ValidationContext
  ) -> ValidationResult:
    del context  # Unused
    if data.shape[0] > 0:
      return SUCCESS
    return ValidationResult(success=False, message="Data is empty")

  @override
  def __str__(self) -> str:
    return "NotEmpty"


class Empty(BaseValidator["pd.Series[float] | pd.DataFrame"]):
  __slots__ = ()
  priority = Priority.STRUCTURAL

  @override
  def validate(
    self, data: pd.Series[float] | pd.DataFrame, context: ValidationContext
  ) -> ValidationResult:
    del context  # Unused
    if data.shape[0] == 0:
      return SUCCESS
    return ValidationResult(success=False, message="Data is not empty")

  @override
  def __str__(self) -> str:
    return "Empty"


class Index(BaseValidator["pd.Series[float] | pd.DataFrame"]):
  __slots__ = ("validator",)
  priority = Priority.STRUCTURAL

  def __init__(self, *validators: BaseValidator[pd.Index[float]]) -> None:
    super().__init__()
    # Clone validators to ensure state isolation (Prototype Pattern)
    cloned: list[BaseValidator[pd.Index[float]]] = [
      ensure_instance(v).clone() for v in validators
    ]
    self.validator: BaseValidator[pd.Index[float]] = (
      And(*cloned) if len(cloned) > 1 else cloned[0]
    )
    self.complexity = self.validator.complexity

  @property
  @override
  def numba_supported(self) -> bool:
    return self.validator.numba_supported

  @property
  @override
  def numba_fusable(self) -> bool:
    return self.validator.numba_fusable

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
    # We use idx_val which is provided by the JIT template.
    # We pass arr_name='index_arr' so that inner validators (like MonoUp)
    # reference the correct array.
    self.validator.build_numba_expr(
      "idx_val", parts, arr_name="index_arr", is_range_index=is_range_index
    )

  @override
  def build_numba_expr_column_mode(
    self,
    arr_name: str,
    idx_name: str,
    ctx: object,
    parts: list[str],
    is_range_index: bool = False,
  ) -> None:
    # Column mode also provides idx_val in the loop
    self.validator.build_numba_expr(
      "idx_val", parts, arr_name="index_arr", is_range_index=is_range_index
    )

  @override
  def validate(
    self, data: pd.Series[float] | pd.DataFrame, context: ValidationContext
  ) -> ValidationResult:
    res = self.validator.validate(data.index, context)
    if not res.success:
      return ValidationResult(
        success=False,
        message=f"Index validation failed: {res.message}",
        mask=res.mask,
      )
    return SUCCESS

  @override
  def decompose(self) -> list[BaseValidator[pd.Series[float] | pd.DataFrame]]:
    inner_atoms = self.validator.decompose()
    return [Index(atom) for atom in inner_atoms]

  @override
  def __str__(self) -> str:
    return f"Index({self.validator})"
