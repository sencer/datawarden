from __future__ import annotations

from typing import TYPE_CHECKING, cast, overload, override

import numpy as np
import pandas as pd

from ..backends.numba import run_numba_validation_column_mode
from ..config import get_config
from .base import SUCCESS, BaseValidator, Priority, ValidationResult

if TYPE_CHECKING:
  from collections.abc import Sequence

  import numpy.typing as npt

  from ..common import NumbaContext
  from ..context import ValidationContext

  type NumericType = (
    int
    | float
    | "np.ndarray[tuple[int, ...], np.dtype[np.floating]]"
    | "pd.Series[float]"
  )
  type BooleanResult = (
    bool | "np.ndarray[tuple[int, ...], np.dtype[np.bool_]]" | "pd.Series[bool]"
  )


class NumericValidator[T: (pd.Series[float], pd.DataFrame, pd.Index)](BaseValidator[T]):
  __slots__ = ("targets",)
  priority = Priority.VECTORIZED

  def __init__(
    self, name: str | None = None, targets: Sequence[str] | None = None
  ) -> None:
    super().__init__(name)
    self.targets = targets

  def _validate_range(self, start: int, stop: int, step: int, length: int) -> bool:
    """O(1) check for RangeIndex values. Return True if all valid, False or raise for fallback."""
    raise NotImplementedError

  @override
  def _get_mask_numpy(self, data: npt.NDArray[np.floating]) -> npt.NDArray[np.bool_]:
    raise NotImplementedError

    vals = data.values
    mask = self._get_mask_numpy(vals)

    if self._check_mask_success(mask):
      return SUCCESS

    return ValidationResult(
      success=False, message=str(self), mask=self._build_error_mask(data, mask)
    )

  @staticmethod
  def _check_mask_success(mask: npt.NDArray[np.bool_]) -> bool:
    # High-performance branching: type() is faster than isinstance() for exact matches
    mask_type = type(mask)
    if mask_type is np.ndarray:
      return bool(mask.all())

    if hasattr(mask, "all"):
      # Pandas ExtensionArrays (like BooleanArray) default to skipna=True.
      # We force skipna=False to ensure nulls are treated as failures.
      try:
        res = mask.all(skipna=False)
        # In pandas, True is Success, False or <NA> is Failure.
        return res is True or res is np.True_
      except TypeError:
        # Fallback for types that don't support skipna
        return bool(mask.all())

    return bool(mask)

  @staticmethod
  def _build_error_mask[T: (pd.Series[float], pd.DataFrame, pd.Index)](
    data: T, mask: npt.NDArray[np.bool_]
  ) -> pd.Series[bool] | pd.DataFrame | None:
    # On failure, we might want a Pandas mask for better error reporting
    if isinstance(data, pd.Series):
      return pd.Series(mask, index=data.index, copy=False)

    if isinstance(data, pd.DataFrame):
      return pd.DataFrame(mask, index=data.index, columns=data.columns, copy=False)

    if isinstance(data, pd.Index):
      return pd.Series(mask, index=data, copy=False)

    return None

  @override
  def validate(self, data: T, context: ValidationContext) -> ValidationResult:  # noqa: PLR0911
    del context  # Unused
    # Fast path for RangeIndex - O(1) bounds check
    if isinstance(data, pd.RangeIndex):
      try:
        if self._validate_range(data.start, data.stop, data.step, len(data)):
          return SUCCESS
        return ValidationResult(success=False, message=str(self))
      except NotImplementedError:
        pass

    # Optimization: Use Numba if possible
    if self.numba_supported and get_config().use_numba:
      try:
        # Single-target numeric validation via Numba
        # This expects a Series or a 1-column DataFrame/Index
        # We use None as column map because it's single-column mode
        success, mask = run_numba_validation_column_mode(
          data, [self], None, len(data), cache_obj=self
        )
        if success:
          return SUCCESS
        if mask is not None:
          return ValidationResult(
            success=False, message=str(self), mask=self._build_error_mask(data, mask)
          )
      except (RuntimeError, ValueError, TypeError, ImportError, AttributeError):
        pass

    # Standard path: NumPy vectorization
    try:
      vals = data.values
      mask = self._get_mask_numpy(vals)

      if self._check_mask_success(mask):
        return SUCCESS

      return ValidationResult(
        success=False, message=str(self), mask=self._build_error_mask(data, mask)
      )
    except NotImplementedError:
      return ValidationResult(success=False, message="Validation logic not implemented")


class ComparisonValidator[T: (pd.Series[float], pd.DataFrame, pd.Index)](
  NumericValidator[T]
):
  __slots__ = ("value",)
  op_str: str

  @overload
  def __init__(self, target: float, /) -> None: ...

  @overload
  def __init__(self, target: str, versus: str, /, *extra_targets: str) -> None: ...

  def __init__(self, target: str | float, *args: str) -> None:
    super().__init__()
    if args:
      self.targets = (str(target), *args)
      self.value = None
    else:
      self.value = float(target)
      self.targets = None

  @property
  @override
  def numba_supported(self) -> bool:
    return True

  def _compare(self, a: NumericType, b: NumericType) -> BooleanResult:
    raise NotImplementedError

  @override
  def _get_mask_numpy(self, data: npt.NDArray[np.floating]) -> npt.NDArray[np.bool_]:
    if self.targets is None:
      return self._compare(data, self.value)
    raise NotImplementedError("N-ary comparison uses validate override")

  @override
  def build_numba_expr_column_mode(
    self,
    arr_name: str,
    idx_name: str,
    ctx: NumbaContext,
    parts: list[str],
    is_range_index: bool = False,
  ) -> None:
    """Build column-mode Numba expression for multi-column comparison.

    For Ge['a', 'b'], generates: (arr[i] >= arr[i + n_rows * 1])
    For Ge['a', 'b', 'c'], generates: (arr[i] >= arr[i + n_rows * 1]) and (arr[i + n_rows * 1] >= arr[i + n_rows * 2])
    """
    if self.targets is None:
      # Scalar mode - use regular expression
      parts.append(f"({arr_name}[{idx_name}] {self.op_str} {self.value})")
      return

    # Multi-column mode: generate chained comparison expressions
    # ctx.col_map maps column name to column index

    parts.append("(")
    for i in range(1, len(self.targets)):
      if i > 1:
        parts.append(" and ")
      col_a = self.targets[i - 1]
      col_b = self.targets[i]
      offset_a = ctx.col_offset(col_a)
      offset_b = ctx.col_offset(col_b)
      parts.append(
        f"({arr_name}[{idx_name}{offset_a}] {self.op_str} {arr_name}[{idx_name}{offset_b}])"
      )
    parts.append(")")

  @override
  def validate(self, data: T, context: ValidationContext) -> ValidationResult:  # noqa: PLR0911
    if self.targets is None:
      return super().validate(data, context)

    if not isinstance(data, pd.DataFrame):
      return ValidationResult(
        success=False,
        message=f"{self.__class__.__name__} multi-column requires a DataFrame",
      )

    # Check if columns exist
    missing = [t for t in self.targets if t not in data.columns]
    if missing:
      return ValidationResult(success=False, message=f"Missing columns: {missing}")

    # Try Numba acceleration for column-mode validation
    cfg = get_config()
    n_rows = len(data)

    if cfg.use_numba and n_rows >= cfg.numba_threshold:
      try:
        # Build column map using actual column positions in the DataFrame
        # This avoids copying the DataFrame - we use the original raveled data
        cols_list = list(data.columns)
        col_map = {col: cols_list.index(col) for col in self.targets}
        success, mask = run_numba_validation_column_mode(
          data, [self], col_map, n_rows, cache_obj=self
        )
        if success:
          return SUCCESS
        if mask is not None:
          pd_mask: pd.Series[bool] = pd.Series(mask, index=data.index, copy=False)
          return ValidationResult(success=False, message=str(self), mask=pd_mask)
        # Fall through to pandas path if mask not available
      except (
        RuntimeError,
        ValueError,
        TypeError,
        ImportError,
        AttributeError,
      ):
        pass  # Fall through to pandas path

    # Fallback: N-ary comparison with pandas
    # Ge("a", "b", "c") validates a >= b AND b >= c
    mask_result: pd.Series[bool] = pd.Series(True, index=data.index)
    for i in range(1, len(self.targets)):
      res = self._compare(data[self.targets[i - 1]], data[self.targets[i]])
      mask_result &= res

    if bool(mask_result.all()):
      return SUCCESS

    return ValidationResult(success=False, message=str(self), mask=mask_result)

  @override
  def __str__(self) -> str:
    if self.targets is None:
      return f"{self.op_str}{self.value}"
    return f"{self.__class__.__name__}({', '.join(map(repr, self.targets))})"


# Max targets for smart negation of N-ary comparisons
MAX_SMART_NEGATION_TARGETS = 2


class Gt[T: (pd.Series[float], pd.DataFrame, pd.Index)](ComparisonValidator[T]):
  __slots__ = ()
  op_str = ">"

  @override
  def _validate_range(self, start: int, stop: int, step: int, length: int) -> bool:
    if length == 0:
      return True
    last = start + (length - 1) * step
    return (start > self.value) and (last > self.value)  # pyright: ignore[reportOperatorIssue]

  @override
  def build_numba_expr(
    self,
    target: str,
    parts: list[str],
    arr_name: str = "arr",
    is_range_index: bool = False,
  ) -> None:
    parts.append(f"({target} > {self.value})")

  @override
  def _compare(self, a: NumericType, b: NumericType) -> BooleanResult:
    return a > b

  @override
  def negate(self) -> Le[T]:
    if self.targets is None:
      return Le(self.value)
    if len(self.targets) == MAX_SMART_NEGATION_TARGETS:
      return Le(*self.targets)
    return cast("Le[T]", super().negate())


class Ge[T: (pd.Series[float], pd.DataFrame, pd.Index)](ComparisonValidator[T]):
  __slots__ = ()
  op_str = ">="

  @override
  def _validate_range(self, start: int, stop: int, step: int, length: int) -> bool:
    if length == 0:
      return True
    last = start + (length - 1) * step
    return (start >= self.value) and (last >= self.value)  # pyright: ignore[reportOperatorIssue]

  @override
  def build_numba_expr(
    self,
    target: str,
    parts: list[str],
    arr_name: str = "arr",
    is_range_index: bool = False,
  ) -> None:
    parts.append(f"({target} >= {self.value})")

  @override
  def _compare(self, a: NumericType, b: NumericType) -> BooleanResult:
    return a >= b

  @override
  def negate(self) -> Lt[T]:
    if self.targets is None:
      return Lt(self.value)
    if len(self.targets) == MAX_SMART_NEGATION_TARGETS:
      return Lt(*self.targets)
    return cast("Lt[T]", super().negate())


class Lt[T: (pd.Series[float], pd.DataFrame, pd.Index)](ComparisonValidator[T]):
  __slots__ = ()
  op_str = "<"

  @override
  def _validate_range(self, start: int, stop: int, step: int, length: int) -> bool:
    if length == 0:
      return True
    last = start + (length - 1) * step
    return (start < self.value) and (last < self.value)  # pyright: ignore[reportOperatorIssue]

  @override
  def build_numba_expr(
    self,
    target: str,
    parts: list[str],
    arr_name: str = "arr",
    is_range_index: bool = False,
  ) -> None:
    parts.append(f"({target} < {self.value})")

  @override
  def _compare(self, a: NumericType, b: NumericType) -> BooleanResult:
    return a < b

  @override
  def negate(self) -> Ge[T]:
    if self.targets is None:
      return Ge(self.value)
    if len(self.targets) == MAX_SMART_NEGATION_TARGETS:
      return Ge(*self.targets)
    return cast("Ge[T]", super().negate())


class Le[T: (pd.Series[float], pd.DataFrame, pd.Index)](ComparisonValidator[T]):
  __slots__ = ()
  op_str = "<="

  @override
  def _validate_range(self, start: int, stop: int, step: int, length: int) -> bool:
    if length == 0:
      return True
    last = start + (length - 1) * step
    return (start <= self.value) and (last <= self.value)  # pyright: ignore[reportOperatorIssue]

  @override
  def build_numba_expr(
    self,
    target: str,
    parts: list[str],
    arr_name: str = "arr",
    is_range_index: bool = False,
  ) -> None:
    parts.append(f"({target} <= {self.value})")

  @override
  def _compare(self, a: NumericType, b: NumericType) -> BooleanResult:
    return a <= b

  @override
  def negate(self) -> Gt[T]:
    if self.targets is None:
      return Gt(self.value)
    if len(self.targets) == MAX_SMART_NEGATION_TARGETS:
      return Gt(*self.targets)
    return cast("Gt[T]", super().negate())


class Eq[T: (pd.Series[float], pd.DataFrame, pd.Index)](NumericValidator[T]):
  __slots__ = ("value",)

  def __init__(self, value: object, /) -> None:
    super().__init__()
    self.value = value

  @property
  @override
  def numba_supported(self) -> bool:
    return True

  @override
  def build_numba_expr(
    self,
    target: str,
    parts: list[str],
    arr_name: str = "arr",
    is_range_index: bool = False,
  ) -> None:
    if not isinstance(self.value, (int, float, bool, np.number)):
      raise TypeError(
        f"Numba acceleration only supports numeric values, got {type(self.value)}"
      )
    parts.append(f"({target} == {self.value!r})")

  @override
  def _get_mask_numpy(self, data: npt.NDArray[np.floating]) -> npt.NDArray[np.bool_]:
    return data == self.value

  @override
  def negate(self) -> Ne[T]:
    return Ne(self.value)

  @override
  def __str__(self) -> str:
    return f"=={self.value}"


class Ne[T: (pd.Series[float], pd.DataFrame, pd.Index)](NumericValidator[T]):
  __slots__ = ("value",)

  def __init__(self, value: object, /) -> None:
    super().__init__()
    self.value = value

  @property
  @override
  def numba_supported(self) -> bool:
    return True

  @override
  def build_numba_expr(
    self,
    target: str,
    parts: list[str],
    arr_name: str = "arr",
    is_range_index: bool = False,
  ) -> None:
    if not isinstance(self.value, (int, float, bool, np.number)):
      raise TypeError(
        f"Numba acceleration only supports numeric values, got {type(self.value)}"
      )
    parts.append(f"({target} != {self.value!r})")

  @override
  def _get_mask_numpy(self, data: npt.NDArray[np.floating]) -> npt.NDArray[np.bool_]:
    return data != self.value

  @override
  def negate(self) -> Eq[T]:
    return Eq(self.value)

  @override
  def __str__(self) -> str:
    return f"!={self.value}"


class IsNaN[T: (pd.Series[float], pd.DataFrame, pd.Index)](NumericValidator[T]):
  __slots__ = ()

  @override
  def _validate_range(self, start: int, stop: int, step: int, length: int) -> bool:
    return False

  @property
  @override
  def numba_supported(self) -> bool:
    return True

  @override
  def build_numba_expr(
    self,
    target: str,
    parts: list[str],
    arr_name: str = "arr",
    is_range_index: bool = False,
  ) -> None:
    parts.append(f"np.isnan({target})")

  @override
  def _get_mask_numpy(self, data: npt.NDArray[np.floating]) -> npt.NDArray[np.bool_]:
    return np.isnan(data)

  @override
  def __str__(self) -> str:
    return "IsNaN"

  @override
  def negate(self) -> NumericValidator[T]:
    return NotNaN()


class NotNaN[T: (pd.Series[float], pd.DataFrame, pd.Index)](NumericValidator[T]):
  __slots__ = ()

  @override
  def _validate_range(self, start: int, stop: int, step: int, length: int) -> bool:
    return True

  @property
  @override
  def numba_supported(self) -> bool:
    return True

  @override
  def build_numba_expr(
    self,
    target: str,
    parts: list[str],
    arr_name: str = "arr",
    is_range_index: bool = False,
  ) -> None:
    parts.append(f"not np.isnan({target})")

  @override
  def _get_mask_numpy(self, data: npt.NDArray[np.floating]) -> npt.NDArray[np.bool_]:
    # np.isnan is faster than pd.notna for numpy arrays
    return ~np.isnan(data)

  @override
  def negate(self) -> NumericValidator[T]:
    return IsNaN()

  @override
  def __str__(self) -> str:
    return "NotNaN"


# Aliases for v1 compatibility
NotIsNaN = NotNaN


class Positive[T: (pd.Series[float], pd.DataFrame, pd.Index)](Gt[T]):
  __slots__ = ()

  def __init__(self) -> None:
    super().__init__(0)

  @override
  def negate(self) -> NonPositive[T]:
    return NonPositive()

  @override
  def __str__(self) -> str:
    return "Positive"


class NonPositive[T: (pd.Series[float], pd.DataFrame, pd.Index)](Le[T]):
  __slots__ = ()

  def __init__(self) -> None:
    super().__init__(0)

  @override
  def negate(self) -> Positive[T]:
    return Positive()

  @override
  def __str__(self) -> str:
    return "NonPositive"


class Negative[T: (pd.Series[float], pd.DataFrame, pd.Index)](Lt[T]):
  __slots__ = ()

  def __init__(self) -> None:
    super().__init__(0)

  @override
  def negate(self) -> NonNegative[T]:
    return NonNegative()

  @override
  def __str__(self) -> str:
    return "Negative"


class NonNegative[T: (pd.Series[float], pd.DataFrame, pd.Index)](Ge[T]):
  __slots__ = ()

  def __init__(self) -> None:
    super().__init__(0)

  @override
  def negate(self) -> Negative[T]:
    return Negative()

  @override
  def __str__(self) -> str:
    return "NonNegative"


class Finite[T: (pd.Series[float], pd.DataFrame, pd.Index)](NumericValidator[T]):
  __slots__ = ()

  @override
  def _validate_range(self, start: int, stop: int, step: int, length: int) -> bool:
    return True

  @property
  @override
  def numba_supported(self) -> bool:
    return True

  @override
  def build_numba_expr(
    self,
    target: str,
    parts: list[str],
    arr_name: str = "arr",
    is_range_index: bool = False,
  ) -> None:
    parts.append(f"np.isfinite({target})")

  @override
  def _get_mask_numpy(self, data: npt.NDArray[np.floating]) -> npt.NDArray[np.bool_]:
    return np.isfinite(data)

  @override
  def __str__(self) -> str:
    return "Finite"


class Infinite[T: (pd.Series[float], pd.DataFrame, pd.Index)](NumericValidator[T]):
  __slots__ = ()

  @property
  @override
  def numba_supported(self) -> bool:
    return True

  @override
  def build_numba_expr(
    self,
    target: str,
    parts: list[str],
    arr_name: str = "arr",
    is_range_index: bool = False,
  ) -> None:
    parts.append(f"np.isinf({target})")

  @override
  def _get_mask_numpy(self, data: npt.NDArray[np.floating]) -> npt.NDArray[np.bool_]:
    return np.isinf(data)

  @override
  def __str__(self) -> str:
    return "Infinite"
