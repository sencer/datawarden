from __future__ import annotations

from abc import ABC, abstractmethod
from contextlib import suppress
import copy
from enum import IntEnum
from typing import TYPE_CHECKING, NamedTuple, Self, override

import numpy as np  # noqa: TC002 (used at runtime for type checks)
import pandas as pd

from ..backends.numba import run_numba_validation
from ..common import NumbaContext, PandasLike
from ..config import get_config

if TYPE_CHECKING:
  import numpy.typing as npt

  from ..common import NumbaContext
  from ..context import ValidationContext


class ValidationResult(NamedTuple):
  success: bool
  message: str | None = None
  mask: pd.Series[bool] | pd.DataFrame | None = None

  def get_summary(self, default_msg: str = "Validation failed") -> str:
    msg = self.message or default_msg
    if (
      not self.success
      and self.mask is not None
      and isinstance(self.mask, (pd.Series, pd.DataFrame))
    ):
      # Count both False and NA as failures.
      # .sum() on boolean-like ignores NA by default, and counts True as 1.
      num_success = self.mask.sum()
      if isinstance(num_success, pd.Series):
        num_success = num_success.sum()
      total = self.mask.size
      num_failures = total - int(num_success)
      msg += f" ({num_failures}/{total} rows failed)"
    return msg


SUCCESS = ValidationResult(success=True)


class Priority(IntEnum):
  STRUCTURAL = 0  # Shape, Empty, NotEmpty
  VECTORIZED = 10  # Numeric comparisons
  COMPLEX = 20  # Monotonicity, Unique
  DEFAULT = 50  # Unknown
  SLOW = 100  # Rows (Python loops)


def ensure_instance[T: PandasLike](
  v: BaseValidator[T] | type[BaseValidator[T]],
) -> BaseValidator[T]:
  if isinstance(v, type) and issubclass(v, BaseValidator):
    return v()
  return v


class BaseValidator[T: PandasLike](ABC):
  __slots__ = ("_numba_key", "complexity", "name")
  priority: int = Priority.DEFAULT

  def __init__(self, name: str | None = None) -> None:
    self.name = name
    self.complexity = 1

  @property
  def numba_supported(self) -> bool:
    """Whether this validator can be accelerated by Numba."""
    return False

  @property
  def uses_index(self) -> bool:
    """Whether this validator requires access to the data index."""
    return False

  @property
  def numba_fusable(self) -> bool:
    """Whether this validator can participate in Numba fusion when composed.

    Typically the same as numba_supported, but some complex validators might
    only be fusable if they are the only ones in the chain.
    """
    return self.numba_supported

  def build_numba_expr(
    self,
    target: str,
    parts: list[str],
    arr_name: str = "arr",
    is_range_index: bool = False,
  ) -> None:
    """Build Numba expression.

    Args:
      target: Name of the variable representing the current value (e.g. 'x' or 'idx_val')
      parts: List to append expression parts to
      arr_name: Name of the array variable being validated (e.g. 'arr' or 'index_arr')
      is_range_index: Whether we are validating a RangeIndex (implies arr_name is not a real array)
    """
    raise NotImplementedError

  def build_numba_expr_column_mode(
    self,
    arr_name: str,
    idx_name: str,
    ctx: NumbaContext,
    parts: list[str],
    is_range_index: bool = False,
  ) -> None:
    """Build Numba expression for column-mode validation.

    Args:
        arr_name: Name of the array variable (e.g., 'arr')
        idx_name: Name of the row index variable (e.g., 'i')
        ctx: NumbaContext with col_map for column offsets
        parts: List to append expression parts to
        is_range_index: Whether the row index is a RangeIndex
    """
    raise NotImplementedError

  def numba_expr(self, target: str) -> str:
    parts: list[str] = []
    self.build_numba_expr(target, parts)
    return "".join(parts)

  @abstractmethod
  def validate(self, data: T, context: ValidationContext) -> ValidationResult: ...

  def _get_mask_numpy(self, data: npt.NDArray[np.floating]) -> npt.NDArray[np.bool_]:
    # Fallback: call validate and extract mask? No, too slow.
    # Most NumericValidators will override this.
    raise NotImplementedError

  def decompose(self) -> list[BaseValidator[T]]:
    return [self]

  def negate(self) -> BaseValidator[T]:
    return Not(self)

  def transform(self) -> list[BaseValidator[T]]:
    return [self]

  def clone(self) -> Self:
    return copy.copy(self)

  @property
  def _state_key(self) -> str:
    """Unique key for storing state in ValidationContext.extra."""
    return f"{self.__class__.__name__.lower()}_{id(self)}"

  def __and__(self, other: BaseValidator[T]) -> And[T]:
    return And(self, other)

  def __or__(self, other: BaseValidator[T]) -> Or[T]:
    return Or(self, other)

  def __invert__(self) -> BaseValidator[T]:
    return self.negate()

  def _get_all_slots(self) -> set[str]:
    slots: set[str] = set()
    for cls in self.__class__.__mro__:
      if hasattr(cls, "__slots__"):
        cls_slots = cls.__slots__
        if isinstance(cls_slots, str):
          slots.add(cls_slots)
        else:
          slots.update(cls_slots)
    return slots

  @override
  def __eq__(self, other: object) -> bool:
    if type(self) is not type(other):
      return False

    slots = self._get_all_slots()
    return all(
      getattr(self, k) == getattr(other, k)
      for k in slots
      if hasattr(self, k) and hasattr(other, k)
    )

  @override
  def __hash__(self) -> int:
    # Hash primary attributes.
    # We can't easily hash everything safely (e.g. lists), but name is basic.
    # Subclasses should override if they have critical state.
    # For now, let's try to grab values from slots that are hashable.
    values: list[object] = [type(self)]
    for k in sorted(self._get_all_slots()):
      if hasattr(self, k):
        val = getattr(self, k)
        # Handle lists/unhashables?
        if isinstance(val, (list, dict, set)):
          # Sort collections to ensure deterministic hash for equivalent objects
          if isinstance(val, dict):
            values.append(tuple(sorted(val.items(), key=str)))
          elif isinstance(val, set):
            values.append(tuple(sorted(val, key=str)))
          else:
            # List is ordered, but safe to tuple-ize for consistency
            values.append(tuple(val))
        else:
          values.append(val)
    return hash(tuple(values))

  def _try_accelerated_validation(
    self, data: T, validators: list[BaseValidator[PandasLike]]
  ) -> ValidationResult | None:
    """Try Numba and Numpy validation paths."""

    cfg = get_config()
    has_values = hasattr(data, "values")

    # Use Numba for large data when enabled
    # Dynamic thresholding based on complexity:
    # - Simple validators (complexity=1): ~20k rows (Numpy is very fast)
    # - Complex chains (complexity>4): ~2k rows (Numba fusion pays off early)
    dynamic_threshold = max(2_000, 20_000 // self.complexity)

    if (
      cfg.use_numba
      and self.numba_supported
      and has_values
      and len(data) >= dynamic_threshold
    ):
      with suppress(Exception):
        success, mask = run_numba_validation(data, validators, cache_obj=self)
        if success:
          return SUCCESS

        if mask is not None:
          if isinstance(data, pd.Series):
            pd_mask: pd.Series[bool] = pd.Series(mask, index=data.index, copy=False)
          else:
            pd_mask = pd.DataFrame(
              mask.reshape(data.shape, order="F"),
              index=data.index,
              columns=data.columns,
              copy=False,
            )
          return ValidationResult(success=False, message=str(self), mask=pd_mask)
        return ValidationResult(success=False, message=str(self))

    # Fallback: numpy path
    if has_values:
      with suppress(NotImplementedError):
        mask = self._get_mask_numpy(data.values)
        if mask.all():
          return SUCCESS
        if isinstance(data, pd.Series):
          pd_mask = pd.Series(mask, index=data.index, copy=False)
        else:
          pd_mask = pd.DataFrame(
            mask,
            index=data.index,
            columns=data.columns,
            copy=False,
          )
        return ValidationResult(success=False, message=str(self), mask=pd_mask)

    return None


class And[T: PandasLike](BaseValidator[T]):
  __slots__ = ("validators",)

  def __init__(
    self, *validators: BaseValidator[T] | type[BaseValidator[T]]
  ) -> None:
    super().__init__()
    # Flatten nested And validators for better debugging
    flattened: list[BaseValidator[T]] = []
    for val in validators:
      instance = ensure_instance(val)
      if isinstance(instance, And):
        flattened.extend(instance.validators)
      else:
        flattened.append(instance)
    # Clone to ensure state isolation
    self.validators: tuple[BaseValidator[T], ...] = tuple(v.clone() for v in flattened)
    self.complexity = max(1, sum(v.complexity for v in self.validators))

  @override
  @override
  def clone(self) -> And[T]:
    # Re-instantiate to trigger deep cloning in __init__
    return And(*self.validators)

  @property
  @override
  def numba_supported(self) -> bool:
    # And can use Numba if all validators support it
    return all(v.numba_supported for v in self.validators)

  @property
  @override
  def uses_index(self) -> bool:
    return any(v.uses_index for v in self.validators)

  @property
  @override
  def numba_fusable(self) -> bool:
    return all(v.numba_fusable for v in self.validators)

  @override
  def build_numba_expr(
    self,
    target: str,
    parts: list[str],
    arr_name: str = "arr",
    is_range_index: bool = False,
  ) -> None:
    parts.append("(")
    for i, v in enumerate(self.validators):
      if i > 0:
        parts.append(" and ")
      v.build_numba_expr(
        target, parts, arr_name=arr_name, is_range_index=is_range_index
      )
    parts.append(")")

  @override
  def build_numba_expr_column_mode(
    self,
    arr_name: str,
    idx_name: str,
    ctx: NumbaContext,
    parts: list[str],
    is_range_index: bool = False,
  ) -> None:
    parts.append("(")
    for i, v in enumerate(self.validators):
      if i > 0:
        parts.append(" and ")
      v.build_numba_expr_column_mode(
        arr_name, idx_name, ctx, parts, is_range_index=is_range_index
      )
    parts.append(")")

  @override
  def _get_mask_numpy(self, data: npt.NDArray[np.floating]) -> npt.NDArray[np.bool_]:
    mask = self.validators[0]._get_mask_numpy(data)
    for v in self.validators[1:]:
      mask &= v._get_mask_numpy(data)
    return mask

  @override
  def validate(self, data: T, context: ValidationContext) -> ValidationResult:
    # Smart Optimization: Split execution if some validators prefer Pandas
    cfg = get_config()
    if data is not None and cfg.use_numba and len(data) >= cfg.numba_threshold:  # pyright: ignore[reportUnnecessaryComparison]
      pandas_preferred = [v for v in self.validators if not v.numba_supported]
      if pandas_preferred:
        # 1. Run Pandas-preferred checks first (e.g. MonoUp)
        for v in pandas_preferred:
          res = v.validate(data, context)
          if not res.success:
            return res

        # 2. Run remaining Numba checks
        numba_supported = [v for v in self.validators if v.numba_supported]
        if not numba_supported:
          return SUCCESS

        remainder = (
          And(*numba_supported) if len(numba_supported) > 1 else numba_supported[0]
        )
        # Recursively validate remainder (will use Numba path)
        return remainder.validate(data, context)

    # Try accelerated paths
    res = self._try_accelerated_validation(data, [self])
    if res is not None:
      return res

    for v in self.validators:
      res = v.validate(data, context)
      if not res.success:
        return res
    return SUCCESS

  @override
  def decompose(self) -> list[BaseValidator[T]]:
    atoms: list[BaseValidator[T]] = []
    for v in self.validators:
      atoms.extend(v.decompose())
    return atoms

  @override
  def __str__(self) -> str:
    return "(" + " & ".join(str(v) for v in self.validators) + ")"

  @override
  def transform(self) -> list[BaseValidator[T]]:
    # Contradiction detection: And(A, ~A) -> Fail
    for v in self.validators:
      if ~v in self.validators:
        return [Fail()]
    return [self]


class Or[T: PandasLike](BaseValidator[T]):
  __slots__ = ("validators",)

  def __init__(
    self, *validators: BaseValidator[T] | type[BaseValidator[T]]
  ) -> None:
    super().__init__()
    # Flatten nested Or validators for better debugging
    flattened: list[BaseValidator[T]] = []
    for val in validators:
      instance = ensure_instance(val)
      if isinstance(instance, Or):
        flattened.extend(instance.validators)
      else:
        flattened.append(instance)
    # Clone to ensure state isolation
    self.validators: tuple[BaseValidator[T], ...] = tuple(v.clone() for v in flattened)
    self.complexity = max(1, sum(v.complexity for v in self.validators))

  @override
  def clone(self) -> Or[T]:
    # Re-instantiate to trigger deep cloning in __init__
    return Or(*self.validators)

  @property
  @override
  def numba_supported(self) -> bool:
    return all(v.numba_supported for v in self.validators)

  @property
  @override
  def uses_index(self) -> bool:
    return any(v.uses_index for v in self.validators)

  @property
  @override
  def numba_fusable(self) -> bool:
    return all(v.numba_fusable for v in self.validators)

  @override
  def build_numba_expr(
    self,
    target: str,
    parts: list[str],
    arr_name: str = "arr",
    is_range_index: bool = False,
  ) -> None:
    parts.append("(")
    for i, v in enumerate(self.validators):
      if i > 0:
        parts.append(" or ")
      v.build_numba_expr(
        target, parts, arr_name=arr_name, is_range_index=is_range_index
      )
    parts.append(")")

  @override
  def build_numba_expr_column_mode(
    self,
    arr_name: str,
    idx_name: str,
    ctx: NumbaContext,
    parts: list[str],
    is_range_index: bool = False,
  ) -> None:
    parts.append("(")
    for i, v in enumerate(self.validators):
      if i > 0:
        parts.append(" or ")
      v.build_numba_expr_column_mode(
        arr_name, idx_name, ctx, parts, is_range_index=is_range_index
      )
    parts.append(")")

  @override
  def _get_mask_numpy(self, data: npt.NDArray[np.floating]) -> npt.NDArray[np.bool_]:
    mask = self.validators[0]._get_mask_numpy(data)
    for v in self.validators[1:]:
      mask |= v._get_mask_numpy(data)
    return mask

  @override
  def validate(self, data: T, context: ValidationContext) -> ValidationResult:
    # Try accelerated paths
    res = self._try_accelerated_validation(data, [self])
    if res is not None:
      return res

    results: list[ValidationResult] = []
    for v in self.validators:
      res = v.validate(data, context)
      if res.success:
        return SUCCESS
      results.append(res)

    # Try to combine masks if all children provided one
    if all(r.mask is not None for r in results):
      combined_mask = results[0].mask
      for r in results[1:]:
        m1, m2 = combined_mask, r.mask
        # Handle broadcasting: Series | DataFrame
        if isinstance(m1, pd.DataFrame) and isinstance(m2, pd.Series):
          m2 = pd.DataFrame(dict.fromkeys(m1.columns, m2), index=m1.index, copy=False)
        elif isinstance(m1, pd.Series) and isinstance(m2, pd.DataFrame):
          m1 = pd.DataFrame(dict.fromkeys(m2.columns, m1), index=m2.index, copy=False)

        combined_mask = m1 | m2

      if combined_mask is not None and bool(combined_mask.values.all()):
        return SUCCESS
      return ValidationResult(success=False, message=str(self), mask=combined_mask)

    return ValidationResult(success=False, message=str(self))

  @override
  def transform(self) -> list[BaseValidator[T]]:
    # Tautology detection: Or(A, ~A) -> Pass
    for v in self.validators:
      if ~v in self.validators:
        return [Pass()]
    return [self]

  @override
  def __str__(self) -> str:
    return "(" + " | ".join(str(v) for v in self.validators) + ")"


class Not[T: PandasLike](BaseValidator[T]):
  __slots__ = ("validator",)

  def __init__(
    self, validator: BaseValidator[T] | type[BaseValidator[T]], /
  ) -> None:
    super().__init__()
    self.validator = ensure_instance(validator).clone()
    self.complexity = self.validator.complexity

  @override
  def clone(self) -> Not[T]:
    return Not(self.validator)

  @override
  def __str__(self) -> str:
    return f"~{self.validator}"

  @property
  @override
  def numba_supported(self) -> bool:
    return self.validator.numba_supported

  @property
  @override
  def numba_fusable(self) -> bool:
    return self.validator.numba_fusable

  @override
  def build_numba_expr(
    self,
    target: str,
    parts: list[str],
    arr_name: str = "arr",
    is_range_index: bool = False,
  ) -> None:
    parts.append("(not (")
    self.validator.build_numba_expr(
      target, parts, arr_name=arr_name, is_range_index=is_range_index
    )
    parts.append("))")

  @override
  def build_numba_expr_column_mode(
    self,
    arr_name: str,
    idx_name: str,
    ctx: NumbaContext,
    parts: list[str],
    is_range_index: bool = False,
  ) -> None:
    parts.append("(not (")
    self.validator.build_numba_expr_column_mode(
      arr_name, idx_name, ctx, parts, is_range_index=is_range_index
    )
    parts.append("))")

  @override
  def _get_mask_numpy(self, data: npt.NDArray[np.floating]) -> npt.NDArray[np.bool_]:
    return ~self.validator._get_mask_numpy(data)

  @override
  def validate(self, data: T, context: ValidationContext) -> ValidationResult:
    # Try accelerated paths
    res = self._try_accelerated_validation(data, [self])
    if res is not None:
      return res

    res = self.validator.validate(data, context)
    if not res.success:
      return SUCCESS
    return ValidationResult(success=False, message=f"Not({self.validator}) failed")


class Pass[T: PandasLike](BaseValidator[T]):
  __slots__ = ()

  @override
  def validate(self, data: T, context: ValidationContext) -> ValidationResult:
    del data, context  # Unused
    return SUCCESS


class Fail[T: PandasLike](BaseValidator[T]):
  __slots__ = ()

  @override
  def validate(self, data: T, context: ValidationContext) -> ValidationResult:
    del data, context  # Unused
    return ValidationResult(success=False, message="Always fail")
