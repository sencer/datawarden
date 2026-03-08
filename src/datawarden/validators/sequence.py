from __future__ import annotations

import datetime
from typing import TYPE_CHECKING, override

import numpy as np
import pandas as pd

from .base import SUCCESS, BaseValidator, Priority, ValidationResult
from .numeric import NumericValidator

if TYPE_CHECKING:
  import numpy.typing as npt

  from ..context import ValidationContext

MIN_LEN_FOR_DIFF = 2


class MaxDiff[T: (pd.Series[float], pd.DataFrame)](NumericValidator[T]):
  """Validates that consecutive differences do not exceed `value`."""

  __slots__ = ("value",)
  priority = Priority.VECTORIZED

  def __init__(self, value: float, /) -> None:
    super().__init__()
    self.value = value

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
    if is_range_index and arr_name == "index_arr":
      parts.append(f"((i == 0) or (abs(r_step) <= {self.value}))")
    else:
      parts.append(f"((i == 0) or (abs({target} - {arr_name}[i-1]) <= {self.value}))")

  @override
  def _get_mask_numpy(self, data: npt.NDArray[np.floating]) -> npt.NDArray[np.bool_]:
    if len(data) == 0:
      return np.array([], dtype=np.bool_)

    has_nans = np.isnan(data).any()

    if data.ndim == 1:
      filled = pd.Series(data, copy=False).ffill().values if has_nans else data
      diffs = np.abs(np.diff(filled))
      mask = np.empty(len(data), dtype=np.bool_)
      mask[0] = True
      mask[1:] = diffs <= self.value
    else:
      filled = pd.DataFrame(data, copy=False).ffill().values if has_nans else data
      diffs = np.abs(np.diff(filled, axis=0))
      mask = np.empty(data.shape, dtype=np.bool_)
      mask[0, :] = True
      mask[1:, :] = diffs <= self.value

    if has_nans:
      mask &= ~np.isnan(data)
    return mask

  @override
  def __str__(self) -> str:
    return f"MaxDiff({self.value})"


class MinDiff[T: (pd.Series[float], pd.DataFrame)](NumericValidator[T]):
  """Validates that consecutive differences are at least `value`."""

  __slots__ = ("value",)
  priority = Priority.VECTORIZED

  def __init__(self, value: float, /) -> None:
    super().__init__()
    self.value = value

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
    if is_range_index and arr_name == "index_arr":
      parts.append(f"((i == 0) or (abs(r_step) >= {self.value}))")
    else:
      parts.append(f"((i == 0) or (abs({target} - {arr_name}[i-1]) >= {self.value}))")

  @override
  def _get_mask_numpy(self, data: npt.NDArray[np.floating]) -> npt.NDArray[np.bool_]:
    if len(data) == 0:
      return np.array([], dtype=np.bool_)

    has_nans = np.isnan(data).any()

    if data.ndim == 1:
      filled = pd.Series(data, copy=False).ffill().values if has_nans else data
      diffs = np.abs(np.diff(filled))
      mask = np.empty(len(data), dtype=np.bool_)
      mask[0] = True
      mask[1:] = diffs >= self.value
    else:
      filled = pd.DataFrame(data, copy=False).ffill().values if has_nans else data
      diffs = np.abs(np.diff(filled, axis=0))
      mask = np.empty(data.shape, dtype=np.bool_)
      mask[0, :] = True
      mask[1:, :] = diffs >= self.value

    if has_nans:
      mask &= ~np.isnan(data)
    return mask

  @override
  def __str__(self) -> str:
    return f"MinDiff({self.value})"


class Unique[T: (pd.Series[float], pd.Index[float])](BaseValidator[T]):
  """Validates that all values are unique."""

  __slots__ = ()
  priority = Priority.COMPLEX

  @override
  def validate(self, data: T, context: ValidationContext) -> ValidationResult:
    del context  # Unused
    if data.is_unique:
      return SUCCESS

    if isinstance(data, pd.Index):
      duplicated: pd.Series[bool] = pd.Series(
        data.duplicated(keep=False), index=data, copy=False
      )
    else:
      duplicated = data.duplicated(keep=False)

    return ValidationResult(
      success=False, message="Values are not unique", mask=~duplicated
    )

  @override
  def __str__(self) -> str:
    return "Unique"


class StateValidator[T: (pd.Series[object], pd.Index)](BaseValidator[T]):
  """Base class for validators that maintain state across chunks.

  These validators use `ValidationContext` to store the last value from a previous
  chunk to ensure continuity (e.g., monotonicity) across streaming data.
  """

  __slots__ = ()

  def _get_last_val(self, context: ValidationContext) -> object | None:
    return context.extra.get(self._state_key)

  def _set_last_val(self, context: ValidationContext, val: object) -> None:
    context.extra[self._state_key] = val


class MonoUp[T: (pd.Series[float], pd.Index)](StateValidator[T]):
  """Validates that a sequence is monotonically increasing.

  Supports stateful validation across chunks when using `ValidationContext`.
  """

  __slots__ = ("strict",)

  def __init__(self, strict: bool = False, name: str | None = None) -> None:
    super().__init__(name)
    self.strict = strict
    self.complexity = 1

  @property
  @override
  def numba_supported(self) -> bool:
    return not self.strict  # Numba loop supports non-strict by default

  @override
  def build_numba_expr(
    self,
    target: str,
    parts: list[str],
    arr_name: str = "arr",
    is_range_index: bool = False,
  ) -> None:
    if is_range_index and arr_name == "index_arr":
      parts.append("((i == 0) or (r_step >= 0))")
    else:
      parts.append(f"((i == 0) or ({arr_name}[i-1] <= {target}))")

  @override
  def validate(self, data: T, context: ValidationContext) -> ValidationResult:
    if len(data) == 0:
      return SUCCESS

    if self.strict:
      success = bool((data.to_series().diff().dropna() > 0).all())
    else:
      success = bool(data.is_monotonic_increasing)

    if not success:
      return ValidationResult(success=False, message=str(self))

    last_val = self._get_last_val(context)
    if last_val is not None:
      first_val = data[0] if isinstance(data, pd.Index) else data.iloc[0]
      cross_success = (first_val > last_val) if self.strict else (first_val >= last_val)
      if not cross_success:
        return ValidationResult(
          success=False, message=f"{self} failed cross-chunk continuity"
        )

    self._set_last_val(
      context, data[-1] if isinstance(data, pd.Index) else data.iloc[-1]
    )
    return SUCCESS

  @override
  def __str__(self) -> str:
    return "MonoUp(strict=True)" if self.strict else "MonoUp"


class MonoDown[T: (pd.Series[float], pd.Index)](StateValidator[T]):
  """Validates that a sequence is monotonically decreasing.

  Supports stateful validation across chunks.
  """

  __slots__ = ("strict",)

  def __init__(self, strict: bool = False, name: str | None = None) -> None:
    super().__init__(name)
    self.strict = strict

  @property
  @override
  def numba_supported(self) -> bool:
    return not self.strict

  @override
  def build_numba_expr(
    self,
    target: str,
    parts: list[str],
    arr_name: str = "arr",
    is_range_index: bool = False,
  ) -> None:
    if is_range_index and arr_name == "index_arr":
      parts.append("((i == 0) or (r_step <= 0))")
    else:
      parts.append(f"((i == 0) or ({arr_name}[i-1] >= {target}))")

  @override
  def validate(self, data: T, context: ValidationContext) -> ValidationResult:
    if len(data) == 0:
      return SUCCESS

    if self.strict:
      success = bool((data.to_series().diff().dropna() < 0).all())
    else:
      success = bool(data.is_monotonic_decreasing)

    if not success:
      return ValidationResult(success=False, message=str(self))

    last_val = self._get_last_val(context)
    if last_val is not None:
      first_val = data[0] if isinstance(data, pd.Index) else data.iloc[0]
      cross_success = (first_val < last_val) if self.strict else (first_val <= last_val)
      if not cross_success:
        return ValidationResult(
          success=False, message=f"{self} failed cross-chunk continuity"
        )

    self._set_last_val(
      context, data[-1] if isinstance(data, pd.Index) else data.iloc[-1]
    )
    return SUCCESS

  @override
  def __str__(self) -> str:
    return "MonoDown(strict=True)" if self.strict else "MonoDown"


class NoTimeGaps[T: (pd.Series[datetime.datetime], pd.DatetimeIndex)](
  StateValidator[T]
):
  """Validates that a datetime sequence has no gaps relative to a frequency."""

  __slots__ = ("freq",)

  def __init__(self, freq: str | pd.DateOffset, name: str | None = None) -> None:
    super().__init__(name)
    self.freq = freq

  @override
  def validate(self, data: T, context: ValidationContext) -> ValidationResult:
    if len(data) == 0:
      return SUCCESS

    actual_diffs = (
      pd.Series(data).diff().dropna()
      if isinstance(data, pd.DatetimeIndex)
      else data.diff().dropna()
    )
    expected_diff = pd.to_timedelta(
      self.freq if isinstance(self.freq, str) else self.freq.kwds.get("nanos", 0)
    )

    if not (actual_diffs == expected_diff).all():
      return ValidationResult(
        success=False, message=f"Time gaps detected (freq={self.freq})"
      )

    last_time = self._get_last_val(context)
    if last_time is not None:
      first_time = data[0] if isinstance(data, pd.DatetimeIndex) else data.iloc[0]
      if (first_time - last_time) != expected_diff:
        return ValidationResult(
          success=False, message="Time gap between chunks detected"
        )

    self._set_last_val(
      context, data[-1] if isinstance(data, pd.DatetimeIndex) else data.iloc[-1]
    )
    return SUCCESS

  @override
  def __str__(self) -> str:
    return f"NoTimeGaps({self.freq})"


class MaxGap[T: (pd.Series[datetime.datetime], pd.DatetimeIndex)](StateValidator[T]):
  """Validates that no time gap exceeds a maximum duration."""

  __slots__ = ("limit",)

  def __init__(self, limit: str | pd.Timedelta, name: str | None = None) -> None:
    super().__init__(name)
    self.limit = pd.to_timedelta(limit)

  @override
  def validate(self, data: T, context: ValidationContext) -> ValidationResult:
    if len(data) == 0:
      return SUCCESS

    actual_diffs = (
      pd.Series(data).diff().dropna()
      if isinstance(data, pd.DatetimeIndex)
      else data.diff().dropna()
    )
    if (actual_diffs > self.limit).any():
      return ValidationResult(success=False, message=f"Gap exceeds {self.limit}")

    last_time = self._get_last_val(context)
    if last_time is not None:
      first_time = data[0] if isinstance(data, pd.DatetimeIndex) else data.iloc[0]
      if (first_time - last_time) > self.limit:
        return ValidationResult(success=False, message="Cross-chunk gap violation")

    self._set_last_val(
      context, data[-1] if isinstance(data, pd.DatetimeIndex) else data.iloc[-1]
    )
    return SUCCESS

  @override
  def __str__(self) -> str:
    return f"MaxGap({self.limit})"
