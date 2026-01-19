from __future__ import annotations

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
    # Per-element diff check: first element passes, rest check |arr[i] - arr[i-1]| <= value
    if is_range_index and arr_name == "index_arr":
      parts.append(f"((i == 0) or (abs(r_step) <= {self.value}))")
    else:
      parts.append(f"((i == 0) or (abs({target} - {arr_name}[i-1]) <= {self.value}))")

  @override
  def _get_mask_numpy(self, data: npt.NDArray[np.floating]) -> npt.NDArray[np.bool_]:
    if len(data) == 0:
      return np.array([], dtype=np.bool_)

    # Optimize: avoid copy if no NaNs are present
    has_nans = np.isnan(data).any()

    if data.ndim == 1:
      filled = pd.Series(data, copy=False).ffill().values if has_nans else data
      diffs = np.abs(np.diff(filled))
      mask = np.empty(len(data), dtype=np.bool_)
      mask[0] = True
      mask[1:] = diffs <= self.value
    else:
      # For DataFrame (2D), diff along rows
      filled = pd.DataFrame(data, copy=False).ffill().values if has_nans else data
      diffs = np.abs(np.diff(filled, axis=0))
      mask = np.empty(data.shape, dtype=np.bool_)
      mask[0, :] = True
      mask[1:, :] = diffs <= self.value

    # Reject NaNs by default (they will be allowed if wrapped in Or(..., IsNaN))
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
    # Per-element diff check: first element passes, rest check |arr[i] - arr[i-1]| >= value
    if is_range_index and arr_name == "index_arr":
      parts.append(f"((i == 0) or (abs(r_step) >= {self.value}))")
    else:
      parts.append(f"((i == 0) or (abs({target} - {arr_name}[i-1]) >= {self.value}))")

  @override
  def _get_mask_numpy(self, data: npt.NDArray[np.floating]) -> npt.NDArray[np.bool_]:
    if len(data) == 0:
      return np.array([], dtype=np.bool_)

    # Optimize
    has_nans = np.isnan(data).any()

    if data.ndim == 1:
      filled = pd.Series(data, copy=False).ffill().values if has_nans else data
      diffs = np.abs(np.diff(filled))
      mask = np.empty(len(data), dtype=np.bool_)
      mask[0] = True
      mask[1:] = diffs >= self.value
    else:
      # For DataFrame (2D), diff along rows
      filled = pd.DataFrame(data, copy=False).ffill().values if has_nans else data
      diffs = np.abs(np.diff(filled, axis=0))
      mask = np.empty(data.shape, dtype=np.bool_)
      mask[0, :] = True
      mask[1:, :] = diffs >= self.value

    # Reject NaNs by default
    if has_nans:
      mask &= ~np.isnan(data)
    return mask

  @override
  def __str__(self) -> str:
    return f"MinDiff({self.value})"


class NoTimeGaps(BaseValidator["pd.Series[float] | pd.Index[float]"]):
  __slots__ = ("freq",)
  priority = Priority.COMPLEX

  def __init__(self, freq: object, /) -> None:
    super().__init__()
    self.freq = pd.to_timedelta(freq) if isinstance(freq, str) else freq

  @property
  @override
  def _state_key(self) -> str:
    return f"notimegaps_{id(self)}_last_ts"

  @override
  def validate(
    self, data: pd.Series[float] | pd.Index[float], context: ValidationContext
  ) -> ValidationResult:
    vals = data.values

    if not np.issubdtype(vals.dtype, np.datetime64):
      return ValidationResult(
        success=False, message="NoTimeGaps requires datetime64 data"
      )

    if len(vals) == 0:
      return SUCCESS

    # Convert to nanoseconds int64 for faster diff
    ts: npt.NDArray[np.int64] = vals.view(np.int64)

    last_ts = context.extra.get(self._state_key)
    expected_ns = (
      int(self.freq.total_seconds() * 1e9)  # pyright: ignore[reportAttributeAccessIssue]
      if hasattr(self.freq, "total_seconds")
      else int(self.freq)  # pyright: ignore[reportArgumentType]
    )

    if last_ts is not None:
      # Check gap between chunks
      first_gap = ts[0] - last_ts
      if first_gap != expected_ns:
        return ValidationResult(
          success=False,
          message=f"Time gap found between chunks (expected {self.freq})",
        )

    if len(vals) < MIN_LEN_FOR_DIFF:
      context.extra[self._state_key] = ts[-1]
      return SUCCESS

    diffs = np.diff(ts)
    mask_diffs = diffs == expected_ns

    context.extra[self._state_key] = ts[-1]

    if mask_diffs.all():
      return SUCCESS

    # Create a full mask
    full_mask = np.empty(len(vals), dtype=np.bool_)
    full_mask[0] = True
    full_mask[1:] = mask_diffs

    pd_mask: pd.Series[bool] = pd.Series(
      full_mask,
      index=data if isinstance(data, pd.Index) else data.index,
      copy=False,
    )

    return ValidationResult(
      success=False,
      message=f"Time gaps found (expected exactly {self.freq})",
      mask=pd_mask,
    )

  @override
  def __str__(self) -> str:
    return f"NoTimeGaps({self.freq})"


class MaxGap(BaseValidator["pd.Series[float] | pd.Index[float]"]):
  __slots__ = ("duration",)
  priority = Priority.COMPLEX

  def __init__(self, duration: object, /) -> None:
    super().__init__()
    self.duration = pd.to_timedelta(duration) if isinstance(duration, str) else duration

  @property
  @override
  def _state_key(self) -> str:
    return f"maxgap_{id(self)}_last_ts"

  @override
  def validate(
    self, data: pd.Series[float] | pd.Index[float], context: ValidationContext
  ) -> ValidationResult:
    vals = data.values

    if not np.issubdtype(vals.dtype, np.datetime64):
      return ValidationResult(success=False, message="MaxGap requires datetime64 data")

    if len(vals) == 0:
      return SUCCESS

    ts: npt.NDArray[np.int64] = vals.view(np.int64)
    max_ns = (
      int(self.duration.total_seconds() * 1e9)  # pyright: ignore[reportAttributeAccessIssue]
      if hasattr(self.duration, "total_seconds")
      else int(self.duration)  # pyright: ignore[reportArgumentType]
    )

    last_ts = context.extra.get(self._state_key)
    if last_ts is not None and (ts[0] - last_ts) > max_ns:
      return ValidationResult(
        success=False,
        message=f"Time gap found between chunks (max {self.duration})",
      )

    if len(vals) < MIN_LEN_FOR_DIFF:
      context.extra[self._state_key] = ts[-1]
      return SUCCESS

    diffs = np.diff(ts)
    mask_diffs = diffs <= max_ns

    context.extra[self._state_key] = ts[-1]

    if mask_diffs.all():
      return SUCCESS

    full_mask = np.empty(len(vals), dtype=np.bool_)
    full_mask[0] = True
    full_mask[1:] = mask_diffs

    pd_mask: pd.Series[bool] = pd.Series(
      full_mask,
      index=data if isinstance(data, pd.Index) else data.index,
      copy=False,
    )

    return ValidationResult(
      success=False,
      message=f"Time gap larger than {self.duration} found",
      mask=pd_mask,
    )

  @override
  def __str__(self) -> str:
    return f"MaxGap({self.duration})"


class Unique[T: (pd.Series[float], pd.Index[float])](BaseValidator[T]):
  __slots__ = ()
  priority = Priority.COMPLEX

  @override
  def validate(self, data: T, context: ValidationContext) -> ValidationResult:
    del context  # Unused
    if data.is_unique:
      return SUCCESS

    # Finding duplicated rows for the mask
    # duplicated() returns True for duplicates (Fail). We want ~duplicated for Pass.
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


class MonoUp(BaseValidator["pd.Series[float] | pd.Index[float]"]):
  __slots__ = ()
  priority = Priority.COMPLEX

  @property
  @override
  def numba_supported(self) -> bool:
    return True

  @property
  @override
  def numba_fusable(self) -> bool:
    # Can participate in Numba fusion when composed with other validators.
    return True

  @override
  def build_numba_expr(
    self,
    target: str,
    parts: list[str],
    arr_name: str = "arr",
    is_range_index: bool = False,
  ) -> None:
    # Per-element monotonicity check: first element passes, rest check arr[i-1] <= arr[i]
    # Using 'arr' and 'i' which are the loop variables in the generated kernel
    if is_range_index and arr_name == "index_arr":
      parts.append("((i == 0) or (r_step >= 0))")
    else:
      parts.append(f"((i == 0) or ({arr_name}[i-1] <= {target}))")

  @property
  @override
  def _state_key(self) -> str:
    return f"monoup_{id(self)}_last"

  @override
  def validate(
    self, data: pd.Series[float] | pd.Index[float], context: ValidationContext
  ) -> ValidationResult:
    if len(data) == 0:
      return SUCCESS

    vals = data.values
    last_val = context.extra.get(self._state_key)
    if last_val is not None and vals[0] < last_val:
      return ValidationResult(
        success=False, message="Monotonicity broken between chunks"
      )

    if data.is_monotonic_increasing:
      context.extra[self._state_key] = vals[-1]
      return SUCCESS

    # Per-element mask for better error reporting and logic combination
    # Use ffill to ignore NaNs for monotonicity check
    has_nans = np.isnan(vals).any()
    filled = pd.Series(vals, copy=False).ffill().values if has_nans else vals
    mask = np.empty(len(vals), dtype=np.bool_)
    mask[0] = True
    mask[1:] = filled[1:] >= filled[:-1]  # pyright: ignore[reportOperatorIssue]

    # Reject NaNs by default (allowed if wrapped in Or(..., IsNaN))
    if has_nans:
      mask &= ~np.isnan(vals)

    context.extra[self._state_key] = vals[-1]
    pd_mask = pd.Series(mask, index=data if isinstance(data, pd.Index) else data.index)
    return ValidationResult(
      success=False, message="Not monotonically increasing", mask=pd_mask
    )

  @override
  def __str__(self) -> str:
    return "MonoUp"


class MonoDown(BaseValidator["pd.Series[float] | pd.Index[float]"]):
  __slots__ = ()
  priority = Priority.COMPLEX

  @property
  @override
  def numba_supported(self) -> bool:
    return True

  @property
  @override
  def numba_fusable(self) -> bool:
    # Can participate in Numba fusion when composed with other validators.
    return True

  @override
  def build_numba_expr(
    self,
    target: str,
    parts: list[str],
    arr_name: str = "arr",
    is_range_index: bool = False,
  ) -> None:
    # Per-element monotonicity check: first element passes, rest check arr[i-1] >= arr[i]
    if is_range_index and arr_name == "index_arr":
      parts.append("((i == 0) or (r_step <= 0))")
    else:
      parts.append(f"((i == 0) or ({arr_name}[i-1] >= {target}))")

  @property
  @override
  def _state_key(self) -> str:
    return f"monodown_{id(self)}_last"

  @override
  def validate(
    self, data: pd.Series[float] | pd.Index[float], context: ValidationContext
  ) -> ValidationResult:
    if len(data) == 0:
      return SUCCESS

    vals = data.values
    last_val = context.extra.get(self._state_key)
    if last_val is not None and vals[0] > last_val:
      return ValidationResult(
        success=False, message="Monotonicity broken between chunks"
      )

    if data.is_monotonic_decreasing:
      context.extra[self._state_key] = vals[-1]
      return SUCCESS

    # Per-element mask
    # Use ffill to ignore NaNs
    has_nans = np.isnan(vals).any()
    filled = pd.Series(vals, copy=False).ffill().values if has_nans else vals
    mask = np.empty(len(vals), dtype=np.bool_)
    mask[0] = True
    mask[1:] = filled[1:] <= filled[:-1]  # pyright: ignore[reportOperatorIssue]

    # Reject NaNs
    if has_nans:
      mask &= ~np.isnan(vals)

    context.extra[self._state_key] = vals[-1]
    pd_mask = pd.Series(mask, index=data if isinstance(data, pd.Index) else data.index)
    return ValidationResult(
      success=False, message="Not monotonically decreasing", mask=pd_mask
    )

  @override
  def __str__(self) -> str:
    return "MonoDown"
