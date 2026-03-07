"""Persistent disk-based Numba compilation backend."""

from __future__ import annotations

from contextlib import contextmanager, suppress
import hashlib
import importlib.util
from pathlib import Path
import sys
import threading
from typing import TYPE_CHECKING, Any, NamedTuple, Protocol, cast

from filelock import FileLock
import numpy as np
import pandas as pd

from ..common import NumbaContext
from ..config import get_config

if TYPE_CHECKING:
  from collections.abc import Generator
  from types import ModuleType

  import numpy.typing as npt

  from ..common import PandasLike, ValidatorProtocol

  class NumbaKernel(Protocol):
    def __call__(self, *args: Any, **kwargs: Any) -> Any: ...  # noqa: ANN401  # pyright: ignore[reportExplicitAny]

  class NumbaCheckKernel(Protocol):
    def __call__(self, *args: Any, **kwargs: Any) -> bool: ...  # noqa: ANN401  # pyright: ignore[reportExplicitAny]


# Cache directory for compiled kernels
CACHE_DIR = Path.home() / ".cache" / "datawarden" / "numba"

# Version string for cache invalidation
BACKEND_VERSION = "1.6.2"


class NumbaKernels(NamedTuple):
  """Container for compiled Numba kernels."""

  kernel_serial: NumbaKernel
  kernel_parallel: NumbaKernel
  kernel_check_serial: NumbaCheckKernel
  kernel_check_parallel: NumbaCheckKernel


class NumbaKernelsColumnMode(NamedTuple):
  """Container for column-mode Numba kernels (with n_rows parameter)."""

  kernel_serial: NumbaKernel
  kernel_parallel: NumbaKernel
  kernel_check_serial: NumbaCheckKernel
  kernel_check_parallel: NumbaCheckKernel


def _fingerprint(
  validators: list[ValidatorProtocol],
  col_mode: bool = False,
  is_range_index: bool = False,
) -> str:
  """Generate unique key for a validator expression."""
  parts = [BACKEND_VERSION, ":"]
  if col_mode:
    parts.append("COL:")

  needs_index = False
  for v in validators:
    if v.uses_index:
      needs_index = True
      break

  if needs_index:
    parts.append("IDX:")
    if is_range_index:
      parts.append("RANGE:")

  for i, v in enumerate(validators):
    if i > 0:
      parts.append(":")
    v.build_numba_expr("x", parts, is_range_index=is_range_index)
  combined = "".join(parts)
  return hashlib.sha256(combined.encode()).hexdigest()[:16]


def _write_source(
  validators: list[ValidatorProtocol], path: Path, is_range_index: bool = False
) -> None:
  """Generate and write Numba kernel source to disk.

  WARNING: This function generates Python code from strings returned by
  `validator.build_numba_expr()`. While built-in validators sanitize their
  inputs (e.g. checking types in `NumericValidator`), this architecture relies
  on the assumption that all validators are trusted and implement safe string
  generation. Malicious or buggy `build_numba_expr` implementations could lead
  to arbitrary code execution.
  """
  parts: list[str] = []
  needs_index = any(v.uses_index for v in validators)

  if needs_index:
    if is_range_index:
      idx_args = ", r_start, r_step"
      idx_load = "idx_val = r_start + i * r_step"
      idx_dummy = "  index_arr = arr # Dummy for RangeIndex mode"
    else:
      idx_args = ", index_arr"
      idx_load = "idx_val = index_arr[i]"
      idx_dummy = ""
  else:
    idx_args = ""
    idx_load = ""
    idx_dummy = ""

  for i, v in enumerate(validators):
    if i > 0:
      parts.append(" and ")
    v.build_numba_expr("x", parts, arr_name="arr", is_range_index=is_range_index)
  combined_expr = "".join(parts)

  source = f'''"""Auto-generated Numba kernel. Do not edit."""
import numba
import numpy as np

@numba.njit(cache=True)
def kernel_serial(arr{idx_args}, out):
  n = len(arr)
{idx_dummy}
  for i in range(n):
    x = arr[i]
    {idx_load}
    out[i] = {combined_expr}

@numba.njit(parallel=True, cache=True)
def kernel_parallel(arr{idx_args}, out):
  n = len(arr)
{idx_dummy}
  for i in numba.prange(n):
    x = arr[i]
    {idx_load}
    out[i] = {combined_expr}

@numba.njit(cache=True)
def kernel_check_serial(arr{idx_args}):
  n = len(arr)
{idx_dummy}
  for i in range(n):
    x = arr[i]
    {idx_load}
    valid = {combined_expr}
    if not valid:
      return False
  return True

@numba.njit(parallel=True, cache=True)
def kernel_check_parallel(arr{idx_args}):
  n = len(arr)
{idx_dummy}
  errors = 0
  for i in numba.prange(n):
    x = arr[i]
    {idx_load}
    valid = {combined_expr}
    if not valid:
      errors += 1
  return errors == 0
'''
  path.write_text(source, encoding="utf-8")


def _write_source_column_mode(
  validators: list[ValidatorProtocol],
  col_map: dict[str, int],
  path: Path,
  is_range_index: bool = False,
) -> None:
  """Generate Numba kernel for multi-column validators."""
  ctx = NumbaContext(col_map=col_map)

  parts: list[str] = []
  needs_index = any(v.uses_index for v in validators)
  if needs_index:
    if is_range_index:
      idx_args = ", r_start, r_step"
      idx_load = "idx_val = r_start + i * r_step"
      idx_dummy = "  index_arr = arr # Dummy for RangeIndex mode"
    else:
      idx_args = ", index_arr"
      idx_load = "idx_val = index_arr[i]"
      idx_dummy = ""
  else:
    idx_args = ""
    idx_load = ""
    idx_dummy = ""

  for i, v in enumerate(validators):
    if i > 0:
      parts.append(" and ")
    v.build_numba_expr_column_mode(
      "arr", "i", ctx, parts, is_range_index=is_range_index
    )
  combined_expr = "".join(parts)

  source = f'''"""Auto-generated Numba kernel (column mode). Do not edit."""
import numba
import numpy as np

@numba.njit(cache=True)
def kernel_serial(arr{idx_args}, out, n_rows):
{idx_dummy}
  for i in range(n_rows):
    {idx_load}
    out[i] = {combined_expr}

@numba.njit(parallel=True, cache=True)
def kernel_parallel(arr{idx_args}, out, n_rows):
{idx_dummy}
  for i in numba.prange(n_rows):
    {idx_load}
    out[i] = {combined_expr}

@numba.njit(cache=True)
def kernel_check_serial(arr{idx_args}, n_rows):
{idx_dummy}
  for i in range(n_rows):
    {idx_load}
    valid = {combined_expr}
    if not valid:
      return False
  return True

@numba.njit(parallel=True, cache=True)
def kernel_check_parallel(arr{idx_args}, n_rows):
{idx_dummy}
  errors = 0
  for i in numba.prange(n_rows):
    {idx_load}
    valid = {combined_expr}
    if not valid:
      errors += 1
  return errors == 0
'''

  path.write_text(source, encoding="utf-8")


@contextmanager
def _file_lock(path: Path) -> Generator[None, None, None]:
  """Process-level lock using filelock."""
  lock_path = path.with_suffix(".lock")
  lock = FileLock(lock_path)
  try:
    with lock:
      yield
  finally:
    pass


# Pre-calculate attribute names for common cache keys
_ATTR_CACHE = {
  (False, False): "_numba_key",
  (False, True): "_numba_key_range",
  (True, False): "_numba_key_col",
  (True, True): "_numba_key_col_range",
}


def _get_cache_key(
  validators: list[ValidatorProtocol],
  cache_obj: ValidatorProtocol | None = None,
  col_mode: bool = False,
  is_range_index: bool = False,
) -> str:
  attr = _ATTR_CACHE[col_mode, is_range_index]
  if cache_obj is not None and hasattr(cache_obj, attr):
    return cast("str", getattr(cache_obj, attr))

  # Also check first validator if it is the only one
  if len(validators) == 1:
    v0 = validators[0]
    if hasattr(v0, attr):
      return cast("str", getattr(v0, attr))

  key = _fingerprint(validators, col_mode=col_mode, is_range_index=is_range_index)
  obj = (
    cache_obj
    if cache_obj is not None
    else (validators[0] if len(validators) == 1 else None)
  )
  if obj is not None:
    with suppress(AttributeError, TypeError):
      setattr(obj, attr, key)
  return key


def _load_module(module_name: str, file_path: Path) -> ModuleType:
  # Do not check sys.modules as we want to avoid polluting it to prevent memory leaks
  # with dynamically generated modules.

  spec = importlib.util.spec_from_file_location(module_name, file_path)
  if not spec or not spec.loader:
    raise RuntimeError(f"Could not load spec for {module_name}")

  module = importlib.util.module_from_spec(spec)
  # Do not add to sys.modules to prevent leak
  spec.loader.exec_module(module)
  return module


def _get_index_values(
  data: PandasLike,
) -> tuple[bool, float, float, npt.NDArray[np.floating] | None] | None:
  """Get index info. Returns (is_range, start, step, values) or None if not supported."""
  idx = data if isinstance(data, pd.Index) else data.index
  if isinstance(idx, pd.RangeIndex):
    return True, float(idx.start), float(idx.step), None

  # Check if index is numeric or datetime/timedelta
  if idx.dtype.kind in "ifMm":
    if idx.dtype.kind in "Mm":
      return False, 0.0, 0.0, idx.values.view(np.int64).astype(np.float64)
    return False, 0.0, 0.0, idx.values

  return None


class DiskCacheBackend:
  """Backend that persists generated source files to disk for Numba caching."""

  __slots__ = ("_column_mode_cache", "_lock", "_memory_cache", "cache_dir")

  def __init__(self, cache_dir: Path | None = None) -> None:
    super().__init__()
    self._memory_cache: dict[str, NumbaKernels] = {}
    self._column_mode_cache: dict[str, NumbaKernelsColumnMode] = {}
    self._lock = threading.Lock()
    self.cache_dir = cache_dir or CACHE_DIR
    self._ensure_cache_dir()

  def _ensure_cache_dir(self) -> None:
    """Create cache directory and add to sys.path."""
    with suppress(OSError, PermissionError):
      self.cache_dir.mkdir(parents=True, exist_ok=True)
      if str(self.cache_dir) not in sys.path:
        sys.path.append(str(self.cache_dir))

  def compile(
    self,
    validators: list[ValidatorProtocol],
    cache_obj: ValidatorProtocol | None = None,
    is_range_index: bool = False,
  ) -> NumbaKernels:
    """Compile validators, using disk cache when available."""
    key = _get_cache_key(
      validators, cache_obj, col_mode=False, is_range_index=is_range_index
    )
    if key in self._memory_cache:
      return self._memory_cache[key]

    with self._lock:
      if key in self._memory_cache:
        return self._memory_cache[key]

      module_name = f"dw_{key}"
      file_path = self.cache_dir / f"{module_name}.py"

      with _file_lock(file_path):
        if not file_path.exists():
          _write_source(validators, file_path, is_range_index=is_range_index)

        try:
          module = _load_module(module_name, file_path)
          kernels = NumbaKernels(
            module.kernel_serial,
            module.kernel_parallel,
            module.kernel_check_serial,
            module.kernel_check_parallel,
          )
          self._memory_cache[key] = kernels
          return kernels
        except Exception as e:
          # Force removal of potentially corrupted file
          with suppress(OSError):
            file_path.unlink(missing_ok=True)
          raise RuntimeError(f"Failed to compile kernel: {e}") from e

  def compile_column_mode(
    self,
    validators: list[ValidatorProtocol],
    col_map: dict[str, int],
    cache_obj: ValidatorProtocol | None = None,
    is_range_index: bool = False,
  ) -> NumbaKernelsColumnMode:
    """Compile column-mode validators, using disk cache when available."""
    key = _get_cache_key(
      validators, cache_obj, col_mode=True, is_range_index=is_range_index
    )
    if key in self._column_mode_cache:
      return self._column_mode_cache[key]

    with self._lock:
      if key in self._column_mode_cache:
        return self._column_mode_cache[key]

      module_name = f"dwc_{key}"
      file_path = self.cache_dir / f"{module_name}.py"

      with _file_lock(file_path):
        if not file_path.exists():
          _write_source_column_mode(
            validators, col_map, file_path, is_range_index=is_range_index
          )

        try:
          module = _load_module(module_name, file_path)
          kernels = NumbaKernelsColumnMode(
            module.kernel_serial,
            module.kernel_parallel,
            module.kernel_check_serial,
            module.kernel_check_parallel,
          )
          self._column_mode_cache[key] = kernels
          return kernels
        except Exception as e:
          # Force removal of potentially corrupted file
          with suppress(OSError):
            file_path.unlink(missing_ok=True)
          raise RuntimeError(f"Failed to compile column-mode kernel: {e}") from e

  def validate(  # noqa: C901, PLR0914
    self,
    data: PandasLike,
    validators: list[ValidatorProtocol],
    cache_obj: ValidatorProtocol | None = None,
  ) -> tuple[bool, npt.NDArray[np.bool_] | None]:
    """Run Numba-accelerated validation."""
    cfg = get_config()

    # Optimization: check length upfront
    n_elements = len(data)
    if hasattr(data, "columns"):
      n_elements *= len(data.columns)

    # Ensure we have a native numpy array for Numba
    arr = data.to_numpy()
    if arr.dtype.kind not in "if":
      raise NotImplementedError(f"Unsupported dtype: {arr.dtype}")

    needs_index = False
    for v in validators:
      if v.uses_index:
        needs_index = True
        break

    idx_info = None
    is_range = False
    if needs_index:
      idx_info = _get_index_values(data)
      if idx_info is None:
        raise NotImplementedError("Unsupported index type")
      is_range = idx_info[0]

    if arr.ndim > 1:
      arr = arr.ravel(order="F")

    kernels = self.compile(validators, cache_obj=cache_obj, is_range_index=is_range)

    # Fast path: check only
    use_parallel = n_elements >= cfg.parallel_threshold_rows
    check_k = (
      kernels.kernel_check_parallel if use_parallel else kernels.kernel_check_serial
    )

    if needs_index:
      _, r_start, r_step, idx_arr = idx_info  # pyright: ignore[reportUnknownVariableType, reportGeneralTypeIssues]
      is_valid = check_k(arr, r_start, r_step) if is_range else check_k(arr, idx_arr)
    else:
      is_valid = check_k(arr)

    if is_valid:
      return True, None

    if cfg.fail_fast:
      return False, None

    # Compute mask
    out: npt.NDArray[np.bool_] = np.empty(len(arr), dtype=np.bool_)
    mask_k = kernels.kernel_parallel if use_parallel else kernels.kernel_serial

    if needs_index:
      _, r_start, r_step, idx_arr = idx_info  # pyright: ignore[reportUnknownVariableType, reportGeneralTypeIssues]
      if is_range:
        mask_k(arr, r_start, r_step, out)
      else:
        mask_k(arr, idx_arr, out)
    else:
      mask_k(arr, out)

    return False, out

  def validate_column_mode(
    self,
    data: pd.DataFrame,
    validators: list[ValidatorProtocol],
    col_map: dict[str, int],
    n_rows: int,
    cache_obj: ValidatorProtocol | None = None,
  ) -> tuple[bool, npt.NDArray[np.bool_] | None]:
    """Run Numba-accelerated column-mode validation."""
    cfg = get_config()
    # Ensure we have a native numpy array for Numba
    arr = data.to_numpy()
    if arr.dtype.kind not in "if":
      raise NotImplementedError(f"Unsupported dtype: {arr.dtype}")

    needs_index = any(v.uses_index for v in validators)
    idx_info = None
    is_range = False
    if needs_index:
      idx_info = _get_index_values(data)
      if idx_info is None:
        raise NotImplementedError("Unsupported index type")
      is_range = idx_info[0]

    # Ravel column-major for column offset access
    arr = arr.ravel(order="F")

    kernels = self.compile_column_mode(
      validators, col_map, cache_obj=cache_obj, is_range_index=is_range
    )

    # Fast path: check only
    use_parallel = n_rows >= cfg.parallel_threshold_rows
    check_k = (
      kernels.kernel_check_parallel if use_parallel else kernels.kernel_check_serial
    )

    if needs_index:
      _, r_start, r_step, idx_arr = idx_info  # pyright: ignore[reportUnknownVariableType, reportGeneralTypeIssues]
      is_valid = (
        check_k(arr, r_start, r_step, n_rows)
        if is_range
        else check_k(arr, idx_arr, n_rows)
      )
    else:
      is_valid = check_k(arr, n_rows)

    if is_valid:
      return True, None

    if cfg.fail_fast:
      return False, None

    # Compute mask - only n_rows elements needed
    out: npt.NDArray[np.bool_] = np.empty(n_rows, dtype=np.bool_)
    mask_k = kernels.kernel_parallel if use_parallel else kernels.kernel_serial

    if needs_index:
      _, r_start, r_step, idx_arr = idx_info  # pyright: ignore[reportUnknownVariableType, reportGeneralTypeIssues]
      if is_range:
        mask_k(arr, r_start, r_step, out, n_rows)
      else:
        mask_k(arr, idx_arr, out, n_rows)
    else:
      mask_k(arr, out, n_rows)

    return False, out


# Singleton instance
_backend = DiskCacheBackend()


def run_numba_validation(
  data: PandasLike,
  validators: list[ValidatorProtocol],
  cache_obj: ValidatorProtocol | None = None,
) -> tuple[bool, npt.NDArray[np.bool_] | None]:
  """Run Numba validation using the disk-cached backend."""
  return _backend.validate(data, validators, cache_obj=cache_obj)


def run_numba_validation_column_mode(
  data: pd.DataFrame,
  validators: list[ValidatorProtocol],
  col_map: dict[str, int],
  n_rows: int,
  cache_obj: ValidatorProtocol | None = None,
) -> tuple[bool, npt.NDArray[np.bool_] | None]:
  """Run Numba column-mode validation using the disk-cached backend."""
  return _backend.validate_column_mode(data, validators, col_map, n_rows, cache_obj)
