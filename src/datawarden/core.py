from __future__ import annotations

import atexit
from collections.abc import Callable
import concurrent.futures
from contextlib import suppress
from functools import wraps
import inspect
import threading
from types import UnionType
from typing import (
  TYPE_CHECKING,
  Annotated,
  ParamSpec,
  Protocol,
  TypeVar,
  Union,
  cast,
  get_args,
  get_origin,
  override,
  runtime_checkable,
)
import warnings

import pandas as pd

from .backends.numba import run_numba_validation
from .config import get_config, skip_validation_var
from .context import ValidationContext
from .exceptions import LogicError, ValidationError
from .validators import (
  SUCCESS,
  And,
  BaseValidator,
  Column,
  Finite,
  Ge,
  Gt,
  IsInstance,
  Le,
  Lt,
  NotNaN,
  NumericValidator,
  PandasLike,
  Priority,
  ValidationResult,
)

if TYPE_CHECKING:
  from collections.abc import Callable

  import numpy as np

# Validated is a direct alias to Annotated to allow both types (Finite) and values (Ge(0))
Validated = Annotated
P = ParamSpec("P")
R = TypeVar("R")
T = TypeVar("T")

NUMBA_FUSION_THRESHOLD = 2


@runtime_checkable
class Validatable(Protocol):
  """Object that can be validated (has .values for numpy/numba)."""

  @property
  def values(self) -> np.ndarray[tuple[int, ...], np.dtype[np.floating]]: ...  # pyright: ignore[reportMissingTypeArgument]


@runtime_checkable
class Slicable(Validatable, Protocol):
  """Object that can be sliced and has a length."""

  def __len__(self) -> int: ...

  def __getitem__(self, key: Any) -> Any: ...


# Global executor pool to avoid thread spawn overhead and allow concurrent worker counts
_executors: dict[int, concurrent.futures.ThreadPoolExecutor] = {}
_executors_lock = threading.RLock()
DEFAULT_MAX_WORKERS = 4
MAX_EXECUTOR_CACHE_SIZE = 8


def _shutdown_executors() -> None:
  with _executors_lock:
    for executor in _executors.values():
      executor.shutdown(wait=True)


atexit.register(_shutdown_executors)


def _get_executor(max_workers: int | None) -> concurrent.futures.ThreadPoolExecutor:
  # Default to 4 if None, consistent with Config default
  mw = max_workers if max_workers is not None else DEFAULT_MAX_WORKERS
  with _executors_lock:
    if mw in _executors:
      # Move to end (LRU behavior)
      executor = _executors.pop(mw)
      _executors[mw] = executor
      return executor

    # Limit cache size to prevent leaks (e.g., keep last 8)
    if len(_executors) >= MAX_EXECUTOR_CACHE_SIZE:
      oldest_mw = next(iter(_executors))
      _executors.pop(oldest_mw)
      # Do not shutdown the evicted executor immediately.
      # It might still be in use by a thread that just retrieved it.
      # Python's ThreadPoolExecutor (>= 3.9) will shutdown automatically
      # when garbage collected.

    _executors[mw] = concurrent.futures.ThreadPoolExecutor(max_workers=mw)
    return _executors[mw]


def _unwrap_annotation(hint: object) -> tuple[object, bool]:
  """Unwrap Union/Optional to find Validated/Annotated and check for None."""
  origin = get_origin(hint)
  if origin is not Union and origin is not UnionType:
    return hint, False

  allow_none = False
  target_hint = hint
  args = get_args(hint)
  if type(None) in args:
    allow_none = True

  # Find the Validated/Annotated arg
  for arg in args:
    arg_origin = get_origin(arg)
    if arg_origin is Annotated or arg_origin is Validated:
      target_hint = arg
      break
  return target_hint, allow_none


def _extract_validators(args: tuple[object, ...]) -> list[BaseValidator[PandasLike]]:
  """Extract and instantiate validators from Annotation arguments."""
  validators: list[BaseValidator[PandasLike]] = []
  for v in args:
    if isinstance(v, BaseValidator):
      validators.append(v)
    elif isinstance(v, type) and issubclass(v, BaseValidator):
      # Only instantiate if the validator supports no-arg init, or raise/skip
      try:
        validators.append(v())
      except TypeError as e:
        # Do not silently ignore validators. Fail fast if usage is incorrect.
        raise TypeError(
          f"Validator {v.__name__} requires arguments and cannot be used as a type."
        ) from e
  return validators


def _get_base_validators(base_type: object) -> list[BaseValidator[PandasLike]]:
  """Get IsInstance validator for the base type."""
  return [IsInstance(base_type)]


class ValidationPlan:
  __slots__ = (
    "_use_numba_at_init",
    "arg_indices",
    "arg_names",
    "arg_plans",
    "defaults",
    "func",
    "func_name",
    "has_var_args",
    "parameters",
    "signature",
    "validation_order",
  )

  def __init__(self, func: Callable[..., object]) -> None:
    super().__init__()
    self.func = func
    self.func_name = func.__name__
    self.signature = inspect.signature(func)
    self.arg_plans: dict[str, OptimizedPlan[PandasLike]] = {}
    self.arg_names = list(self.signature.parameters.keys())
    self.parameters = list(self.signature.parameters.values())
    self.arg_indices = {name: i for i, name in enumerate(self.arg_names)}
    self.has_var_args = any(
      p.kind in {inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD}
      for p in self.parameters
    )
    self.defaults: dict[str, object] = {
      p.name: p.default
      for p in self.parameters
      if p.default is not inspect.Parameter.empty
    }

    self._use_numba_at_init = get_config().use_numba
    self._parse_signature()

    # Pre-calculated list of names that have validators
    self.validation_order = [name for name in self.arg_names if name in self.arg_plans]

  def _reoptimize(self, use_numba: bool) -> None:
    """Re-optimize all argument plans with new Numba setting."""
    self._use_numba_at_init = use_numba
    self._parse_signature()
    self.validation_order = [name for name in self.arg_names if name in self.arg_plans]

  def _parse_signature(self) -> None:
    self.arg_plans.clear()
    for name, param in self.signature.parameters.items():
      hint = param.annotation
      target_hint, allow_none = _unwrap_annotation(hint)

      origin = get_origin(target_hint)
      if origin is Annotated or origin is Validated:
        args = get_args(target_hint)
        base_type = args[0]
        validators = _extract_validators(args[1:])

        if validators:
          all_validators = _get_base_validators(base_type) + validators
          self.arg_plans[name] = OptimizedPlan(all_validators, allow_none=allow_none)

  def validate_args(self, *args: object, **kwargs: object) -> None:
    cfg = get_config()
    if cfg.enforce_cow:
      with pd.option_context("mode.copy_on_write", True):
        self._validate_args_internal(args, kwargs)
    else:
      self._validate_args_internal(args, kwargs)

  def _validate_args_internal(
    self, args: tuple[object, ...], kwargs: dict[str, object]
  ) -> None:
    if not self.has_var_args and len(args) <= len(self.arg_names):
      self._validate_positional(args)
      self._validate_keyword_and_defaults(len(args), kwargs)
    else:
      bound = self.signature.bind(*args, **kwargs)
      bound.apply_defaults()
      for name in self.validation_order:
        if name in bound.arguments:
          self._validate_one_fast(name, bound.arguments[name])

  def _validate_positional(self, args: tuple[object, ...]) -> None:
    for i, arg in enumerate(args):
      name = self.arg_names[i]
      if name in self.arg_plans:
        self._validate_one_fast(name, arg)

  def _validate_keyword_and_defaults(
    self,
    num_pos: int,
    kwargs: dict[str, object],
  ) -> None:
    for name in self.validation_order:
      # Check if name is already validated as positional
      if self.arg_indices[name] < num_pos:
        continue

      if name in kwargs:
        self._validate_one_fast(name, kwargs[name])
      elif name in self.defaults:
        self._validate_one_fast(name, self.defaults[name])

  def _validate_one_fast(self, name: str, value: object) -> None:
    res = self.arg_plans[name].execute_fast(value)
    if res is not None:
      raise ValidationError(
        f"Argument '{name}' failed validation in '{self.func_name}':\n - {res}"
      )


class GlobalScopeWrap(BaseValidator[pd.DataFrame]):
  __slots__ = ("exclude_columns", "validator")

  def __init__(
    self,
    validator: BaseValidator[PandasLike],
    exclude_columns: list[str],
  ) -> None:
    super().__init__()
    self.validator = validator
    self.exclude_columns = exclude_columns

  @override
  def validate(
    self,
    data: pd.DataFrame,
    context: ValidationContext,
  ) -> ValidationResult:
    # Optimized: Find columns to check without dropping
    cols_to_check = [c for c in data.columns if c not in self.exclude_columns]
    if not cols_to_check:
      return SUCCESS

    # If single column, pass as Series (usually a view)
    # If all columns, pass data as is
    target: pd.DataFrame | pd.Series[float]
    if len(cols_to_check) == 1:
      target = data[cols_to_check[0]]
    elif len(cols_to_check) < len(data.columns):
      target = data[cols_to_check]
    else:
      target = data

    res = self.validator.validate(target, context)
    if not res.success:
      return ValidationResult(
        success=False,
        message=f"Global check failed (excluding {self.exclude_columns}): {res.message}",
        mask=res.mask,
      )
    return SUCCESS

  @override
  def __str__(self) -> str:
    return f"Global({self.validator}, exclude={self.exclude_columns})"


class NumbaFusedValidator(BaseValidator["pd.Series[float] | pd.DataFrame"]):
  __slots__ = ("fallback", "validators")

  def __init__(self, validators: list[BaseValidator[PandasLike]]) -> None:
    super().__init__()
    self.validators = validators
    self.fallback = And(*validators)

  @property
  @override
  def numba_supported(self) -> bool:
    return True

  @property
  @override
  def uses_index(self) -> bool:
    return any(v.uses_index for v in self.validators)

  @override
  def validate(
    self,
    data: pd.Series[float] | pd.DataFrame,
    context: ValidationContext,
  ) -> ValidationResult:
    if get_config().use_numba and len(data) >= get_config().numba_threshold:
      with suppress(Exception):
        success, mask = run_numba_validation(data, self.validators, cache_obj=self)

        if success:
          return SUCCESS

        if mask is not None:
          if isinstance(data, pd.Series):
            pd_mask: pd.Series[bool] | pd.DataFrame = pd.Series(
              mask, index=data.index, copy=False
            )
            return ValidationResult(success=False, message=str(self), mask=pd_mask)
          if isinstance(data, pd.DataFrame):
            reshaped = mask.reshape(data.shape, order="F")
            pd_mask = pd.DataFrame(
              reshaped, index=data.index, columns=data.columns, copy=False
            )
            return ValidationResult(success=False, message=str(self), mask=pd_mask)

        # Fast path failed, no mask available
        return ValidationResult(
          success=False, message=f"Fused validation failed: {self}"
        )

    return self.fallback.validate(data, context)

  @override
  def __str__(self) -> str:
    return "(" + " & ".join(str(v) for v in self.validators) + ")"


def _group_atoms[T: PandasLike](
  atoms: list[BaseValidator[T]], claimed_columns: list[str]
) -> dict[str, list[BaseValidator[PandasLike]]]:
  groups: dict[str, list[BaseValidator[PandasLike]]] = {}
  for a in atoms:
    if isinstance(a, Column):
      groups.setdefault(f"col:{a.name}", []).append(a.validator)
    elif isinstance(a, NumericValidator) and a.targets is None and claimed_columns:
      groups.setdefault("__global__", []).append(a)
    else:
      groups.setdefault("__root__", []).append(a)
  return groups


def _map_to_final[T: PandasLike](
  final: list[BaseValidator[T]],
  simplified: list[BaseValidator[PandasLike]],
  target: str,
  claimed_columns: list[str],
) -> None:
  for sv in simplified:
    if target.startswith("col:"):
      final.append(Column(target[4:], sv))
    elif target == "__global__":
      final.append(GlobalScopeWrap(sv, claimed_columns))
    else:
      final.append(sv)


def _get_lb(gt_v: float | None, ge_v: float | None) -> tuple[float | None, bool]:
  if gt_v is not None and ge_v is not None:
    return (gt_v, True) if gt_v >= ge_v else (ge_v, False)
  if gt_v is not None:
    return gt_v, True
  if ge_v is not None:
    return ge_v, False
  return None, False


def _get_ub(lt_v: float | None, le_v: float | None) -> tuple[float | None, bool]:
  if lt_v is not None and le_v is not None:
    return (lt_v, True) if lt_v <= le_v else (le_v, False)
  if lt_v is not None:
    return lt_v, True
  if le_v is not None:
    return le_v, False
  return None, False


def _combine_bounds[T: PandasLike](
  gt_v: float | None,
  ge_v: float | None,
  lt_v: float | None,
  le_v: float | None,
) -> list[BaseValidator[T]]:
  res: list[BaseValidator[T]] = []
  lb, lb_strict = _get_lb(gt_v, ge_v)
  if lb is not None:
    res.append(cast("BaseValidator[T]", Gt(lb) if lb_strict else Ge(lb)))

  ub, ub_strict = _get_ub(lt_v, le_v)
  if ub is not None:
    res.append(cast("BaseValidator[T]", Lt(ub) if ub_strict else Le(ub)))

  if lb is not None and ub is not None:
    contradiction = lb > ub or (lb == ub and (lb_strict or ub_strict))
    if contradiction:
      raise LogicError(f"Contradictory bounds: {res}")
  return res


def _fuse_numeric[T: PandasLike](
  validators: list[BaseValidator[T]],
) -> list[BaseValidator[T]]:
  cfg = get_config()
  if cfg.use_numba:
    # If Numba is enabled, we prefer to keep individual atom validators
    # so that NumbaFusedValidator can fuse them into a single loop.
    # Simplifying them to a single Between/Ge might actually lose 
    # some Numba-specific optimizations or clarity.
    return validators

  # Merge Gt/Ge/Lt/Le on same column/targets
  # If same target, we can find the tightest bound.
  # e.g. Ge(5) & Ge(10) -> Ge(10)
  # e.g. Ge(5) & Le(15) -> Between(5, 15)
  # For now, just basic LB/UB merging for scalars
  others: list[BaseValidator[T]] = []
  gt_v, ge_v, lt_v, le_v = None, None, None, None
  has_not_nan = False
  has_finite = False

  for v in validators:
    if isinstance(v, Gt) and v.targets is None:
      gt_v = max(gt_v, v.value) if gt_v is not None else v.value
    elif isinstance(v, Ge) and v.targets is None:
      ge_v = max(ge_v, v.value) if ge_v is not None else v.value
    elif isinstance(v, Lt) and v.targets is None:
      lt_v = min(lt_v, v.value) if lt_v is not None else v.value
    elif isinstance(v, Le) and v.targets is None:
      le_v = min(le_v, v.value) if le_v is not None else v.value
    elif isinstance(v, NotNaN):
      has_not_nan = True
    elif isinstance(v, Finite):
      has_finite = True
      others.append(v)
    else:
      others.append(v)

  if has_finite:
    has_not_nan = False

  res = _combine_bounds(gt_v, ge_v, lt_v, le_v)
  if has_not_nan and not any(x is not None for x in [gt_v, ge_v, lt_v, le_v]):
    res.append(cast("BaseValidator[T]", NotNaN))
  return res + others


def _fuse_and_simplify(
  validators: list[BaseValidator[PandasLike]],
) -> list[BaseValidator[PandasLike]]:
  cfg = get_config()
  simplified = _fuse_numeric(validators)
  if (
    cfg.use_numba
    and len([v for v in simplified if v.numba_supported]) >= NUMBA_FUSION_THRESHOLD
  ):
    numba_ready = [v for v in simplified if v.numba_supported]
    others = [v for v in simplified if not v.numba_supported]
    return [NumbaFusedValidator(numba_ready), *others]
  return simplified


def _optimize[T: PandasLike](
  validators: list[BaseValidator[T]],
) -> list[BaseValidator[T]]:
  atoms: list[BaseValidator[T]] = []
  for v in validators:
    for transformed in v.transform():
      atoms.extend(transformed.decompose())

  claimed_columns = [v.name for v in atoms if isinstance(v, Column)]
  groups = _group_atoms(atoms, claimed_columns)

  final_validators: list[BaseValidator[T]] = []
  targets = ["__root__", *[t for t in groups if t != "__root__"]]

  for target in targets:
    if target not in groups:
      continue
    simplified = _fuse_and_simplify(groups[target])
    _map_to_final(final_validators, simplified, target, claimed_columns)

  final_validators.sort(key=lambda x: x.priority)
  return final_validators


class OptimizedPlan[T: PandasLike]:
  __slots__ = ("allow_none", "heavy_validator_count", "validators")

  def __init__(
    self,
    validators: list[BaseValidator[T]],
    allow_none: bool = False,
  ) -> None:
    super().__init__()
    self.allow_none = allow_none
    self.validators = _optimize(validators)
    self.heavy_validator_count = sum(
      1 for v in self.validators if v.priority >= Priority.VECTORIZED
    )

  def execute_fast(self, data: object) -> str | None:
    if data is None and self.allow_none:
      return None

    cfg = get_config()
    data_len = -1
    has_len = hasattr(data, "__len__")

    if cfg.chunk_size_rows and has_len and isinstance(data, Slicable):
      data_len = len(data)
      if data_len > cfg.chunk_size_rows:
        return self._execute_chunked_msg(data, cfg.chunk_size_rows)

    if cfg.parallel_threshold_rows and self.heavy_validator_count > 1 and has_len:
      if data_len == -1:
        data_len = len(data)
      if data_len > cfg.parallel_threshold_rows and isinstance(
        data, (pd.Series, pd.DataFrame)
      ):
        return self._execute_parallel_msg(data)

    ctx: ValidationContext | None = None
    for v in self.validators:
      if ctx is None:
        ctx = ValidationContext(root_data=data)
      res = v.validate(data, ctx)
      if not res.success:
        return res.get_summary()

    return None

  def _execute_parallel_msg(self, data: PandasLike) -> str | None:
    ctx = ValidationContext(root_data=data)
    results = self._execute_parallel(data, ctx)
    failed = [r for r in results if not r.success]
    if not failed:
      return None
    return "\n".join(f" - {r.get_summary()}" for r in failed)

  def _execute_parallel(
    self,
    data: PandasLike,
    context: ValidationContext,
  ) -> list[ValidationResult]:
    results: list[ValidationResult | None] = [None] * len(self.validators)
    cfg = get_config()

    # Use cached executor
    executor = _get_executor(cfg.max_workers)

    futures: dict[concurrent.futures.Future[ValidationResult], int] = {
      executor.submit(v.validate, data, context): i
      for i, v in enumerate(self.validators)
    }

    for future in concurrent.futures.as_completed(futures):
      idx = futures[future]
      try:
        results[idx] = future.result()
      except (
        RuntimeError,
        ValueError,
        TypeError,
        AttributeError,
        ImportError,
        IndexError,
        KeyError,
        NameError,
        AssertionError,
        MemoryError,
        ArithmeticError,
      ) as e:
        results[idx] = ValidationResult(
          success=False, message=f"Validator crashed: {e}"
        )

    return [r for r in results if r is not None]

  def _execute_chunked_msg(self, data: Slicable, chunk_size: int) -> str | None:
    if not isinstance(data, Slicable):
      return None

    results = self._execute_chunked(data, chunk_size)
    failed = [r for r in results if not r.success]
    if not failed:
      return None
    return "\n".join(f" - {r.get_summary()}" for r in failed)

  def _execute_chunked(
    self,
    data: PandasLike,
    chunk_size: int,
  ) -> list[ValidationResult]:
    validator_results: list[list[ValidationResult]] = [[] for _ in self.validators]

    total_rows = len(data)
    context = ValidationContext(root_data=data)
    cfg = get_config()

    # Check if we should parallelize WITHIN chunks
    # We only do this if the chunk itself is large enough to warrant parallelism
    use_parallel = (
      cfg.parallel_threshold_rows
      and chunk_size >= cfg.parallel_threshold_rows
      and self.heavy_validator_count > 1
    )

    for start in range(0, total_rows, chunk_size):
      end = min(start + chunk_size, total_rows)
      chunk = data.iloc[start:end] if hasattr(data, "iloc") else data[start:end]

      if use_parallel:
        chunk_results = self._execute_parallel(chunk, context)
        for i, res in enumerate(chunk_results):
          validator_results[i].append(res)
      else:
        for i, v in enumerate(self.validators):
          res = v.validate(chunk, context)
          validator_results[i].append(res)

    return [
      OptimizedPlan._aggregate_results(v_res_list) for v_res_list in validator_results
    ]

  @staticmethod
  def _aggregate_results(
    results: list[ValidationResult],
  ) -> ValidationResult:
    all_success = all(r.success for r in results)
    if all_success:
      return SUCCESS

    has_masks = all(r.mask is not None for r in results)
    combined_mask: pd.Series[bool] | pd.DataFrame | None = None
    if has_masks:
      with suppress(Exception):
        masks = [r.mask for r in results]
        combined_mask = pd.concat(masks)

    msg = next((r.message for r in results if not r.success), "Validation failed")
    return ValidationResult(success=False, message=msg, mask=combined_mask)


def validate[**P, R](func: Callable[P, R]) -> Callable[P, R]:
  cfg = get_config()
  if cfg.skip_validation:
    return func

  plan = ValidationPlan(func)
  get_skip = skip_validation_var.get

  @wraps(func)
  def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
    if get_skip():
      return func(*args, **kwargs)

    cfg = get_config()

    if cfg.skip_validation:
      return func(*args, **kwargs)

    # Re-optimize if Numba setting changed since plan creation
    if cfg.use_numba != plan._use_numba_at_init:
      plan._reoptimize(cfg.use_numba)

    try:
      plan.validate_args(*args, **kwargs)
    except ValidationError as e:
      if cfg.warn_only:
        warnings.warn(str(e), stacklevel=2)
      else:
        raise

    return func(*args, **kwargs)

  return wrapper
