from collections.abc import Callable
from contextvars import ContextVar
import dataclasses
import os
from types import TracebackType
from typing import TYPE_CHECKING, TypedDict, Unpack, cast, override

if TYPE_CHECKING:
  from contextvars import Token


class ConfigParams(TypedDict, total=False):
  chunk_size_rows: int | None
  enforce_cow: bool
  fail_fast: bool
  max_workers: int
  numba_threshold: int
  parallel_threshold_rows: int
  skip_validation: bool
  use_numba: bool
  warn_only: bool


def _getenv[T: (int, bool)](key: str, default: T, cast_func: Callable[[str], T]) -> T:
  if (val := os.environ.get(key)) is not None:
    return cast_func(val)

  return default


def _get_bool(key: str, default: bool) -> bool:
  return _getenv(key, default, lambda x: x.lower() == "true")


def _get_int(key: str, default: int) -> int:
  return _getenv(key, default, int)


# Optimization: Direct ContextVar for skip_validation to avoid object/attr overhead in tight loops
skip_validation_var: ContextVar[bool] = ContextVar("skip_validation", default=False)


class Config:
  """Configuration state.

  Implemented as a manual class instead of a dataclass to allow a dynamic
  `skip_validation` property while maintaining frozen-like behavior and slots.
  """

  __slots__ = (
    "_skip_validation",
    "chunk_size_rows",
    "enforce_cow",
    "fail_fast",
    "max_workers",
    "numba_threshold",
    "parallel_threshold_rows",
    "use_numba",
    "warn_only",
  )

  _skip_validation: bool | None  # pyright: ignore[reportUninitializedInstanceVariable]
  chunk_size_rows: int | None  # pyright: ignore[reportUninitializedInstanceVariable]
  enforce_cow: bool  # pyright: ignore[reportUninitializedInstanceVariable]
  fail_fast: bool  # pyright: ignore[reportUninitializedInstanceVariable]
  max_workers: int  # pyright: ignore[reportUninitializedInstanceVariable]
  numba_threshold: int  # pyright: ignore[reportUninitializedInstanceVariable]
  parallel_threshold_rows: int  # pyright: ignore[reportUninitializedInstanceVariable]
  use_numba: bool  # pyright: ignore[reportUninitializedInstanceVariable]
  warn_only: bool  # pyright: ignore[reportUninitializedInstanceVariable]

  def __init__(  # noqa: PLR0913
    self,
    *,
    use_numba: bool = True,
    numba_threshold: int = 5_000,
    parallel_threshold_rows: int = 100_000,
    chunk_size_rows: int | None = None,
    max_workers: int = 4,
    warn_only: bool = False,
    fail_fast: bool = False,
    enforce_cow: bool = False,
    _skip_validation: bool | None = None,
  ) -> None:
    super().__init__()
    # Use object.__setattr__ to bypass our own __setattr__ which raises error
    set_attr = object.__setattr__
    set_attr(self, "use_numba", use_numba)
    set_attr(self, "numba_threshold", numba_threshold)
    set_attr(self, "parallel_threshold_rows", parallel_threshold_rows)
    set_attr(self, "chunk_size_rows", chunk_size_rows)
    set_attr(self, "max_workers", max_workers)
    set_attr(self, "warn_only", warn_only)
    set_attr(self, "fail_fast", fail_fast)
    set_attr(self, "enforce_cow", enforce_cow)
    set_attr(self, "_skip_validation", _skip_validation)

  @property
  def skip_validation(self) -> bool:
    if self._skip_validation is not None:
      return self._skip_validation
    return skip_validation_var.get()

  @override
  def __setattr__(self, name: str, value: object) -> None:
    raise dataclasses.FrozenInstanceError(f"cannot assign to field {name!r}")

  @override
  def __delattr__(self, name: str) -> None:
    raise dataclasses.FrozenInstanceError(f"cannot delete field {name!r}")

  @override
  def __repr__(self) -> str:
    fields = [
      f"{s}={getattr(self, s)!r}" for s in self.__slots__ if not s.startswith("_")
    ]
    fields.append(f"skip_validation={self.skip_validation!r}")
    return f"Config({', '.join(fields)})"

  @override
  def __eq__(self, other: object) -> bool:
    if not isinstance(other, Config):
      return NotImplemented
    return all(getattr(self, s) == getattr(other, s) for s in self.__slots__)

  @override
  def __hash__(self) -> int:
    return hash(tuple(getattr(self, s) for s in self.__slots__))

  @classmethod
  def from_env(cls) -> "Config":
    chunk_size_rows: int | None = None
    if (chunk_size_env := os.environ.get("DATAWARDEN_CHUNK_SIZE")) is not None:
      chunk_size_rows = int(chunk_size_env)

    return cls(
      chunk_size_rows=chunk_size_rows,
      enforce_cow=_get_bool("DATAWARDEN_ENFORCE_COW", False),
      fail_fast=_get_bool("DATAWARDEN_FAIL_FAST", False),
      max_workers=_get_int("DATAWARDEN_MAX_WORKERS", 4),
      numba_threshold=_get_int("DATAWARDEN_NUMBA_THRESHOLD", 5000),
      parallel_threshold_rows=_get_int("DATAWARDEN_PARALLEL_THRESHOLD", 100_000),
      use_numba=_get_bool("DATAWARDEN_USE_NUMBA", True),
      warn_only=_get_bool("DATAWARDEN_WARN_ONLY", False),
    )

  def replace(self, **kwargs: Unpack[ConfigParams]) -> "Config":
    """Optimized replacement method avoiding dataclasses.replace overhead."""
    # Manual caching to avoid memory leaks of lru_cache on methods while preserving speed
    cache_key = (self, tuple(sorted(kwargs.items())))
    if (cached := _CONFIG_CACHE.get(cache_key)) is not None:
      return cached

    new_cfg = Config(
      chunk_size_rows=kwargs.get("chunk_size_rows", self.chunk_size_rows),
      enforce_cow=kwargs.get("enforce_cow", self.enforce_cow),
      fail_fast=kwargs.get("fail_fast", self.fail_fast),
      max_workers=kwargs.get("max_workers", self.max_workers),
      numba_threshold=kwargs.get("numba_threshold", self.numba_threshold),
      parallel_threshold_rows=kwargs.get(
        "parallel_threshold_rows", self.parallel_threshold_rows
      ),
      _skip_validation=kwargs.get("skip_validation", self._skip_validation),
      use_numba=kwargs.get("use_numba", self.use_numba),
      warn_only=kwargs.get("warn_only", self.warn_only),
    )

    if len(_CONFIG_CACHE) >= MAX_CONFIG_CACHE_SIZE:
      _CONFIG_CACHE.clear()

    _CONFIG_CACHE[cache_key] = new_cfg

    return new_cfg


# Internal cache for derived configurations
MAX_CONFIG_CACHE_SIZE = 1024
_CONFIG_CACHE: dict[tuple[Config, tuple[tuple[str, object], ...]], Config] = {}

# Initialize global configuration
_config_var: ContextVar[Config] = ContextVar("datawarden_config")
_initial_config = Config.from_env()
_config_var.set(_initial_config)

# Sync skip_validation_var with initial env state
skip_validation_var.set(_get_bool("DATAWARDEN_SKIP_VALIDATION", False))


def get_config() -> Config:
  return _config_var.get()


def set(**kwargs: Unpack[ConfigParams]) -> None:
  """Update global configuration."""
  current = _config_var.get()
  new_config = current.replace(**kwargs)
  _config_var.set(new_config)
  if "skip_validation" in kwargs:
    skip_validation_var.set(kwargs["skip_validation"])


_UNSET = object()


class Overrides:
  """Context manager for temporary config overrides.

  Optimized to be a class to avoid generator overhead of @contextmanager.
  """

  __slots__ = ("_new_skip_val", "_only_skip", "new_config", "skip_token", "token")

  def __init__(  # noqa: PLR0913, C901
    self,
    *,
    chunk_size_rows: int | object | None = _UNSET,
    enforce_cow: bool | object = _UNSET,
    fail_fast: bool | object = _UNSET,
    max_workers: int | object = _UNSET,
    numba_threshold: int | object = _UNSET,
    parallel_threshold_rows: int | object = _UNSET,
    skip_validation: bool | object = _UNSET,
    use_numba: bool | object = _UNSET,
    warn_only: bool | object = _UNSET,
  ) -> None:
    super().__init__()
    # Optimization: if only skip_validation is changing, avoid Config copy and dict creation
    is_skip_optimization = (
      skip_validation is not _UNSET
      and chunk_size_rows is _UNSET
      and enforce_cow is _UNSET
      and fail_fast is _UNSET
      and max_workers is _UNSET
    )
    is_skip_optimization = (
      is_skip_optimization
      and numba_threshold is _UNSET
      and parallel_threshold_rows is _UNSET
      and use_numba is _UNSET
      and warn_only is _UNSET
    )

    if is_skip_optimization:
      self._only_skip = True
      self._new_skip_val = cast("bool", skip_validation)
      self.new_config = None
    else:
      self._only_skip = False
      current = _config_var.get()

      # Construct kwargs only if needed
      kwargs: ConfigParams = {}
      if chunk_size_rows is not _UNSET:
        kwargs["chunk_size_rows"] = cast("int | None", chunk_size_rows)
      if enforce_cow is not _UNSET:
        kwargs["enforce_cow"] = cast("bool", enforce_cow)
      if fail_fast is not _UNSET:
        kwargs["fail_fast"] = cast("bool", fail_fast)
      if max_workers is not _UNSET:
        kwargs["max_workers"] = cast("int", max_workers)
      if numba_threshold is not _UNSET:
        kwargs["numba_threshold"] = cast("int", numba_threshold)
      if parallel_threshold_rows is not _UNSET:
        kwargs["parallel_threshold_rows"] = cast("int", parallel_threshold_rows)
      if skip_validation is not _UNSET:
        kwargs["skip_validation"] = cast("bool", skip_validation)
      if use_numba is not _UNSET:
        kwargs["use_numba"] = cast("bool", use_numba)
      if warn_only is not _UNSET:
        kwargs["warn_only"] = cast("bool", warn_only)

      self.new_config = current.replace(**kwargs)

    self.token: Token[Config] | None = None
    self.skip_token: Token[bool] | None = None

  def __enter__(self) -> None:
    if self._only_skip:
      self.skip_token = skip_validation_var.set(self._new_skip_val)
    elif self.new_config is not None:
      self.token = _config_var.set(self.new_config)
      self.skip_token = skip_validation_var.set(self.new_config.skip_validation)

  def __exit__(
    self,
    exc_type: type[BaseException] | None,
    exc_val: BaseException | None,
    exc_tb: TracebackType | None,
  ) -> None:
    if self.token is not None:
      _config_var.reset(self.token)
    if self.skip_token is not None:
      skip_validation_var.reset(self.skip_token)
