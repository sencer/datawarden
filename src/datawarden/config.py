"""Global configuration for the datawarden library."""

from __future__ import annotations

import contextlib
import dataclasses
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
  from collections.abc import Iterator


@dataclasses.dataclass
class Config:
  """Global configuration settings.

  Attributes:
    parallel_threshold_rows: Minimum rows required to trigger parallel validation
      for multiple arguments (default: 50,000).
    max_workers: Maximum number of threads for parallel validation (default: 4).
    skip_validation: Whether to skip validation globally (default: False).
    warn_only: Whether to only warn on validation failures globally (default: False).
    chunk_size_rows: Number of rows per chunk for validation (default: None, disabled).
      Set this to process large datasets in chunks to reduce memory usage.
  """

  parallel_threshold_rows: int = 50_000
  max_workers: int = 4
  skip_validation: bool = False
  warn_only: bool = False
  chunk_size_rows: int | None = None


# Singleton instance
_config = Config()


def get_config() -> Config:
  """Get the global configuration."""
  return _config


def reset_config() -> None:
  """Reset configuration to defaults (mostly for testing)."""
  global _config
  _config = Config()


@contextlib.contextmanager
def overrides(**kwargs: Any) -> Iterator[None]:
  """Context manager to temporarily override configuration.

  Useful for:
  - Temporarily disabling validation (skip_validation=True)
  - Switching to warning mode (warn_only=True)
  - Enabling memory-efficient chunking (chunk_size_rows=10000)

  Example:
    ```python
    # Process large file in chunks without blocking UI
    with overrides(chunk_size_rows=50_000):
      process_large_dataset(df)
    ```
  """
  original = {}
  for key, value in kwargs.items():
    if hasattr(_config, key):
      original[key] = getattr(_config, key)
      setattr(_config, key, value)
    else:
      raise AttributeError(f"Config has no attribute '{key}'")

  try:
    yield
  finally:
    for key, value in original.items():
      setattr(_config, key, value)
