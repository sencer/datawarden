"""Global configuration for the datawarden library."""

from __future__ import annotations

import dataclasses


@dataclasses.dataclass
class Config:
  """Global configuration settings.

  Attributes:
    parallel_threshold_rows: Minimum rows required to trigger parallel validation
      for multiple arguments (default: 50,000).
    max_workers: Maximum number of threads for parallel validation (default: 4).
    skip_validation: Whether to skip validation globally (default: False).
    warn_only: Whether to only warn on validation failures globally (default: False).
  """

  parallel_threshold_rows: int = 50_000
  max_workers: int = 4
  skip_validation: bool = False
  warn_only: bool = False


# Singleton instance
_config = Config()


def get_config() -> Config:
  """Get the global configuration."""
  return _config


def reset_config() -> None:
  """Reset configuration to defaults (mostly for testing)."""
  global _config
  _config = Config()
