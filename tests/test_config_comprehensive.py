"""Comprehensive tests for configuration handling.

Tests skip_validation, warn_only, and other config options.
"""

from dataclasses import FrozenInstanceError

import numpy as np
import pandas as pd
import pytest

from datawarden import Overrides, get_config
from datawarden.context import ValidationContext
from datawarden.validators import Finite


class TestConfigSkipValidation:
  """Test skip_validation configuration."""

  def test_skip_validation_default_false(self) -> None:
    """Default config should have skip_validation=False."""
    cfg = get_config()
    assert cfg.skip_validation is False

  def test_skip_validation_with_overrides(self) -> None:
    """skip_validation can be set via Overrides()."""
    # Invalid data
    data = pd.Series([np.inf, -np.inf])
    validator = Finite()
    ctx = ValidationContext(root_data=data)

    # Normal validation should fail
    result = validator.validate(data, ctx)
    assert not result.success

    # With skip_validation, should pass
    with Overrides(skip_validation=True):
      cfg = get_config()
      assert cfg.skip_validation is True
      # Validation is skipped at decorator level, not validator level
      # So this test just confirms config is set

  def test_skip_validation_context_manager(self) -> None:
    """skip_validation should be context-local."""
    assert get_config().skip_validation is False

    with Overrides(skip_validation=True):
      assert get_config().skip_validation is True

    # Should revert after context
    assert get_config().skip_validation is False

  def test_nested_overrides(self) -> None:
    """Nested Overrides should work correctly."""
    assert get_config().skip_validation is False

    with Overrides(skip_validation=True):
      assert get_config().skip_validation is True

      with Overrides(skip_validation=False):
        assert get_config().skip_validation is False

      # Should restore to outer context
      assert get_config().skip_validation is True

    assert get_config().skip_validation is False


class TestConfigNumba:
  """Test Numba-related configuration."""

  def test_use_numba_default(self) -> None:
    """Check default use_numba setting."""
    cfg = get_config()
    # Default may vary, just check it exists
    assert isinstance(cfg.use_numba, bool)

  def test_use_numba_override(self) -> None:
    """use_numba can be overridden."""
    original = get_config().use_numba

    with Overrides(use_numba=not original):
      assert get_config().use_numba == (not original)

    assert get_config().use_numba == original

  def test_numba_threshold(self) -> None:
    """numba_threshold should be configurable."""
    cfg = get_config()
    assert isinstance(cfg.numba_threshold, int)
    assert cfg.numba_threshold > 0

    with Overrides(numba_threshold=5000):
      assert get_config().numba_threshold == 5000

  def test_max_workers(self) -> None:
    """max_workers should be configurable."""
    cfg = get_config()
    assert isinstance(cfg.max_workers, int)
    assert cfg.max_workers > 0

    with Overrides(max_workers=8):
      assert get_config().max_workers == 8


class TestConfigChunking:
  """Test chunking-related configuration."""

  def test_chunk_size_rows(self) -> None:
    """chunk_size_rows should be configurable."""
    cfg = get_config()
    # chunk_size_rows can be None (default)
    assert cfg.chunk_size_rows is None or isinstance(cfg.chunk_size_rows, int)

    with Overrides(chunk_size_rows=5000):
      assert get_config().chunk_size_rows == 5000

  def test_parallel_threshold_rows(self) -> None:
    """parallel_threshold_rows should be configurable."""
    cfg = get_config()
    assert isinstance(cfg.parallel_threshold_rows, int)
    assert cfg.parallel_threshold_rows > 0

    with Overrides(parallel_threshold_rows=20000):
      assert get_config().parallel_threshold_rows == 20000


class TestConfigMultipleOverrides:
  """Test multiple config Overrides at once."""

  def test_multiple_overrides(self) -> None:
    """Multiple config options can be overridden together."""
    with Overrides(skip_validation=True, use_numba=False, max_workers=16):
      cfg = get_config()
      assert cfg.skip_validation is True
      assert cfg.use_numba is False
      assert cfg.max_workers == 16

    # All should revert
    cfg = get_config()
    assert cfg.skip_validation is False


class TestConfigImmutability:
  """Test that config is immutable (frozen dataclass)."""

  def test_config_is_frozen(self) -> None:
    """Config should be a frozen dataclass."""
    cfg = get_config()
    with pytest.raises(FrozenInstanceError):
      cfg.skip_validation = True
