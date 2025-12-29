import pandas as pd
import pytest

from datawarden import Finite, Validated, validate
from datawarden.config import get_config, overrides, reset_config


class TestGlobalConfig:
  """Tests for global configuration settings."""

  def setup_method(self):
    """Reset config before each test."""
    reset_config()

  def teardown_method(self):
    """Reset config after each test."""
    reset_config()

  def test_global_skip_validation(self):
    """Test that global skip_validation works."""
    get_config().skip_validation = True

    @validate
    def process(_data: Validated[pd.Series, Finite]):
      return True

    # Should not raise even with invalid data (Infinite)
    assert process(pd.Series([float("inf")])) is True

  def test_global_skip_validation_override(self):
    """Test that local override takes precedence over global skip."""
    get_config().skip_validation = True

    @validate(skip_validation_by_default=False)
    def process(_data: Validated[pd.Series, Finite]):
      return True

    # Should raise because we forced validation locally
    with pytest.raises(ValueError, match=r"(Finite|IgnoringNaNs)"):
      process(pd.Series([float("inf")]))

  def test_global_warn_only(self):
    """Test that global warn_only works."""
    get_config().warn_only = True

    @validate
    def process(_data: Validated[pd.Series, Finite]):
      return True

    # Should return None (and log error) instead of raising
    assert process(pd.Series([float("inf")])) is None

  def test_global_warn_only_override(self):
    """Test that local override takes precedence over global warn_only."""
    get_config().warn_only = True

    @validate(warn_only_by_default=False)
    def process(_data: Validated[pd.Series, Finite]):
      return True

    # 1. Global warn=True, Explicit warn=False (should raise)
    get_config().warn_only = True
    with pytest.raises(ValueError, match=r"(Finite|IgnoringNaNs)"):
      process(pd.Series([float("inf")]), warn_only=False)

  def test_kwargs_override_global(self):
    """Test that function call kwargs override global config."""
    get_config().skip_validation = True

    @validate
    def process(_data: Validated[pd.Series, Finite]):
      return True

    # Should raise ValueError because we force validation
    with pytest.raises(ValueError, match=r"(Finite|IgnoringNaNs)"):
      process(pd.Series([float("inf")]), skip_validation=False)

  def test_config_overrides(self):
    """Test that config.overrides() context manager works."""
    assert get_config().skip_validation is False

    with overrides(skip_validation=True):
      assert get_config().skip_validation is True

      @validate
      def process(_data: Validated[pd.Series, Finite]):
        return True

      # Should not raise
      assert process(pd.Series([float("inf")])) is True

    # Should reset after context
    assert get_config().skip_validation is False

    with (
      pytest.raises(AttributeError, match="Config has no attribute 'non_existent'"),
      overrides(non_existent=True),
    ):
      pass
