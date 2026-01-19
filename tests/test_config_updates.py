from collections.abc import Generator

import numpy as np
import pandas as pd
import pytest

from datawarden import config
from datawarden.backends.numba import run_numba_validation
from datawarden.validators import Gt


@pytest.fixture(autouse=True)
def reset_config() -> Generator[None, None, None]:
  """Ensure config is reset after each test."""
  original = config.get_config()
  yield
  config._config_var.set(original)


def test_config_set_helper() -> None:
  """Test that datawarden.config.set updates the global configuration."""
  original = config.get_config()
  assert original.warn_only is False

  config.set(warn_only=True, numba_threshold=999)

  current = config.get_config()
  assert current.warn_only is True
  assert current.numba_threshold == 999
  # Check other fields remain same
  assert current.use_numba == original.use_numba


def test_fail_fast_config_true() -> None:
  """When fail_fast=True (default), failure returns None mask."""
  config.set(use_numba=True, fail_fast=True, numba_threshold=0)

  df = pd.DataFrame({"a": [10, 0, 10]})  # Middle is invalid
  validator = Gt(5)

  # We must invoke run_numba_validation directly to check the low-level return
  # because ValidateResult wraps it.
  success, mask = run_numba_validation(df["a"], [validator])

  assert success is False
  assert mask is None


def test_fail_fast_config_false() -> None:
  """When fail_fast=False, failure returns valid boolean mask."""
  config.set(use_numba=True, fail_fast=False, numba_threshold=0)

  df = pd.DataFrame({"a": [10, 0, 10]})
  validator = Gt(5)

  success, mask = run_numba_validation(df["a"], [validator])

  assert success is False
  assert mask is not None
  assert isinstance(mask, np.ndarray)
  assert not mask.all()
  assert not mask[1]
  assert mask[0]
  assert mask[2]


def test_context_overrides_set_helper() -> None:
  """Test interaction between config.set and with config.Overrides()."""
  config.set(warn_only=True, fail_fast=True)

  original = config.get_config()
  assert original.warn_only is True
  assert original.fail_fast is True

  with config.Overrides(warn_only=False, fail_fast=False):
    current = config.get_config()
    assert current.warn_only is False
    assert current.fail_fast is False
    # Check unmodified remains
    assert current.use_numba == original.use_numba

  # Should revert to what .set() established
  after = config.get_config()
  assert after.warn_only is True
  assert after.fail_fast is True


def test_context_overrides_nested() -> None:
  config.set(warn_only=True)

  with config.Overrides(warn_only=False):
    assert config.get_config().warn_only is False

    with config.Overrides(fail_fast=False):
      c = config.get_config()
      assert c.warn_only is False  # inherited from outer context
      assert c.fail_fast is False

  assert config.get_config().warn_only is True
