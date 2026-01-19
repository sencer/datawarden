from typing import Annotated

import pytest

from datawarden import Overrides, get_config, validate
from datawarden.exceptions import ValidationError
from datawarden.validators import Gt


def test_basic_override() -> None:
  original = get_config()
  with Overrides(use_numba=not original.use_numba):
    assert get_config().use_numba != original.use_numba
  assert get_config().use_numba == original.use_numba


def test_nested_overrides() -> None:
  original = get_config()
  orig_numba = original.use_numba

  # Outer override
  with Overrides(use_numba=not orig_numba):
    assert get_config().use_numba != orig_numba

    # Inner override - change something else
    with Overrides(max_workers=999):
      assert get_config().use_numba != orig_numba  # Should still be overridden
      assert get_config().max_workers == 999

    # Back to outer
    assert get_config().use_numba != orig_numba
    assert get_config().max_workers == original.max_workers

  # Back to original
  assert get_config().use_numba == orig_numba


def test_nested_overrides_same_key() -> None:
  original = get_config()
  orig_thresh = original.numba_threshold

  with Overrides(numba_threshold=100):
    assert get_config().numba_threshold == 100

    with Overrides(numba_threshold=200):
      assert get_config().numba_threshold == 200

    assert get_config().numba_threshold == 100

  assert get_config().numba_threshold == orig_thresh


def test_simulated_import_context() -> None:
  """
  Simulates importing a module inside an override context.
  With sticky behavior removed, defining inside an override should NOT persist.
  """

  @validate
  def func_a(x: Annotated[int, Gt(-100)]) -> int:
    return x

  with Overrides(warn_only=True):

    @validate
    def func_b(x: Annotated[int, Gt(-100)]) -> int:
      return x

  # Both should raise (default) because warn_only=True was only during definition of func_b
  with pytest.raises(ValidationError):
    func_a("not an int")

  with pytest.raises(ValidationError):
    func_b("not an int")

  # Both should warn if CALLED inside an override
  with Overrides(warn_only=True):
    with pytest.warns(UserWarning):
      func_a("not an int")
    with pytest.warns(UserWarning):
      func_b("not an int")
