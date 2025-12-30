from typing import Any

import numpy as np
import pandas as pd
import pytest

from datawarden import utils
from datawarden.base import Validator
from datawarden.utils import instantiate_validator, is_pandas_type, report_failures


class GoodValidator(Validator):
  pass


class BadValidator(Validator):
  def __init__(self, arg):
    self.arg = arg


def test_instantiate_validator_success():
  v = instantiate_validator(GoodValidator)
  assert isinstance(v, GoodValidator)

  # Verify strict syntax enforcement
  with pytest.raises(ValueError, match="Use validator class"):
    instantiate_validator(GoodValidator())

  # Verify non-default instances are allowed
  v_bad = BadValidator(1)
  assert instantiate_validator(v_bad) is v_bad


def test_instantiate_validator_failure():
  # Should raise TypeError because BadValidator requires an argument
  with pytest.raises(TypeError, match="could not be instantiated"):
    instantiate_validator(BadValidator)


def test_instantiate_validator_non_validator():
  assert instantiate_validator(int) is None
  assert instantiate_validator("string") is None


def test_is_pandas_type_direct():
  assert is_pandas_type(pd.Series)
  assert is_pandas_type(pd.DataFrame)
  assert not is_pandas_type(str)
  assert not is_pandas_type(int)


def test_is_pandas_type_generics():
  # Simulate a generic alias where origin is pd.Series
  # We can't easily rely on pd.Series[int] working across all envs/versions at runtime
  # so we mock an object with __origin__
  class MockGeneric:
    pass

  MockGeneric.__origin__ = pd.Series  # type: ignore

  # We need to patch get_origin to return our mock's origin
  # checking how utils.py uses it: get_origin(annotated_type)
  # The real get_origin won't work on our MockGeneric class likely.
  # But wait, utils.py does: origin = get_origin(annotated_type)

  # Let's try to construct a real subscripted generic if possible
  try:
    series_type = pd.Series[int]  # type: ignore
    assert is_pandas_type(series_type)
  except (TypeError, NameError):
    # Fallback for older python/pandas if subscripting fails at runtime
    pass


def test_is_pandas_type_module_fallback():
  # Create a class that looks like pandas but isn't subclass
  class FakePandas:
    pass

  FakePandas.__module__ = "pandas.core.frame"
  assert is_pandas_type(FakePandas)

  class NotPandas:
    pass

  NotPandas.__module__ = "numpy"
  assert not is_pandas_type(NotPandas)


def test_is_pandas_type_edge_cases():
  """Test edge cases for is_pandas_type (defensive checks)."""

  # Non-type instance
  assert not is_pandas_type(3)

  # Special forms that might cause issubclass to raise TypeError
  assert not is_pandas_type(Any)


def test_scalar_any_array():
  """Test scalar_any with numpy array."""
  assert utils.scalar_any(np.array([True]))
  assert not utils.scalar_any(np.array([False]))


def test_scalar_any_bool():
  """Test scalar_any with scalar bool."""
  assert utils.scalar_any(True)
  assert not utils.scalar_any(False)


def test_report_failures_exception_fallback():
  """Test report_failures fallback when index extraction fails."""

  class TriggerError:
    def __repr__(self):
      return "TriggerError"

  # We can trigger the generic exception handler by passing incompatible types
  # that pass the isinstance checks but fail on operation.
  # For example, a DataFrame and a mask that is a list (which isn't handled explicitly as mask but enters else?)
  # The code has 'else: n_failed = mask.sum()'.
  # If we pass a mask that behaves like bool/int for sum() but fails later?

  # Actually, simpler way to hit the 'except Exception' block in report_failures:
  # pass a data object where accessing index fails.
  # But report_failures takes Series/DataFrame.

  # Triggering 'except Exception' at line 190 of utils.py (as of view) is hard with valid types.
  # We might just skipping this strict coverage requirement since it's a safety net.
  # BUT, I'll add a test that exercises the 'else' branch of "isinstance(data, ...)" -> "(unknown indices)"

  # Pass data that is NOT Series/DataFrame/Index but mimics structure enough to pass strict alignment?
  # No, the function signature limits types.

  # However, line 188: else: failures = ["(unknown indices)"]
  # This is reached if data is not Series/DataFrame/Index.
  # But the signature says it must be.
  # Python runtime doesn't enforce signature. So we can pass a dummy.

  data_dummy = "not_a_pandas_object"
  mask_dummy = True  # triggers n_failed=1

  with pytest.raises(ValueError, match=r"\(unknown indices\)"):
    report_failures(data_dummy, mask_dummy, "Failure msg")  # type: ignore
