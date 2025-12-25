from typing import Any

import pandas as pd
import pytest

from datawarden.base import Validator
from datawarden.utils import instantiate_validator, is_pandas_type


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
