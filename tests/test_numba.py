import numpy as np
import pandas as pd
import pytest

from datawarden import Validated, validate
from datawarden.exceptions import ValidationError
from datawarden.validators.numeric import Gt, Lt
from datawarden.validators.structural import Column


def test_numba_fusion_execution() -> None:
  # Between 0 and 10, using Gt and Lt which should be fused
  @validate
  def func(df: Validated[pd.DataFrame, Column("a", Gt(0), Lt(10))]) -> bool:
    del df
    return True

  # Pass
  func(pd.DataFrame({"a": [1, 5, 9]}))

  # Fail
  with pytest.raises(ValidationError) as excinfo:
    func(pd.DataFrame({"a": [0, 5, 11]}))

  # Error message should come from Column('a', FusedValidator)
  msg = str(excinfo.value)
  assert "(>0.0 & <10.0)" in msg


def test_numba_large_data() -> None:
  # Ensure it works with larger datasets where Numba shine
  data = np.random.randn(100_000).clip(-4, 4)
  df = pd.DataFrame({"a": data})

  @validate
  def check(df: Validated[pd.DataFrame, Gt(-5), Lt(5)]) -> bool:
    del df
    return True

  check(df)

  # Check fallback or error handling
  df_str = pd.DataFrame({"a": ["a", "b", "c"]})

  @validate
  def check_str(df: Validated[pd.DataFrame, Gt(0)]) -> bool:
    del df
    return True

  # Should raise TypeError from Numba backend or ValidationError fallback
  with pytest.raises((ValidationError, TypeError)):
    check_str(df_str)


def test_numba_root_fusion() -> None:
  @validate
  def func(df: Validated[pd.DataFrame, Gt(0), Lt(10)]) -> bool:
    del df
    return True

  # Fail
  with pytest.raises(ValidationError) as excinfo:
    func(pd.DataFrame({"a": [0, 5], "b": [1, 11]}))

  msg = str(excinfo.value)
  assert "(>0.0 & <10.0)" in msg
