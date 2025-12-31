import numpy as np
import pandas as pd
import pytest

from datawarden import Ge, IsNaN, Not, Validated, validate


def test_index_plan_domain_resolution():
  # If pd.Index uses Plan mode, Ge(0) and Ge(10) should be merged to Ge(10)

  @validate
  def process(idx: Validated[pd.Index, Ge(0), Ge(10)]):
    return idx

  # Should pass
  process(pd.Index([10, 11, 12]))

  # Should fail with Ge(10) error, not Ge(0)
  with pytest.raises(ValueError, match="must be >= 10"):
    process(pd.Index([5, 11, 12]))


def test_index_plan_global_nan():
  # If pd.Index uses Plan mode, Not(IsNaN) should be applied

  @validate
  def process(idx: Validated[pd.Index, Not(IsNaN)]):
    return idx

  # Should fail on NaN
  with pytest.raises(ValueError, match="Cannot validate not contain NaN with NaN"):
    process(pd.Index([1.0, np.nan, 3.0]))


if __name__ == "__main__":
  pytest.main([__file__])
