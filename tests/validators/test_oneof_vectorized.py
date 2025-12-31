import numpy as np
import pandas as pd
import pytest

from datawarden import IgnoringNaNs, Not, OneOf, Validated, validate


class TestOneOfVectorized:
  def test_oneof_vectorized_direct(self):
    """Test OneOf directly via validate_vectorized protocol."""
    v = OneOf("a", "b")
    s = pd.Series(["a", "b", "c", np.nan])
    mask = v.validate_vectorized(s)
    # Expect: True, True, False, True (NaN ignored)
    assert mask.tolist() == [True, True, False, True]

  def test_not_oneof(self):
    """Test Not(OneOf(...)) which relies on vectorization."""

    @validate
    def func(data: Validated[pd.Series, Not(OneOf("bad", "worse"))]):
      return data

    # "bad" and "worse" should fail.
    # "good" should pass.
    # NaN? OneOf allows NaN. Not(OneOf) should disallow it?
    # Not(OneOf) -> ~ (isin | nan) -> ~isin & ~nan.
    # So NaN should FAIL.

    valid = pd.Series(["good", "ok"])
    assert func(valid).equals(valid)

    invalid = pd.Series(["bad", "good"])
    with pytest.raises(ValueError, match="must not OneOf"):
      func(invalid)

    invalid_nan = pd.Series(["good", np.nan])
    with pytest.raises(ValueError, match="Cannot validate not OneOf"):
      func(invalid_nan)

  def test_fused_oneof(self):
    """Test OneOf works in fusion (IgnoringNaNs(OneOf))."""

    # Although OneOf already ignores NaNs, wrapping it keeps it in fusion chain.
    @validate
    def func(data: Validated[pd.Series, IgnoringNaNs(OneOf("a", "b"))]):
      return data

    s = pd.Series(["a", np.nan])
    assert func(s).equals(s)
