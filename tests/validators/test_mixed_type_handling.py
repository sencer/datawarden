from typing import Annotated

import numpy as np
import pandas as pd
import pytest

from datawarden import Finite, IgnoringNaNs, Rows, StrictFinite, validate
from datawarden.plan import ValidationPlanBuilder


class TestMixedTypeHandling:
  def test_finite_dataframe_ignores_strings(self):
    """Test that Finite ignores string columns in DataFrames."""
    df = pd.DataFrame({"numeric": [1.0, 2.0, 3.0], "text": ["a", "b", "c"]})
    v = Finite()
    # Should pass without TypeError
    v.validate(df)

  def test_finite_dataframe_finds_inf_in_numeric_columns(self):
    """Test that Finite still finds Inf in numeric columns of mixed DataFrames."""
    df = pd.DataFrame({"numeric": [1.0, np.inf, 3.0], "text": ["a", "b", "c"]})
    v = Finite()
    with pytest.raises(ValueError, match="must be finite"):
      v.validate(df)

  def test_strict_finite_dataframe_ignores_strings(self):
    """Test that StrictFinite ignores string columns in DataFrames."""
    df = pd.DataFrame({"numeric": [1.0, 2.0, 3.0], "text": ["a", "b", "c"]})
    v = StrictFinite()
    # Should pass without TypeError
    v.validate(df)

  def test_finite_series_requires_numeric(self):
    """Test that Finite raises TypeError on non-numeric Series."""
    s = pd.Series(["a", "b", "c"])
    v = Finite()
    with pytest.raises(TypeError, match="requires numeric data"):
      v.validate(s)

  def test_ignoring_nans_rows_is_holistic(self):
    """Test that IgnoringNaNs(Rows) is correctly identified as holistic and not dropped."""

    @validate
    def process(df: Annotated[pd.DataFrame, IgnoringNaNs(Rows(lambda r: r.sum() > 0))]):
      return df

    # Inspect the plan
    builder = ValidationPlanBuilder(process)
    arg_validators, _ = builder.build()
    plan = arg_validators["df"]

    assert len(plan["slow_holistic"]) == 1
    assert isinstance(plan["slow_holistic"][0][0], IgnoringNaNs)
    assert isinstance(plan["slow_holistic"][0][0].wrapped, Rows)
    assert len(plan["default"]) == 0

  def test_ignoring_nans_rows_execution(self):
    """Test execution of IgnoringNaNs(Rows)."""

    @validate
    def process(df: Annotated[pd.DataFrame, IgnoringNaNs(Rows(lambda r: r.sum() > 5))]):
      return df

    # Row 0 sums to 6 (> 5)
    # Row 1 has NaN, should be dropped by IgnoringNaNs
    # Row 2 sums to 7 (> 5)
    df = pd.DataFrame({"a": [1, np.nan, 3], "b": [5, 2, 4]})

    # Should pass
    process(df)

    # Now make it fail
    df2 = pd.DataFrame({
      "a": [1, np.nan, 2],  # sum 6, NaN, sum 6
      "b": [5, 2, 2],  # Row 2 sum 4 (< 5)
    })
    with pytest.raises(ValueError, match="Rows failed predicate check"):
      process(df2)

  def test_finite_mixed_type_decorator(self):
    """Test that Finite on mixed-type DataFrame works via decorator."""

    @validate
    def func(df: Annotated[pd.DataFrame, Finite]):
      pass

    df = pd.DataFrame({"numeric": [1.0, 2.0], "text": ["a", "b"]})
    # Should pass (promoted to holistic or skipped via planning)
    func(df)

  def test_strict_finite_mixed_type_decorator(self):
    """Test that StrictFinite on mixed-type DataFrame works via decorator."""

    @validate
    def func(df: Annotated[pd.DataFrame, StrictFinite]):
      pass

    df = pd.DataFrame({"numeric": [1.0, 2.0], "text": ["a", "b"]})
    # Should pass
    func(df)
