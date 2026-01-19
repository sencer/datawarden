"""Hypothesis property-based tests for datawarden validators.

These tests use property-based testing to verify invariants and edge cases
that might be missed by example-based testing.
"""

from hypothesis import assume, given, settings, strategies as st
import numpy as np
import pandas as pd
import pytest

from datawarden import Validated, validate
from datawarden.exceptions import ValidationError
from datawarden.validators import (
  Finite,
  Ge,
  Le,
  Negative,
  NonNegative,
  NonPositive,
  Positive,
)

# =============================================================================
# Strategies for generating test data
# =============================================================================

# Strategy for finite floats (no NaN, no Inf)
finite_floats = st.floats(
  allow_nan=False, allow_infinity=False, min_value=-1e10, max_value=1e10
)


# Strategy for Series of finite floats
def finite_series(min_size: int = 1, max_size: int = 100) -> st.SearchStrategy:
  return st.lists(finite_floats, min_size=min_size, max_size=max_size).map(pd.Series)


# =============================================================================
# Numeric Validator Properties
# =============================================================================


class TestNumericProperties:
  """Property-based tests for numeric validators."""

  @given(
    st.lists(
      st.floats(min_value=0.01, max_value=1e6, allow_nan=False, allow_infinity=False),
      min_size=1,
      max_size=50,
    )
  )
  def test_positive_accepts_all_positive(self, values: list[float]) -> None:
    """Positive validator should accept all positive values."""

    @validate
    def process(data: Validated[pd.Series, Positive]) -> pd.Series:
      return data

    series = pd.Series(values)
    result = process(series)
    assert result.equals(series)

  @given(
    st.lists(
      st.floats(min_value=-1e6, max_value=-0.01, allow_nan=False, allow_infinity=False),
      min_size=1,
      max_size=50,
    )
  )
  def test_negative_accepts_all_negative(self, values: list[float]) -> None:
    """Negative validator should accept all negative values."""

    @validate
    def process(data: Validated[pd.Series, Negative]) -> pd.Series:
      return data

    series = pd.Series(values)
    result = process(series)
    assert result.equals(series)

  @given(
    st.lists(
      st.floats(min_value=-1e6, max_value=0, allow_nan=False, allow_infinity=False),
      min_size=1,
      max_size=50,
    )
  )
  def test_nonpositive_accepts_all_nonpositive(self, values: list[float]) -> None:
    """NonPositive validator should accept all non-positive values."""

    @validate
    def process(data: Validated[pd.Series, NonPositive]) -> pd.Series:
      return data

    series = pd.Series(values)
    result = process(series)
    assert result.equals(series)

  @given(
    st.lists(
      st.floats(min_value=0, max_value=1e6, allow_nan=False, allow_infinity=False),
      min_size=1,
      max_size=50,
    )
  )
  def test_nonnegative_accepts_all_nonnegative(self, values: list[float]) -> None:
    """NonNegative validator should accept all non-negative values."""

    @validate
    def process(data: Validated[pd.Series, NonNegative]) -> pd.Series:
      return data

    series = pd.Series(values)
    result = process(series)
    assert result.equals(series)


# =============================================================================
# Comparison Validator Properties
# =============================================================================


class TestComparisonProperties:
  """Property-based tests for comparison validators."""

  @given(
    st.data(),
    st.floats(min_value=-1000, max_value=1000, allow_nan=False, allow_infinity=False),
  )
  def test_ge_semantics(self, data: st.DataObject, threshold: float) -> None:
    """Ge(x) should accept all values >= x."""
    values = data.draw(
      st.lists(
        st.floats(
          min_value=threshold, max_value=1e10, allow_nan=False, allow_infinity=False
        ),
        min_size=1,
        max_size=50,
      )
    )

    @validate
    def process(data: Validated[pd.Series, Ge(threshold)]) -> pd.Series:
      return data

    series = pd.Series(values)
    result = process(series)
    assert result.equals(series)

  @given(
    st.data(),
    st.floats(min_value=-1000, max_value=1000, allow_nan=False, allow_infinity=False),
  )
  def test_le_semantics(self, data: st.DataObject, threshold: float) -> None:
    """Le(x) should accept all values <= x."""
    values = data.draw(
      st.lists(
        st.floats(
          min_value=-1e10, max_value=threshold, allow_nan=False, allow_infinity=False
        ),
        min_size=1,
        max_size=50,
      )
    )

    @validate
    def process(data: Validated[pd.Series, Le(threshold)]) -> pd.Series:
      return data

    series = pd.Series(values)
    result = process(series)
    assert result.equals(series)


# =============================================================================
# Logic Composition Properties
# =============================================================================


class TestLogicProperties:
  """Property-based tests for logic composition."""

  @given(
    st.lists(
      st.floats(min_value=0.1, max_value=99.9, allow_nan=False, allow_infinity=False),
      min_size=1,
      max_size=50,
    )
  )
  @settings(max_examples=30)
  def test_and_commutativity(self, values: list[float]) -> None:
    """And(A, B) should be equivalent to And(B, A)."""
    series = pd.Series(values)

    @validate
    def process_ab(data: Validated[pd.Series, Ge(0), Le(100)]) -> pd.Series:
      return data

    @validate
    def process_ba(data: Validated[pd.Series, Le(100), Ge(0)]) -> pd.Series:
      return data

    result_ab = process_ab(series)
    result_ba = process_ba(series)
    assert result_ab.equals(result_ba)

  @given(st.lists(finite_floats, min_size=1, max_size=50))
  @settings(max_examples=30)
  def test_or_commutativity(self, values: list[float]) -> None:
    """Or(A, B) should be equivalent to Or(B, A)."""
    series = pd.Series(values)
    # Values that satisfy Positive OR Negative (i.e., not zero)
    assume(all(v != 0 for v in values))

    @validate
    def process_ab(data: Validated[pd.Series, Positive | Negative]) -> pd.Series:
      return data

    @validate
    def process_ba(data: Validated[pd.Series, Negative | Positive]) -> pd.Series:
      return data

    result_ab = process_ab(series)
    result_ba = process_ba(series)
    assert result_ab.equals(result_ba)


# =============================================================================
# Finite Validator Properties
# =============================================================================


class TestFiniteProperties:
  """Property-based tests for Finite validator."""

  @given(st.lists(finite_floats, min_size=1, max_size=100))
  def test_finite_accepts_all_finite(self, values: list[float]) -> None:
    """Finite should accept all finite values."""

    @validate
    def process(data: Validated[pd.Series, Finite]) -> pd.Series:
      return data

    series = pd.Series(values)
    result = process(series)
    assert result.equals(series)

  @given(st.lists(finite_floats, min_size=0, max_size=50))
  def test_finite_rejects_inf(self, values: list[float]) -> None:
    """Finite should reject Inf values."""
    values_with_inf = [*values, np.inf]

    @validate
    def process(data: Validated[pd.Series, Finite]) -> pd.Series:
      return data

    series = pd.Series(values_with_inf)
    with pytest.raises(ValidationError):
      process(series)

  @given(st.lists(finite_floats, min_size=0, max_size=50))
  def test_finite_rejects_nan(self, values: list[float]) -> None:
    """Finite should reject NaN values."""
    values_with_nan = [*values, np.nan]

    @validate
    def process(data: Validated[pd.Series, Finite]) -> pd.Series:
      return data

    series = pd.Series(values_with_nan)
    with pytest.raises(ValidationError):
      process(series)
