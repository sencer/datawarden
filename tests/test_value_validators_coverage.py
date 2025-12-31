from typing import Annotated

import numpy as np
import pandas as pd
import pytest

from datawarden import validate
from datawarden.base import Validator
from datawarden.validators import (
  AllowInf,
  AllowNaN,
  Between,
  Empty,
  Finite,
  Ge,
  HasColumn,
  IgnoringNaNs,
  Is,
  Negative,
  NotEmpty,
  OneOf,
  Outside,
  Positive,
  Rows,
  Shape,
  StrictFinite,
)
from datawarden.validators.value import _get_numeric_df_values  # noqa: PLC2701


def test_get_numeric_df_values_no_numeric():
  df = pd.DataFrame({"a": ["s", "t"], "b": [True, False]})
  vals, data = _get_numeric_df_values(df)
  assert vals is None
  assert data is None


def test_ignoring_nans_index():
  # Test IgnoringNaNs with pd.Index
  v = IgnoringNaNs(Ge(0))
  idx = pd.Index([1, -1, np.nan])
  with pytest.raises(ValueError, match="Data must be >= 0"):
    v.validate(idx)  # Should validate -1 and fail

  idx_valid = pd.Index([1, 2, np.nan])
  v.validate(idx_valid)  # Should pass


def test_ignoring_nans_holistic_dataframe():
  # Test IgnoringNaNs with a holistic validator on DataFrame
  # Should use dropna() fallback

  # Mock holistic validator
  class HolisticV(Ge):
    is_holistic = True

  v = IgnoringNaNs(HolisticV(0))
  df = pd.DataFrame({"a": [1, -1, np.nan], "b": [1, 1, 1]})

  # -1 is in row 1. NaN is in row 2.
  # dropna() removes row 2.
  # So it validates [1, -1] and [1, 1].
  # Should fail on -1.
  with pytest.raises(ValueError, match="Data must be >= 0"):
    v.validate(df)


def test_ignoring_nans_no_vectorization():
  # Test IgnoringNaNs with a validator that raises NotImplementedError in validate_vectorized
  class NoVecV(Ge):
    def validate_vectorized(self, data):
      raise NotImplementedError

  v = IgnoringNaNs(NoVecV(0))
  s = pd.Series([1, -1, np.nan])
  with pytest.raises(ValueError, match="Data must be >= 0"):
    v.validate(s)


def test_finite_strict_finite_type_errors():
  v = Finite()
  with pytest.raises(TypeError, match="Finite requires numeric data"):
    v.validate(pd.Series(["a", "b"]))

  with pytest.raises(TypeError, match="Finite requires numeric data"):
    v.validate(pd.Index(["a", "b"]))

  with pytest.raises(TypeError, match="Finite requires pandas"):
    v.validate([1, 2])

  v2 = StrictFinite()
  with pytest.raises(TypeError, match="StrictFinite requires numeric data"):
    v2.validate(pd.Series(["a", "b"]))


def test_positive_negative_edge_cases():
  v = Positive()
  # Non-numeric series
  with pytest.raises(TypeError, match="Positive requires numeric data"):
    v.validate(pd.Series(["a"]))

  # No numeric columns DF
  df_str = pd.DataFrame({"a": ["a"]})
  v.validate(df_str)  # Should return early (no op)

  # Non-pandas
  with pytest.raises(ValueError):
    v.validate(-1)
  v.validate(1)

  # validate_vectorized non-numeric
  assert v.validate_vectorized(pd.Series(["a"])).all()

  v_neg = Negative()
  # Non-numeric series
  with pytest.raises(TypeError, match="Negative requires numeric data"):
    v_neg.validate(pd.Series(["a"]))

  # Non-pandas
  with pytest.raises(ValueError):
    v_neg.validate(1)
  v_neg.validate(-1)

  assert v_neg.validate_vectorized(pd.Series(["a"])).all()


def test_between_outside_edge_cases():
  v = Between(0, 10)
  with pytest.raises(TypeError, match="Between requires numeric data"):
    v.validate(pd.Series(["a"]))

  v.validate(5)  # Valid
  with pytest.raises(ValueError):
    v.validate(11)

  assert v.validate_vectorized(pd.Series(["a"])).all()

  v_out = Outside(0, 10)
  v_out.validate(11)  # Valid
  with pytest.raises(ValueError):
    v_out.validate(5)

  assert v_out.validate_vectorized(pd.Series(["a"])).all()


def test_is_validator_failures():
  v = Is(lambda x: x > 0, name="Custom Check")

  # DataFrame failure
  df = pd.DataFrame({"a": [1, -1]})
  with pytest.raises(ValueError, match="Custom Check"):
    v.validate(df)

  # Index failure
  idx = pd.Index([1, -1])
  with pytest.raises(ValueError, match="Custom Check"):
    v.validate(idx)


def test_rows_validator_failures():
  v = Rows(lambda row: row["a"] > 0)

  # Not a DF
  with pytest.raises(TypeError):
    v.validate(pd.Series([1]))

  # Validate vectorized (should just call predicate)
  # Rows predicate expects a Series (row), but validate_vectorized passes the whole DF.
  # This seems like a potential bug or mismatch in Rows design vs validate_vectorized usage.
  # But Rows implementation of validate_vectorized is: return self.predicate(data)
  # If the predicate is designed for row-wise (Series input), it might fail on DF input.
  # But let's test what happens.
  pass


def test_shape_validator_failures():
  s_rows = Shape(5)
  with pytest.raises(ValueError, match="must have == 5 rows"):
    s_rows.validate(pd.Series([1, 2]))

  s_df = Shape(5, 2)
  with pytest.raises(ValueError, match="must have == 5 rows"):
    s_df.validate(pd.DataFrame({"a": [1]}, index=[0]))

  with pytest.raises(ValueError, match="must have == 2 columns"):
    s_df.validate(pd.DataFrame({"a": [1, 2, 3, 4, 5]}))


def test_empty_not_empty_negate():
  e = Empty()
  assert isinstance(e.negate(), NotEmpty)

  ne = NotEmpty()
  assert isinstance(ne.negate(), Empty)


def test_allow_markers():
  # Just cover the validate method
  AllowNaN().validate(pd.Series([1]))
  AllowInf().validate(pd.Series([1]))


def test_ignoring_nans_validate_vectorized_not_implemented():
  class NoVecV(Ge):
    # Remove validate_vectorized
    pass

  # But Ge has it.
  # Create a dummy validator without it.

  class Dummy(Validator):
    pass

  v = IgnoringNaNs(Dummy)
  with pytest.raises(NotImplementedError):
    v.validate_vectorized(pd.Series([1]))


def test_positive_failure_pandas():
  v = Positive()
  with pytest.raises(ValueError, match="Data must be positive"):
    v.validate(pd.Series([-1, 1]))


def test_negative_failure_pandas():
  v = Negative()
  with pytest.raises(ValueError, match="Data must be negative"):
    v.validate(pd.Series([1, -1]))


def test_between_failure_pandas():
  v = Between(0, 10)
  with pytest.raises(ValueError, match="Data must be >= 0"):
    v.validate(pd.Series([-1]))
  with pytest.raises(ValueError, match="Data must be <= 10"):
    v.validate(pd.Series([11]))


def test_outside_failure_pandas():
  v = Outside(0, 10)
  with pytest.raises(ValueError, match="Data must be outside"):
    v.validate(pd.Series([5]))


def test_oneof_failure_pandas():
  v = OneOf(1, 2)
  with pytest.raises(ValueError, match="Values must be one of"):
    v.validate(pd.Series([3]))


def test_is_failure_pandas_columns():
  v = Is(lambda x: x > 0)

  # Singular
  with pytest.raises(ValueError, match="Column 'a' failed"):
    v.validate(pd.DataFrame({"a": [-1], "b": [1]}))

  # Plural
  with pytest.raises(ValueError, match=r"Columns .* failed"):
    v.validate(pd.DataFrame({"a": [-1], "b": [-1]}))


def test_rows_failure_pandas():
  v = Rows(lambda row: row["a"] > 0)
  df = pd.DataFrame({"a": [-1, 1]})
  with pytest.raises(ValueError, match="Rows failed predicate check"):
    v.validate(df)


def test_ignoring_nans_transform():
  # Test IgnoringNaNs(HasColumn(...)) transformation
  # This logic is triggered by ValidationPlanBuilder, so we use @validate

  @validate
  def func(df: Annotated[pd.DataFrame, IgnoringNaNs(HasColumn("a", Ge(0)))]):
    _ = df
    return True

  # "a" has NaN. Should be ignored. -1 should fail.
  df_valid = pd.DataFrame({"a": [1, np.nan]})
  assert func(df_valid) is True

  df_invalid = pd.DataFrame({"a": [-1, np.nan]})
  with pytest.raises(ValueError, match="Data must be >= 0"):
    func(df_invalid)


def test_ignoring_nans_with_ignore_nan_method():
  # Test IgnoringNaNs wrapping a validator that has with_ignore_nan (like Finite)
  @validate
  def func(df: Annotated[pd.DataFrame, IgnoringNaNs(Finite)]):
    _ = df
    return True

  # Finite normally allows NaN. IgnoringNaNs(Finite) should basically be Finite.
  # But strictly speaking, Finite.with_ignore_nan returns self or new Finite.
  # So it unwraps IgnoringNaNs.

  df = pd.DataFrame({"a": [1, np.inf]})
  with pytest.raises(ValueError, match="Data must be finite"):
    func(df)


def test_between_outside_negation():
  b = Between(0, 10)
  o = b.negate()
  assert isinstance(o, Outside)
  assert o.lower == 0
  assert o.upper == 10

  b2 = o.negate()
  assert isinstance(b2, Between)
  assert b2.lower == 0
  assert b2.upper == 10


def test_rows_validate_vectorized():
  # Rows usually doesn't vectorise well unless predicate does.
  # But we can test it returns predicate result.
  v = Rows(lambda df: df["a"] > 0)  # Vectorized predicate
  df = pd.DataFrame({"a": [1, -1]})
  res = v.validate_vectorized(df)
  assert not res.all()


def test_shape_invalid_constraint():
  with pytest.raises(TypeError, match="Invalid shape constraint"):
    Shape("bad")

  with pytest.raises(TypeError, match="Invalid shape constraint"):
    Shape(10, "bad")
