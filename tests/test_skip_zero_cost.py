import pandas as pd

from datawarden import Validated, config, validate
from datawarden.validators.numeric import Gt


def test_skip_validation_is_zero_cost_unwrapped():
  """
  Verifies that when skip_validation is True at decoration time,
  the decorator returns the original function directly (zero overhead).
  """

  with config.Overrides(skip_validation=True):

    @validate
    def skipped_func(df: Validated[pd.DataFrame, Gt(0)]) -> pd.Series:
      return df.sum()

  # The validate decorator should return the function as is
  # so it should NOT have the __wrapped__ attribute usually added by @wraps
  # or any other wrapper logic.
  assert not hasattr(skipped_func, "__wrapped__"), (
    "Function should be unwrapped when skip_validation is True"
  )

  # Verify it actually works normally
  df = pd.DataFrame({"a": [1, 2, 3]})
  assert skipped_func(df).sum() == 6

  # Verify it doesn't validate even if we pass bad data (because it's the raw function)
  bad_df = pd.DataFrame({"a": [-1, -2, -3]})
  # This would normally raise ValidationError if wrapped
  assert skipped_func(bad_df).sum() == -6


def test_normal_validation_is_wrapped():
  """
  Sanity check that validation IS wrapped by default.
  """

  @validate
  def normal_func(df: Validated[pd.DataFrame, Gt(0)]) -> pd.Series:
    return df.sum()

  assert hasattr(normal_func, "__wrapped__"), (
    "Normal validated function should be wrapped"
  )
