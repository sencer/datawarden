import pandas as pd
import pytest

from datawarden import Positive, Validated, validate
from datawarden.config import Overrides
from datawarden.exceptions import ValidationError


@validate
def positive_only(data: Validated[pd.Series, Positive]) -> pd.Series:
  return data


def test_warn_only() -> None:
  s = pd.Series([-1, 1])

  # Standard behavior: raises ValidationError
  with pytest.raises(ValidationError):
    positive_only(s)

  # warn_only=True: issues warning, returns data
  with Overrides(warn_only=True), pytest.warns(UserWarning, match=">0.0"):
    res = positive_only(s)
    assert res is s


def test_global_warn_only() -> None:
  s = pd.Series([-1, 1])

  with Overrides(warn_only=True), pytest.warns(UserWarning):
    positive_only(s)
