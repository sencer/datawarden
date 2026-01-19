import time
from typing import Any

import numpy as np
import pandas as pd
import pytest

from datawarden import Overrides
from datawarden.context import ValidationContext
from datawarden.validators.base import BaseValidator
from datawarden.validators.numeric import Ge, Le


def _time_validation(
  validator: BaseValidator[Any],
  data: pd.Series,
  use_numba: bool,
  iterations: int = 10,
) -> float:
  times = []
  ctx = ValidationContext(root_data=data)
  with Overrides(use_numba=use_numba):
    # Warmup
    for _ in range(3):
      validator.validate(data, ctx)
    # Measure
    for _ in range(iterations):
      start = time.perf_counter_ns()
      validator.validate(data, ctx)
      times.append(time.perf_counter_ns() - start)
  return min(times)


class TestNumbaPerformanceRegression:
  @pytest.fixture
  def large_data(self) -> pd.Series:
    return pd.Series(np.random.uniform(-0.5, 0.5, 10_000_000))

  @pytest.fixture
  def complex_validator(self) -> BaseValidator[pd.Series]:
    # Scenario that benefits from loop fusion
    return Ge(-2.0) & Le(2.0) & Ge(-1.5) & Le(1.5)

  def test_numba_faster_than_standard(
    self, large_data: pd.Series, complex_validator: BaseValidator[pd.Series]
  ) -> None:
    standard_time = _time_validation(complex_validator, large_data, use_numba=False)
    numba_time = _time_validation(complex_validator, large_data, use_numba=True)

    # We expect some speedup
    assert numba_time < standard_time
