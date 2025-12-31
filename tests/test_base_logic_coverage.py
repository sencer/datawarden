from typing import Any

import pandas as pd
import pytest

from datawarden.base import CompoundValidator, Validator
from datawarden.validators import Ge, Not


class FlakyVectorized(Validator[Any]):
  priority = 10

  def validate_vectorized(self, data):
    # Claim to support it but fail
    raise NotImplementedError("Oops")

  def validate(self, data):
    # Fallback should call this
    if data == "fail":
      raise ValueError("Failed in fallback")


def test_compound_validator_fallback():
  # Force fusion by using consecutive vectorizable validators
  # But checking Fusion logic in plan.py is hard to control directly via @validate
  # unless we use a custom validator that PlanBuilder thinks is vectorizable.

  # But we can instantiate CompoundValidator directly.
  v = CompoundValidator([FlakyVectorized()])

  # Should fall back to validate()
  v.validate("ok")  # Should pass

  with pytest.raises(ValueError, match="Failed in fallback"):
    v.validate("fail")


def test_not_validator_logic():
  # Not(Ge(0)) -> < 0
  v = Not(Ge(0))
  v.validate(-1)
  with pytest.raises(ValueError):
    v.validate(1)

  # Vectorized
  assert v.validate_vectorized(pd.Series([-1])).all()
  assert not v.validate_vectorized(pd.Series([1])).all()

  # Double negation optimization
  _ = Not(Not(Ge(0)))


def test_not_validator_generic_fallback():
  # Use a validator that does NOT have negate()
  class NoNegate(Validator):
    def validate(self, data):
      if data == "valid":
        pass
      else:
        raise ValueError("Invalid")

  v = Not(NoNegate())

  # Not(NoNegate) means: if NoNegate PASSES, Not FAILS.
  # If NoNegate FAILS, Not PASSES.

  v.validate("invalid")  # NoNegate raises -> Not catches -> Valid

  with pytest.raises(ValueError, match="must not"):
    v.validate("valid")  # NoNegate passes -> Not raises
