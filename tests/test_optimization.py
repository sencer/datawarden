import pandas as pd
import pytest

from datawarden import Validated, validate
from datawarden.core import ValidationPlan
from datawarden.exceptions import LogicError, ValidationError
from datawarden.validators.numeric import Ge, Gt, Lt, Positive
from datawarden.validators.structural import Column


def test_bound_simplification() -> None:
  @validate
  def func(df: Validated[pd.DataFrame, Gt(0), Gt(5), Ge(10)]) -> pd.DataFrame:
    return df

  plan = ValidationPlan(func)
  opt_plan = plan.arg_plans["df"]

  # Gt(0), Gt(5), Ge(10) should simplify to Ge(10)
  # Plus IsInstance(pd.DataFrame)
  assert len(opt_plan.validators) == 2
  # Find the numeric one
  numeric_v = next(
    v for v in opt_plan.validators if not str(v).startswith("IsInstance")
  )
  assert ">=10" in str(numeric_v)


def test_not_nan_removal() -> None:
  @validate
  def func(df: Validated[pd.DataFrame, Positive]) -> pd.DataFrame:
    return df

  plan = ValidationPlan(func)
  opt_plan = plan.arg_plans["df"]

  # Positive decomposes to Gt(0) and NotNaN.
  # NotNaN should be removed because Gt(0) implies it for masks.
  # Total: IsInstance, Gt(0)
  assert len(opt_plan.validators) == 2
  assert any("NotNaN" in str(v) for v in opt_plan.validators) is False


def test_column_decomposition_and_simplification() -> None:
  @validate
  def func(
    df: Validated[pd.DataFrame, Column("a", Positive), Column("a", Gt(5))],
  ) -> pd.DataFrame:
    return df

  plan = ValidationPlan(func)
  opt_plan = plan.arg_plans["df"]

  # Column("a", Positive) -> Column("a", Gt(0)), Column("a", NotNaN)
  # Column("a", Gt(5)) -> Column("a", Gt(5))
  # Simplifies to: Column("a", Gt(5))
  # Plus IsInstance(pd.DataFrame)

  assert len(opt_plan.validators) == 2
  col_v = next(v for v in opt_plan.validators if str(v).startswith("Column"))
  assert ">5" in str(col_v)
  assert "NotNaN" not in str(col_v)


def test_execution_with_simplified_plan() -> None:
  @validate
  def func(_df: Validated[pd.DataFrame, Gt(0), Gt(10)]) -> bool:
    return True

  # Should only fail with >10 message if value is 5
  df = pd.DataFrame({"a": [5]})
  with pytest.raises(ValidationError) as excinfo:
    func(df)

  # It should NOT mention >0
  assert ">10" in str(excinfo.value)
  assert ">0" not in str(excinfo.value)


def test_implicit_override_logic() -> None:
  # Global: >10, Column A: <5. A=4 should pass.
  @validate
  def process(df: Validated[pd.DataFrame, Gt(10), Column("A", Lt(5))]) -> pd.DataFrame:
    return df

  process(pd.DataFrame({"A": [4], "B": [15]}))

  with pytest.raises(ValidationError):
    process(pd.DataFrame({"A": [12], "B": [15]}))


def test_contradiction_logic_error() -> None:
  with pytest.raises(LogicError):

    @validate
    def process(_df: Validated[pd.DataFrame, Gt(10), Lt(5)]) -> None:
      pass
