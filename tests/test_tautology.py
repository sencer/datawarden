import pandas as pd

from datawarden import Gt, Le, Or, Validated, validate
from datawarden.core import OptimizedPlan
from datawarden.validators.base import Pass


def test_tautology_optimization():
  # Or(Gt(0), ~Gt(0)) -> Or(Gt(0), Le(0)) -> Tautology
  v1 = Gt(0)
  v2 = Le(0)

  # Check if negate works as expected for this test
  assert v1.negate() == v2

  # OptimizedPlan should simplify Or(v1, v2)
  plan = OptimizedPlan([Or(v1, v2)])

  # It should have simplified to Pass()
  # Note: Or.transform() returns [Pass()]
  # OptimizedPlan._optimize calls v.decompose() then groups, then simplifies.
  # Wait, Or.transform() is currently NOT called by OptimizedPlan._optimize.
  # I should check where transform() should be called.

  # OptimizedPlan decomposes and then simplifies numeric bounds.
  # Logic tautologies might need a call to transform().

  # Let's see if Or(v1, v2) contains Pass after optimization
  found_pass = any(isinstance(v, Pass) for v in plan.validators)
  assert found_pass, f"Tautology not simplified: {plan.validators}"


def test_tautology_execution() -> None:
  @validate
  def func(_x: Validated[pd.Series, Or(Gt(0), Le(0))]) -> bool:
    return True

  # Should pass for anything
  assert func(pd.Series([-1, 0, 1])) is True
