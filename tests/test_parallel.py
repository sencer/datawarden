import pandas as pd

from datawarden import Overrides, Validated, validate
from datawarden.validators.numeric import Gt
from datawarden.validators.structural import Column


def test_parallel_execution() -> None:
  df = pd.DataFrame({"a": range(20), "b": range(20)})

  @validate
  def process(
    _df: Validated[pd.DataFrame, Column("a", Gt(-1)), Column("b", Gt(-1))],
  ) -> bool:
    return True

  # Trigger parallel path
  with Overrides(parallel_threshold_rows=10):
    process(df)

  # Trigger serial path
  with Overrides(parallel_threshold_rows=100):
    process(df)
