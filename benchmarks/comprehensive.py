from loguru import logger
import numpy as np
import pandas as pd

from benchmarks.utils import run_benchmark
from datawarden import (
  And,
  Between,
  Column,
  Dtype,
  Eq,
  Finite,
  Ge,
  Gt,
  IsNaN,
  Le,
  Lt,
  MaxDiff,
  MonoUp,
  Ne,
  Negative,
  Not,
  NotEmpty,
  NotNaN,
  OneOf,
  Or,
  Outside,
  Positive,
  Shape,
  Unique,
)
from datawarden.context import ValidationContext
from datawarden.validators.base import BaseValidator


def run_comprehensive() -> None:
  logger.info("=" * 60)
  logger.info("COMPREHENSIVE VALIDATOR BENCHMARKS (N=10,000)")
  logger.info("=" * 60)

  n = 10_000
  # Data setup
  d = {}
  d["float"] = pd.Series(np.random.uniform(0.1, 100.0, n))
  d["int"] = pd.Series(np.random.randint(1, 100, n))
  d["neg"] = pd.Series(np.random.uniform(-100.0, -0.1, n))
  d["nan"] = d["float"].copy()
  d["nan"].iloc[::10] = np.nan

  d["dt"] = pd.Series(pd.date_range("2024-01-01", periods=n, freq="1min"))
  d["mono_up"] = pd.Series(np.cumsum(np.abs(np.random.randn(n))))
  d["mono_down"] = pd.Series(-np.cumsum(np.abs(np.random.randn(n))))
  d["unique"] = pd.Series(np.arange(n, dtype=float))

  d["df"] = pd.DataFrame({
    "a": np.random.uniform(0, 100, n),
    "b": np.random.uniform(0, 100, n),
  })

  def bench(name: str, validator: BaseValidator, data_key: str) -> None:
    data = d[data_key]
    ctx = ValidationContext(root_data=data)
    run_benchmark(name, lambda: validator.validate(data, ctx), iterations=500)

  logger.info("\n--- Numeric ---")
  bench("Gt(0)", Gt(0), "float")
  bench("Ge(0)", Ge(0), "float")
  bench("Lt(200)", Lt(200), "float")
  bench("Le(200)", Le(200), "float")
  bench("Eq(50)", Eq(50), "int")
  bench("Ne(0)", Ne(0), "int")
  bench("IsNaN", IsNaN(), "nan")
  bench("NotNaN", NotNaN(), "float")
  bench("Finite", Finite(), "float")

  logger.info("\n--- Value ---")
  bench("Between(10, 90)", Between(10, 90), "float")
  bench("Outside(-1000, 1000)", Outside(-1000, 1000), "float")
  bench("OneOf(range(100))", OneOf(*range(1, 101)), "int")

  logger.info("\n--- Structural ---")
  bench("NotEmpty", NotEmpty(), "float")
  bench("Shape(rows=10k)", Shape(rows=n), "float")
  bench("Dtype(float64)", Dtype("float64"), "float")
  bench("Column('a', Ge(0))", Column("a", Ge(0)), "df")

  logger.info("\n--- Sequence ---")
  bench("Unique", Unique(), "unique")
  bench("MonoUp", MonoUp(), "mono_up")
  bench("MaxDiff(1000)", MaxDiff(1000), "float")

  logger.info("\n--- Logic (And) ---")
  bench("And(Ge(0))", And(Ge(0)), "float")
  bench("And(Ge(-10), Le(200))", And(Ge(-10), Le(200)), "float")
  bench("And(5 validators)", And(Ge(-10), Le(200), Gt(-20), Lt(300), Finite()), "float")

  logger.info("\n--- Logic (Or) ---")
  bench("Or(Positive)", Or(Positive()), "float")
  bench("Or(Positive, IsNaN)", Or(Positive(), IsNaN()), "nan")
  bench("Or(5 validators)", Or(Lt(-100), Lt(-50), Gt(150), Gt(200), IsNaN()), "nan")

  logger.info("\n--- Logic (Not) ---")
  bench("Not(Negative)", Not(Negative()), "float")
  bench("Not(IsNaN)", Not(IsNaN()), "float")


if __name__ == "__main__":
  run_comprehensive()
