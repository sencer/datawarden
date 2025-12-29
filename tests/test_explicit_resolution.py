from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from datawarden import Validated, validate
from datawarden.validators import (
  Finite,
  Gt,
  HasColumn,
  HasColumns,
  IsDtype,
  Lt,
  Positive,
)

# =============================================================================
# 1. Explicit Override
# =============================================================================


def test_explicit_override_gt():
  @validate
  def process(df: Validated[pd.DataFrame, Gt(0), HasColumn("A", Gt(10))]):
    return df

  # Global Gt(0) applies to all columns.
  # HasColumn["A", Gt(10)] overrides Gt(0) for A.

  # A = 5 (>0 but <10) -> Should Fail Gt(10)
  # B = 5 (>0) -> Should Pass Gt(0)
  df_fail = pd.DataFrame({"A": [5], "B": [5]})

  with pytest.raises(ValueError, match="> 10"):
    process(df_fail)

  # A = 15 (>10) -> Pass
  # B = 5 (>0) -> Pass
  df_pass = pd.DataFrame({"A": [15], "B": [5]})
  process(df_pass)


# =============================================================================
# 2. Explicit Disable (Mutually Exclusive Global vs Local)
# =============================================================================


def test_explicit_disable_lt_add_gt():
  # Global Lt(1). Local A: Lt(None) [Disabled], Gt(5).
  @validate
  def process(df: Validated[pd.DataFrame, Lt(1), HasColumn("A", Lt(None), Gt(5))]):
    return df

  # A = 6 (>5, not <1) -> Should Pass because Lt is disabled for A.
  # B = 0 (<1) -> Should Pass.
  df = pd.DataFrame({"A": [6], "B": [0]})
  process(df)

  # B = 2 (not <1) -> Fail Lt(1)
  df_fail_b = pd.DataFrame({"A": [6], "B": [2]})
  with pytest.raises(ValueError, match="< 1"):
    process(df_fail_b)

  # A = 4 (not >5) -> Fail Gt(5)
  df_fail_a = pd.DataFrame({"A": [4], "B": [0]})
  with pytest.raises(ValueError, match="> 5"):
    process(df_fail_a)


# =============================================================================
# 3. Additive (Orthogonal Types)
# =============================================================================


def test_additive_checks():
  @validate
  def process(df: Validated[pd.DataFrame, Finite, HasColumn("A", Positive)]):
    return df

  # A checks Finite AND Positive.
  # B checks Finite.

  # Invalid A
  df_fail_a = pd.DataFrame({"A": [-1.0, 2.0], "B": [1.0, -1.0]})
  with pytest.raises(ValueError, match=r"(positive|Data must be > 0)"):
    process(df_fail_a)

  # B = Inf (Not Finite) -> Fail Finite
  df_fail_b = pd.DataFrame({"A": [1], "B": [np.inf]})
  with pytest.raises(ValueError, match="finite"):
    process(df_fail_b)


# =============================================================================
# 4. HasColumns (Plural)
# =============================================================================


def test_plural_has_columns_override():
  @validate
  def process(df: Validated[pd.DataFrame, Gt(0), HasColumns(["A", "B"], Gt(10))]):
    return df

  # A, B need > 10.
  # C needs > 0.

  df = pd.DataFrame({"A": [11], "B": [11], "C": [1]})
  process(df)

  df_fail = pd.DataFrame({
    "A": [5],  # Fails Gt(10)
    "B": [11],
    "C": [1],
  })
  with pytest.raises(ValueError, match="> 10"):
    process(df_fail)


# =============================================================================
# 5. Type-Based Resolution (Different Instances)
# =============================================================================


def test_type_based_resolution():
  # Global: IsDtype float
  # Local A: IsDtype int -> Should override.
  @validate
  def process(
    df: Validated[pd.DataFrame, IsDtype(float), HasColumn("A", IsDtype(int))],
  ):
    return df

  df = pd.DataFrame({
    "A": [1, 2],  # int
    "B": [1.0, 2.0],  # float
  })
  process(df)

  df_fail = pd.DataFrame({
    "A": [1.0, 2.0],  # float (Fail local)
    "B": [1.0, 2.0],
  })
  with pytest.raises(ValueError, match="int"):
    process(df_fail)


# =============================================================================
# 6. Global is "Default" (Implicitly applies to new columns)
# =============================================================================


def test_global_default():
  @validate
  def process(df: Validated[pd.DataFrame, Gt(0)]):
    return df

  # No HasColumn used. Gt(0) applies to everything.
  df = pd.DataFrame({"Z": [-1]})
  with pytest.raises(ValueError, match="> 0"):
    process(df)
