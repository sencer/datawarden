from __future__ import annotations

import pandas as pd
import pytest

from datawarden import Validated, validate  # noqa: TC001
from datawarden.exceptions import LogicError
from datawarden.validators import Gt, HasColumn, Lt, OneOf  # noqa: TC001

# =============================================================================
# 1. Automatic Contradiction Resolution (Implicit Override)
# =============================================================================


def test_implicit_override_contradiction():
  # Global: Greater than 10
  # Local: Less than 5
  # Contradiction -> Local should win
  @validate
  def process(df: Validated[pd.DataFrame, Gt(10), HasColumn("A", Lt(5))]):
    return df

  # A = 4 (< 5, not > 10). Should Pass.
  df = pd.DataFrame({"A": [4]})
  process(df)

  # A = 12 (> 10, not < 5). Should Fail (Lt(5) is active).
  df_fail = pd.DataFrame({"A": [12]})
  with pytest.raises(ValueError, match="< 5"):
    process(df_fail)


# =============================================================================
# 2. Intersection (Overlap)
# =============================================================================


def test_intersection_overlap():
  # Global: Greater than 0
  # Local: Less than 10
  # Result: Between 0 and 10
  @validate
  def process(df: Validated[pd.DataFrame, Gt(0), HasColumn("A", Lt(10))]):
    return df

  # A = 5. Pass.
  df = pd.DataFrame({"A": [5]})
  process(df)

  # A = -1. Fail Gt(0)
  df_fail_low = pd.DataFrame({"A": [-1]})
  with pytest.raises(ValueError, match="> 0"):
    process(df_fail_low)

  # A = 11. Fail Lt(10)
  df_fail_high = pd.DataFrame({"A": [11]})
  with pytest.raises(ValueError, match="< 10"):
    process(df_fail_high)


# =============================================================================
# 3. Logic Errors (Declaration Time)
# =============================================================================


def test_local_contradiction_raises_error():
  # Local: Greater than 10 AND Less than 5 -> Impossible
  with pytest.raises(LogicError):

    @validate
    def process(df: Validated[pd.DataFrame, HasColumn("A", Gt(10), Lt(5))]):
      pass


def test_global_contradiction_raises_error():
  # Global: Greater than 10 AND Less than 5 -> Impossible
  with pytest.raises(LogicError):

    @validate
    def process(df: Validated[pd.DataFrame, Gt(10), Lt(5)]):
      pass


# =============================================================================
# 4. Set Intersection (OneOf)
# =============================================================================


def test_oneof_intersection():
  # Global: OneOf 'a', 'b', 'c'
  # Local: OneOf 'a', 'b'
  # Result: Subset 'a', 'b'
  @validate
  def process(
    df: Validated[
      pd.DataFrame,
      OneOf("a", "b", "c"),
      HasColumn("A", OneOf("a", "b")),
    ],
  ):
    return df

  df = pd.DataFrame({"A": ["a", "b"]})
  process(df)

  # "c" is in Global but not Local. Should Fail local.
  df_fail = pd.DataFrame({"A": ["c"]})
  with pytest.raises(ValueError, match="invalid"):
    process(df_fail)


def test_oneof_contradiction():
  # Global: OneOf 'a', 'b'
  # Local: OneOf 'c', 'd'
  # Intersection Empty -> Local wins (User intent override)
  @validate
  def process(
    df: Validated[
      pd.DataFrame,
      OneOf("a", "b"),
      HasColumn("A", OneOf("c", "d")),
    ],
  ):
    return df

  df = pd.DataFrame({"A": ["c"]})
  process(df)

  # "a" is in Global but explicitly overridden by local. Should fail.
  # Because logic is: If intersection empty -> Local only.
  df_fail = pd.DataFrame({"A": ["a"]})
  with pytest.raises(ValueError, match="invalid"):
    process(df_fail)


# =============================================================================
# 5. Mixed Domain (Range + Set)
# =============================================================================
# Note: Current implementation maps Range and OneOf to same Domain object.
# Implementing interaction between them (e.g. OneOf[1, 5, 10] AND Gt(4) -> OneOf[5, 10])


def test_range_filters_allowed_set():

  @validate
  def process(df: Validated[pd.DataFrame, OneOf(1, 5, 10), HasColumn("A", Gt(4))]):
    return df

  df = pd.DataFrame({"A": [5, 10]})
  process(df)

  # 1 is in Global OneOf but fails Local Gt(4)
  # The domain logic should intersect them: Set & Range -> filtered Set.
  df_fail = pd.DataFrame({"A": [1]})
  with pytest.raises(
    ValueError, match="invalid"
  ):  # OneOf raises "invalid", Gt raises ">"
    # If successfully filtered, the active validator is the FILTERED OneOf.
    # So error message comes from OneOf.
    process(df_fail)
