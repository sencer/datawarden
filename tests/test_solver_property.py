from __future__ import annotations

from hypothesis import given, strategies as st
import pytest

from datawarden.exceptions import LogicError
from datawarden.solver import resolve_domains
from datawarden.validators import Ge, Gt, Le, Lt


@given(
  low=st.integers(min_value=-100, max_value=100),
  high=st.integers(min_value=-100, max_value=100),
)
def test_resolve_domains_range_intersection(low, high):
  """Property: resolve_domains should correctly merge Ge and Le or raise LogicError if contradictory in same scope."""
  v_low = Ge(low)
  v_high = Le(high)

  try:
    # In same scope (both local), contradiction raises LogicError
    resolved = resolve_domains([], [v_low, v_high])
    if low > high:
      pytest.fail(
        f"Contradiction Ge({low}), Le({high}) in same scope did not raise LogicError"
      )
    # Should have Ge, Le and NonNaN (default domain behavior)
    assert len(resolved) >= 2
  except LogicError:
    assert low > high


@st.composite
def range_constraint(draw):
  op = draw(st.sampled_from([Ge, Le, Gt, Lt]))
  val = draw(st.integers(min_value=-100, max_value=100))
  return op(val)


@given(c1=range_constraint(), c2=range_constraint())
def test_resolve_domains_arbitrary_merge(c1, c2):
  """Property: resolve_domains should merge any two range constraints or raise LogicError if contradictory in same scope."""
  try:
    resolved = resolve_domains([], [c1, c2])
    # Max resolved: Ge, Le, NonNaN = 3
    assert len(resolved) <= 3
  except LogicError:
    # Valid outcome for contradictions in same scope
    return


@given(g=range_constraint(), local_c=range_constraint())
def test_resolve_domains_override_behavior(g, local_c):
  """Property: Local constraints should override global ones if they contradict."""
  try:
    resolved = resolve_domains([g], [local_c])
    # Should never raise LogicError when merging Global and Local because
    # resolve_domains implements "Local overrides Global if contradiction"
    assert len(resolved) >= 1
  except LogicError:
    pytest.fail(
      f"Global {g} and Local {local_c} caused LogicError, but Local should override Global."
    )
