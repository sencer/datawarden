import pandas as pd
import pytest

from datawarden import Overrides, Validated, validate
from datawarden.exceptions import ValidationError
from datawarden.validators import And, Column, MonoUp, Or


def test_base_clone_creates_copy() -> None:
  """Test that clone() creates a new instance."""
  m1 = MonoUp()
  m2 = m1.clone()
  assert m1 is not m2
  assert isinstance(m2, type(m1))


def test_monoup_state_key_isolation() -> None:
  """Test that cloned MonoUp instances have different state keys."""
  m1 = MonoUp().clone()
  m2 = MonoUp().clone()

  assert m1 is not m2
  assert m1._state_key != m2._state_key
  assert f"{id(m1)}" in m1._state_key
  assert f"{id(m2)}" in m2._state_key


def test_column_clones_validator() -> None:
  """Test that Column clones the passed validator."""
  original = MonoUp()
  col = Column("a", original)

  # Column stores the validator in self.validator
  assert col.validator is not original
  assert isinstance(col.validator, type(original))
  # Check state key difference
  assert col.validator._state_key != original._state_key


def test_and_clones_children() -> None:
  """Test that And clones its children."""
  original = MonoUp()
  combined = And(original)

  assert combined.validators[0] is not original
  assert combined.validators[0]._state_key != original._state_key


def test_or_clones_children() -> None:
  """Test that Or clones its children."""
  original = MonoUp()
  combined = Or(original)

  assert combined.validators[0] is not original


def test_nested_cloning() -> None:
  """Test deep cloning in nested structures."""
  original = MonoUp()
  # Column -> And -> MonoUp
  col = Column("a", And(original))

  # Access the inner MonoUp
  inner = col.validator.validators[0]

  assert inner is not original
  assert inner._state_key != original._state_key


def test_state_isolation_in_validation() -> None:
  """Verify that two columns using the same singleton do not interfere."""
  # We use a single call with chunking enabled.
  # Column A: [1, 2, 3, 4] -> Monotonic across chunks [1, 2] and [3, 4]
  # Column B: [1, 2, 1, 2] -> Monotonic within chunks [1, 2], [1, 2], but NOT across (2 -> 1)

  @validate
  def process(
    _df: Validated[pd.DataFrame, Column("a", MonoUp), Column("b", MonoUp)],
  ) -> bool:
    return True

  df = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0], "b": [1.0, 2.0, 1.1, 2.1]})

  # With chunk_size=2, it should fail on column B at the boundary of second chunk.
  with Overrides(chunk_size_rows=2):
    with pytest.raises(ValidationError) as exc:
      process(df)

    assert "Column 'b' failed" in str(exc.value)
    assert "Monotonicity broken between chunks" in str(exc.value)
    # Ensure A didn't fail (it is monotonic)
    assert "Column 'a' failed" not in str(exc.value)


def test_recursive_clone_and() -> None:
  """Test that And.clone() recursively clones children."""
  a1 = And(MonoUp())
  a2 = a1.clone()

  # a2 should be a new And
  assert a1 is not a2
  # a2's children should be new clones, NOT the same objects as a1's children
  assert a1.validators[0] is not a2.validators[0]
  assert a1.validators[0]._state_key != a2.validators[0]._state_key
