from typing import Any

import pytest

from datawarden.base import CompoundValidator, Validator
from datawarden.domain import ValidationDomain


class TestValidationDomain:
  def test_is_empty(self):
    d = ValidationDomain()
    assert not d.is_empty()

    # Min > Max
    d.min_val = 10
    d.max_val = 5
    assert d.is_empty()

    # Min equals Max, inclusive
    d.min_val = 5
    d.max_val = 5
    d.min_inclusive = True
    d.max_inclusive = True
    assert not d.is_empty()

    # Min equals Max, exclusive
    d.max_inclusive = False
    assert d.is_empty()

    d.max_inclusive = True
    d.min_inclusive = False
    assert d.is_empty()

    # Empty set
    d = ValidationDomain(allowed_values=set())
    assert d.is_empty()

  def test_intersect_sets(self):
    d1 = ValidationDomain(allowed_values={1, 2, 3})
    d2 = ValidationDomain(allowed_values={2, 3, 4})

    res = d1.intersect(d2)
    assert res.allowed_values == {2, 3}

    d3 = ValidationDomain()  # No set constraint
    res = d1.intersect(d3)
    assert res.allowed_values == {1, 2, 3}

    res = d3.intersect(d1)
    assert res.allowed_values == {1, 2, 3}

  def test_intersect_ranges(self):
    # [0, 10] intersect [5, 15] -> [5, 10]
    d1 = ValidationDomain(min_val=0, max_val=10)
    d2 = ValidationDomain(min_val=5, max_val=15)
    res = d1.intersect(d2)
    assert res.min_val == 5
    assert res.max_val == 10

    # (0, 10) intersect [0, 10] -> (0, 10]
    d1 = ValidationDomain(min_val=0, max_val=10, min_inclusive=False)
    d2 = ValidationDomain(min_val=0, max_val=10, min_inclusive=True)
    res = d1.intersect(d2)
    assert res.min_val == 0
    assert res.min_inclusive is False

    # Same logic for max
    d1 = ValidationDomain(max_val=10, max_inclusive=False)
    d2 = ValidationDomain(max_val=10, max_inclusive=True)
    res = d1.intersect(d2)
    assert res.max_val == 10
    assert res.max_inclusive is False

  def test_intersect_range_and_set(self):
    # Set {1, 5, 10}, Range [4, 6] -> {5}
    d1 = ValidationDomain(allowed_values={1, 5, 10})
    d2 = ValidationDomain(min_val=4, max_val=6)
    res = d1.intersect(d2)
    assert res.allowed_values == {5}

    # Strict inequality checks
    d1 = ValidationDomain(allowed_values={4, 5, 6})
    d2 = ValidationDomain(
      min_val=4, max_val=6, min_inclusive=False, max_inclusive=False
    )
    res = d1.intersect(d2)
    assert res.allowed_values == {5}  # 4 and 6 excluded

  def test_is_subset(self):
    # Range subset
    d1 = ValidationDomain(min_val=5, max_val=10)
    d2 = ValidationDomain(min_val=0, max_val=20)
    assert d1.is_subset(d2)  # [5,10] is subset of [0,20]
    assert not d2.is_subset(d1)

    # Exclusive boundaries
    d1 = ValidationDomain(min_val=0, min_inclusive=True)  # [0, inf)
    d2 = ValidationDomain(min_val=0, min_inclusive=False)  # (0, inf)
    assert not d1.is_subset(d2)  # 0 is in d1 but not d2
    assert d2.is_subset(d1)

    # Flags
    d1 = ValidationDomain(allows_nan=True)
    d2 = ValidationDomain(allows_nan=False)
    assert not d1.is_subset(d2)

    # Sets
    d1 = ValidationDomain(allowed_values={1})
    d2 = ValidationDomain(allowed_values={1, 2})
    assert d1.is_subset(d2)
    assert not d2.is_subset(d1)

    # Set vs No Set
    d1 = ValidationDomain(allowed_values={1})
    d2 = ValidationDomain()
    assert d1.is_subset(d2)  # {1} is subset of Any
    assert not d2.is_subset(d1)  # Any is not subset of {1}

  def test_compound_validator_fallback(self):
    class FlakyVectorized(Validator[Any]):
      priority = 10

      def validate_vectorized(self, data):
        # Claim to support it but fail
        raise NotImplementedError("Oops")

      def validate(self, data):
        # Fallback should call this
        if data == "fail":
          raise ValueError("Failed in fallback")

    # But we can instantiate CompoundValidator directly.
    v = CompoundValidator([FlakyVectorized()])

    # Should fall back to validate()
    v.validate("ok")  # Should pass

    with pytest.raises(ValueError, match="Failed in fallback"):
      v.validate("fail")
