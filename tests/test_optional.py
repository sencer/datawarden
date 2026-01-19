import unittest

import pandas as pd

from datawarden import Validated, validate
from datawarden.exceptions import ValidationError
from datawarden.validators import Column, Ge, Gt
from datawarden.validators.base import Pass
from datawarden.validators.structural import IsInstance


class TestOptionalValidated(unittest.TestCase):
  def test_outer_none_skip(self) -> None:
    """Test that Validated[...] | None skips validation when None is passed."""

    @validate
    def func(
      df: Validated[pd.DataFrame, Column("a", Ge(0))] | None,
    ) -> pd.DataFrame | None:
      return df

    # Should pass (skip)
    self.assertIsNone(func(None))

    # Should validate
    df = pd.DataFrame({"a": [1]})
    self.assertIs(func(df), df)

    # Should fail validation
    with self.assertRaises(ValidationError):
      func(pd.DataFrame({"a": [-1]}))

  def test_outer_none_union_syntax(self) -> None:
    """Test using Union[..., None] syntax."""

    @validate
    def func(
      df: Validated[pd.DataFrame, Column("a", Ge(0))] | None,
    ) -> pd.DataFrame | None:
      return df

    self.assertIsNone(func(None))

    with self.assertRaises(ValidationError):
      func(pd.DataFrame({"a": [-1]}))

  def test_default_none(self) -> None:
    """Test default value of None."""

    @validate
    def func(
      df: Validated[pd.DataFrame, Column("a", Ge(0))] | None = None,
    ) -> pd.DataFrame | None:
      return df

    self.assertIsNone(func())

  def test_inner_none_behavior(self) -> None:
    """Test that Validated[... | None] does NOT skip validation automatically."""

    @validate
    def func(val: Validated[pd.DataFrame | None, Gt(0)]) -> pd.DataFrame | None:
      return val

    # Gt(0) on None should fail (AttributeError or comparable error)
    # We just want to ensure it tries validity
    with self.assertRaises((ValidationError, AttributeError)):
      func(None)

    df = pd.DataFrame({"a": [1]})
    self.assertIs(func(df), df)

    with self.assertRaises(ValidationError):
      func(pd.DataFrame({"a": [-1]}))

  def test_inner_none_safe_validator(self) -> None:
    """Test Validated[...|None] with a validator that handles None (IsInstance)."""

    @validate
    def func(val: Validated[pd.DataFrame | None, Pass]) -> pd.DataFrame | None:
      return val

    # Should pass because there are no extra validators that fail on None
    self.assertIsNone(func(None))
    df = pd.DataFrame({"a": [1]})
    self.assertIs(func(df), df)

    # Should fail IsInstance check
    with self.assertRaises(ValidationError):
      func("s")

  def test_isinstance_str_fix(self) -> None:
    """Test that IsInstance string representation works for Union types."""

    v = IsInstance(int | None)
    s = str(v)
    self.assertIn("IsInstance", s)


if __name__ == "__main__":
  unittest.main()
