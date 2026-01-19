from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import pandas as pd

type PandasLike = pd.Series | pd.DataFrame | pd.Index


@runtime_checkable
class ValidatorProtocol(Protocol):
  """Protocol for validators used by Numba backend."""

  @property
  def uses_index(self) -> bool: ...

  def build_numba_expr(
    self,
    target: str,
    parts: list[str],
    arr_name: str = "arr",
    is_range_index: bool = False,
  ) -> None: ...

  def build_numba_expr_column_mode(
    self,
    arr_name: str,
    idx_name: str,
    ctx: NumbaContext,
    parts: list[str],
    is_range_index: bool = False,
  ) -> None: ...


@dataclass
class NumbaContext:
  """Context for building Numba expressions with column offsets."""

  n_rows: int = 0
  col_map: dict[str, int] | None = None

  def __post_init__(self) -> None:
    if self.col_map is None:
      self.col_map = {}

  def col_offset(self, col_name: str) -> str:
    """Get offset expression for a column."""
    if self.col_map is None:
      return ""
    idx = self.col_map.get(col_name, 0)
    if idx == 0:
      return ""
    return f" + n_rows * {idx}"
