from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
  from .common import PandasLike


@dataclass(slots=True)
class ValidationContext:
  root_data: "PandasLike"
  extra: dict[str, object] = field(default_factory=dict)
