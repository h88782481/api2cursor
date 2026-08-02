"""请求级数据与统一错误。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from ..settings import Route

CursorDialect = Literal['function', 'custom_grammar']


@dataclass(slots=True)
class Exchange:
    route: Route
    stream: bool
    custom_tools: set[str]
    log: dict[str, Any] | None = None
