"""网关支持的协议类型。"""

from typing import Literal, TypeAlias

WireProtocol: TypeAlias = Literal['chat', 'responses', 'messages', 'gemini']
ConfiguredProtocol: TypeAlias = Literal['auto'] | WireProtocol
