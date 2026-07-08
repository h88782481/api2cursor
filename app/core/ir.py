"""统一中间表示（IR）

所有协议编解码器共享的内部数据结构。任意"Cursor 格式 × 中转站格式"的组合
都通过 IR 中转：请求方向 `客户端格式 → IRRequest → 上游格式`，响应方向反之，
流式方向则以 Stream* 事件为中间态。

设计原则：
- 用 dataclass 而不是 pydantic 模型，避免对 LLM 报文做强校验导致兼容性下降
- 内容块（Block）覆盖文本 / 图片 / 思考 / 工具调用 / 工具结果 五类语义
- 工具调用参数统一为 JSON 字符串，便于流式增量拼接
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any, Union


def gen_id(prefix: str = '') -> str:
    """生成带可选前缀的短随机 ID，用于请求和工具调用标识。"""
    return f'{prefix}{uuid.uuid4().hex[:24]}'


# ═══════════════════════════════════════════════════════════
#  内容块
# ═══════════════════════════════════════════════════════════


@dataclass
class TextBlock:
    text: str = ''


@dataclass
class ImageBlock:
    media_type: str = 'image/png'
    data: str = ''  # base64 数据（不含 data: 前缀）
    url: str = ''   # 外链图片地址（与 data 互斥）

    def to_data_url(self) -> str:
        """还原成 OpenAI image_url 需要的地址。"""
        if self.data:
            return f'data:{self.media_type};base64,{self.data}'
        return self.url


@dataclass
class ThinkingBlock:
    text: str = ''
    signature: str = ''


@dataclass
class ToolCallBlock:
    id: str = ''
    name: str = ''
    arguments: str = '{}'  # JSON 字符串


@dataclass
class ToolResultBlock:
    call_id: str = ''
    content: str = ''
    name: str = ''  # 工具名（Gemini functionResponse 等协议需要）


Block = Union[TextBlock, ImageBlock, ThinkingBlock, ToolCallBlock, ToolResultBlock]


# ═══════════════════════════════════════════════════════════
#  请求 / 响应
# ═══════════════════════════════════════════════════════════


@dataclass
class IRMessage:
    role: str = 'user'  # user / assistant
    blocks: list[Block] = field(default_factory=list)

    def text(self) -> str:
        """拼接所有文本块，主要用于哈希和摘要。"""
        return '\n'.join(b.text for b in self.blocks if isinstance(b, TextBlock))

    def tool_calls(self) -> list[ToolCallBlock]:
        return [b for b in self.blocks if isinstance(b, ToolCallBlock)]

    def has_thinking(self) -> bool:
        return any(isinstance(b, ThinkingBlock) for b in self.blocks)


@dataclass
class IRTool:
    name: str = ''
    description: str = ''
    parameters: dict[str, Any] = field(default_factory=dict)


@dataclass
class IRRequest:
    model: str = ''
    system: str = ''
    messages: list[IRMessage] = field(default_factory=list)
    tools: list[IRTool] = field(default_factory=list)
    tool_choice: Any = None  # None / 'auto' / 'required' / 'none' / {'name': ...}
    stream: bool = False
    max_tokens: int | None = None
    temperature: float | None = None
    top_p: float | None = None
    stop: list[str] = field(default_factory=list)


@dataclass
class IRUsage:
    input_tokens: int = 0
    output_tokens: int = 0
    cached_tokens: int = 0
    reasoning_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


@dataclass
class IRResponse:
    id: str = ''
    model: str = ''
    blocks: list[Block] = field(default_factory=list)
    finish_reason: str = 'stop'  # stop / length / tool_calls / content_filter
    usage: IRUsage = field(default_factory=IRUsage)

    def text(self) -> str:
        return ''.join(b.text for b in self.blocks if isinstance(b, TextBlock))

    def thinking(self) -> str:
        return ''.join(b.text for b in self.blocks if isinstance(b, ThinkingBlock))

    def tool_calls(self) -> list[ToolCallBlock]:
        return [b for b in self.blocks if isinstance(b, ToolCallBlock)]


# ═══════════════════════════════════════════════════════════
#  流式事件
# ═══════════════════════════════════════════════════════════


@dataclass
class StreamStart:
    id: str = ''
    model: str = ''


@dataclass
class TextDelta:
    text: str = ''


@dataclass
class ThinkingDelta:
    text: str = ''


@dataclass
class ToolCallStart:
    index: int = 0
    id: str = ''
    name: str = ''


@dataclass
class ToolCallDelta:
    index: int = 0
    arguments: str = ''  # 参数增量片段


@dataclass
class ToolCallEnd:
    """工具调用结束事件，携带解码器已知的完整参数（可能为空）。"""

    index: int = 0
    id: str = ''
    name: str = ''
    arguments: str = ''


@dataclass
class StreamEnd:
    finish_reason: str = 'stop'
    usage: IRUsage | None = None


@dataclass
class StreamError:
    message: str = ''
    error_type: str = 'upstream_error'


StreamEvent = Union[
    StreamStart,
    TextDelta,
    ThinkingDelta,
    ToolCallStart,
    ToolCallDelta,
    ToolCallEnd,
    StreamEnd,
    StreamError,
]
