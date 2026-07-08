"""请求体格式检测

处理 Cursor BYOK 的已知 bug：请求发到 /v1/chat/completions，
但请求体实际是 OpenAI Responses API 格式（无 messages，有 input/instructions 等）。
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

# Responses API 独有的顶层字段标记（text 为 Responses 的 {verbosity/format} 配置，CC 没有该顶层字段）
_RESPONSES_MARKERS = (
    'previous_response_id',
    'instructions',
    'parallel_tool_calls',
    'reasoning',
    'truncation',
    'max_output_tokens',
    'text',
    'include',
)


def sanitize_responses_payload(payload: dict) -> None:
    """就地清理 Responses 风格请求体中的 Chat Completions 专属字段。

    Cursor 的 BYOK bug 会在 Responses 风格 body 里混入 CC 独有的
    `stream_options: {include_usage: true}`，原样转发会被 /v1/responses 上游拒绝。
    `metadata` 也常导致部分中转站不兼容（与两个参考桥接项目的处理一致）。
    """
    removed = [key for key in ('stream_options', 'metadata') if payload.pop(key, None) is not None]
    if removed:
        logger.info('已从 Responses 请求体移除不兼容字段: %s', removed)


def looks_like_responses_payload(payload: Any) -> bool:
    """判断请求体是否为 Responses API 风格。

    判定规则（参考 Cursor-BYOK-Bridge，比只看 input 更健壮）：
    - 不是字典或包含 messages → 不是
    - 包含 input → 是
    - 包含 Responses 独有标记字段（instructions/previous_response_id/reasoning 等）→ 是
    """
    if not isinstance(payload, dict):
        return False
    if 'messages' in payload:
        return False
    if 'input' in payload:
        return True
    return any(marker in payload for marker in _RESPONSES_MARKERS)
