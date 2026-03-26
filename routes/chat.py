"""路由: /v1/chat/completions

处理 Cursor 发来的 OpenAI Chat Completions 格式请求。
根据模型映射的后端类型，通过统一的出站转换器转发到不同后端。
"""

from __future__ import annotations

import json
import logging
from typing import Any

from flask import Blueprint, jsonify, request

from adapters.openai_compat_fixer import normalize_request
from adapters.responses_cc_adapter import responses_to_cc
from adapters.unified import handle_non_stream, handle_stream
from routes.common import (
    CCClientFormatter,
    build_route_context,
    get_outbound,
    inject_instructions_cc,
    log_route_context,
    should_inject_thinking,
)
from utils.request_logger import start_turn
from utils.thinking_cache import thinking_cache

logger = logging.getLogger(__name__)

bp = Blueprint('chat', __name__)


@bp.route('/v1/chat/completions', methods=['POST'])
def chat_completions():
    """处理聊天补全请求并按模型映射分发到不同后端。"""
    original_payload = request.get_json(force=True)
    payload, message_count = _normalize_chat_payload(
        json.loads(json.dumps(original_payload, ensure_ascii=False, default=str))
    )

    client_model = payload.get('model', 'unknown')
    is_stream = payload.get('stream', False)
    ctx = build_route_context(client_model, is_stream)
    turn = start_turn(
        route='chat',
        client_model=client_model,
        backend=ctx.backend,
        stream=is_stream,
        client_request=original_payload,
        request_headers=dict(request.headers),
        target_url=ctx.target_url,
        upstream_model=ctx.upstream_model,
        metadata={'message_count': message_count},
    )

    log_route_context('聊天补全', ctx, extra=f'消息数={message_count}')
    _log_messages(payload)

    payload['model'] = ctx.upstream_model
    payload = normalize_request(payload)
    if should_inject_thinking(ctx.backend):
        payload['messages'] = thinking_cache.inject(payload.get('messages', []))
    payload = inject_instructions_cc(payload, ctx.custom_instructions, ctx.instructions_position)

    outbound = get_outbound(ctx.backend)
    client_fmt = CCClientFormatter()

    if ctx.is_stream:
        result = handle_stream(ctx, outbound, client_fmt, payload, turn)
    else:
        result = handle_non_stream(ctx, outbound, client_fmt, payload, turn)

    if not ctx.is_stream and isinstance(result, tuple):
        response_data = result
    elif hasattr(result, 'json'):
        try:
            response_data = result.get_json(silent=True) or {}
        except Exception:
            response_data = {}
    else:
        response_data = {}

    _try_cache_thinking(response_data)
    return result


def _normalize_chat_payload(payload: dict[str, Any]) -> tuple[dict[str, Any], int]:
    """整理聊天补全入口的请求体。

    当 Cursor 或调用方把 Responses 格式误发到 `/v1/chat/completions` 时，
    先降级转换成 Chat Completions，再进入统一主流程。
    """
    message_count = len(payload.get('messages', []))

    if message_count == 0 and 'input' in payload:
        logger.info('检测到 Responses 格式误入聊天补全接口，已自动转换为 Chat Completions 格式')
        payload = responses_to_cc(payload)
        message_count = len(payload.get('messages', []))
    elif message_count == 0:
        logger.warning('消息列表为空，请求字段=%s', list(payload.keys()))

    return payload, message_count


def _try_cache_thinking(response_data: dict[str, Any]) -> None:
    """尝试从非流式响应中缓存思维链内容。"""
    if not isinstance(response_data, dict):
        return
    for choice in response_data.get('choices', []):
        msg = choice.get('message', {})
        if msg.get('reasoning_content'):
            thinking_cache.store_from_response(
                request.get_json(silent=True, force=True).get('messages', []),
                msg['reasoning_content'],
            )
            break


def _log_messages(payload: dict[str, Any]) -> None:
    """记录消息摘要，方便排查请求形态是否符合预期。"""
    for index, message in enumerate(payload.get('messages', [])):
        role = message.get('role', '?')
        content = message.get('content')
        extra = ''

        if 'tool_calls' in message:
            extra += f' 工具调用数={len(message["tool_calls"])}'
        if message.get('tool_call_id'):
            extra += f' 工具调用ID={message["tool_call_id"]}'

        if isinstance(content, list):
            content_info = f'列表[{len(content)}]'
        elif isinstance(content, str):
            content_info = f'文本[{len(content)}]'
        else:
            content_info = type(content).__name__

        logger.info('  消息[%s] 角色=%s 内容=%s%s', index, role, content_info, extra)
