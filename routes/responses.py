"""路由: /v1/responses

处理 Cursor 对 GPT、Claude-Opus 等模型发出的 Responses API 请求。
请求先转换为 Chat Completions 中间表示，再通过统一出站转换器分发。
"""

from __future__ import annotations

import json
import logging
from typing import Any

import settings
from flask import Blueprint, jsonify, request

from adapters.openai_compat_fixer import normalize_request
from adapters.responses_cc_adapter import (
    AnthropicOutboundForResponses,
    ResponsesNativeOutbound,
    responses_to_cc,
)
from adapters.unified import handle_non_stream, handle_stream
from routes.common import (
    ResponsesClientFormatter,
    ResponsesPassthroughFormatter,
    build_route_context,
    get_outbound,
    inject_instructions_cc,
    inject_instructions_responses,
    log_route_context,
    should_inject_thinking,
)
from utils.request_logger import start_turn
from utils.thinking_cache import thinking_cache

logger = logging.getLogger(__name__)

bp = Blueprint('responses', __name__)


@bp.route('/v1/responses', methods=['POST'])
def responses_endpoint():
    """处理 Responses 请求并按模型映射分发。"""
    original_payload = request.get_json(force=True)
    payload = json.loads(json.dumps(original_payload, ensure_ascii=False, default=str))
    client_model = payload.get('model', 'unknown')
    is_stream = payload.get('stream', False)

    ctx = build_route_context(client_model, is_stream)
    turn = start_turn(
        route='responses',
        client_model=client_model,
        backend=ctx.backend,
        stream=is_stream,
        client_request=original_payload,
        request_headers=dict(request.headers),
        target_url=ctx.target_url,
        upstream_model=ctx.upstream_model,
    )
    log_route_context('响应生成', ctx)

    if ctx.backend == 'responses':
        return _handle_native_responses(ctx, payload, turn)

    cc_payload = _build_cc_payload(payload, ctx)

    if ctx.backend == 'anthropic':
        outbound = AnthropicOutboundForResponses()
    else:
        outbound = get_outbound(ctx.backend)

    client_fmt = ResponsesClientFormatter(model=ctx.client_model)

    if ctx.is_stream:
        return handle_stream(ctx, outbound, client_fmt, cc_payload, turn)
    return handle_non_stream(ctx, outbound, client_fmt, cc_payload, turn)


def _handle_native_responses(ctx, payload: dict[str, Any], turn: dict[str, Any]):
    """处理走原生 Responses 后端的请求（直接透传）。"""
    payload = dict(payload)
    payload['model'] = ctx.upstream_model
    payload = inject_instructions_responses(payload, ctx.custom_instructions, ctx.instructions_position)

    outbound = ResponsesNativeOutbound()
    client_fmt = ResponsesPassthroughFormatter(model=ctx.client_model)

    if ctx.is_stream:
        return handle_stream(ctx, outbound, client_fmt, payload, turn)
    return handle_non_stream(ctx, outbound, client_fmt, payload, turn)


def _build_cc_payload(payload: dict[str, Any], ctx) -> dict[str, Any]:
    """将 Responses 请求统一降级为 Chat Completions 中间表示。"""
    cc_payload = responses_to_cc(payload)
    cc_payload['model'] = ctx.upstream_model
    cc_payload = normalize_request(cc_payload)
    if should_inject_thinking(ctx.backend):
        cc_payload['messages'] = thinking_cache.inject(cc_payload.get('messages', []))
    cc_payload = inject_instructions_cc(cc_payload, ctx.custom_instructions, ctx.instructions_position)
    return cc_payload
