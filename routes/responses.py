"""路由: /v1/responses

处理 Cursor 对 GPT、Claude-Opus 等模型发出的 Responses API 请求。
请求会先转换为 Chat Completions 中间表示，再按后端类型分发，最后转换回 Responses 格式。
"""

from __future__ import annotations

import json
import logging
from typing import Any

from flask import Blueprint, jsonify, request

from adapters.cc_anthropic_adapter import cc_to_messages_request, messages_to_cc_response
from adapters.cc_gemini_adapter import GeminiStreamConverter, cc_to_gemini_request, gemini_to_cc_response
from adapters.openai_compat_fixer import fix_response, fix_stream_chunk, normalize_request
from adapters.responses_cc_adapter import ResponsesStreamConverter, cc_to_responses, responses_to_cc
from config import Config
from routes.common import (
    RouteContext,
    apply_body_modifications,
    apply_header_modifications,
    build_anthropic_target,
    build_gemini_target,
    build_openai_target,
    build_responses_target,
    build_route_context,
    inject_instructions_anthropic,
    inject_instructions_cc,
    inject_instructions_responses,
    log_route_context,
    log_usage,
    responses_error_event,
)
from utils.http import (
    forward_request,
    gen_id,
    iter_anthropic_sse,
    iter_gemini_sse,
    iter_openai_sse,
    iter_responses_sse,
    sse_response,
)
from utils.request_logger import (
    append_client_event,
    append_upstream_event,
    attach_client_response,
    attach_error,
    attach_upstream_request,
    attach_upstream_response,
    finalize_turn,
    set_stream_summary,
    start_turn,
)
from utils.think_tag import ThinkTagExtractor
from utils.thinking_cache import thinking_cache
from utils.usage_tracker import usage_tracker

logger = logging.getLogger(__name__)

bp = Blueprint('responses', __name__)


def _dbg(message: str) -> None:
    """仅在调试模式下输出详细日志。"""
    if Config.DEBUG:
        logger.info('[响应生成调试] %s', message)


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

    cc_payload = _build_cc_payload(payload, ctx)

    if ctx.backend == 'openai':
        return _handle_openai_backend(ctx, cc_payload, turn)
    if ctx.backend == 'responses':
        return _handle_responses_backend(ctx, payload, turn)
    if ctx.backend == 'gemini':
        return _handle_gemini_backend(ctx, cc_payload, turn)
    return _handle_anthropic_backend(ctx, cc_payload, turn)


def _build_cc_payload(payload: dict[str, Any], ctx: RouteContext) -> dict[str, Any]:
    """将 Responses 请求统一降级为 Chat Completions 中间表示。

    这样后续无论走 OpenAI 兼容后端还是 Anthropic 后端，都能复用一套
    中间协议，避免在路由层同时维护两套完全不同的请求编排逻辑。
    """
    cc_payload = responses_to_cc(payload)
    cc_payload['model'] = ctx.upstream_model
    cc_payload['messages'] = thinking_cache.inject(cc_payload.get('messages', []))
    cc_payload = inject_instructions_cc(cc_payload, ctx.custom_instructions, ctx.instructions_position)
    _dbg(
        '已转换为聊天补全中间表示：字段=' + str(list(cc_payload.keys()))
        + f' 消息数={len(cc_payload.get("messages", []))}'
    )
    return cc_payload


def _handle_openai_backend(ctx: RouteContext, cc_payload: dict[str, Any], turn: dict[str, Any]):
    """处理走 OpenAI 兼容后端的 Responses 请求。"""
    cc_payload = normalize_request(cc_payload)
    _dbg(
        f'标准化完成：模型={cc_payload.get("model")} '
        f'工具数={len(cc_payload.get("tools", []))}'
    )

    url, headers = build_openai_target(ctx)
    cc_payload = apply_body_modifications(cc_payload, ctx.body_modifications)
    headers = apply_header_modifications(headers, ctx.header_modifications)

    if ctx.is_stream:
        return _handle_openai_stream(ctx, cc_payload, url, headers, turn)
    return _handle_openai_non_stream(ctx, cc_payload, url, headers, turn)


def _handle_openai_non_stream(
    ctx: RouteContext,
    cc_payload: dict[str, Any],
    url: str,
    headers: dict[str, str],
    turn: dict[str, Any],
):
    """处理 OpenAI 兼容后端的非流式 Responses 返回。"""
    cc_payload['stream'] = False
    attach_upstream_request(turn, cc_payload, headers)
    resp, err = forward_request(url, headers, cc_payload)
    if err:
        attach_error(turn, {'stage': 'forward_request', 'message': 'upstream request failed'})
        finalize_turn(turn)
        return err

    raw = resp.json()
    attach_upstream_response(turn, raw)
    _dbg('上游原始响应=' + json.dumps(raw, ensure_ascii=False, default=str)[:1000])

    fixed = fix_response(raw)
    response_data = cc_to_responses(fixed, ctx.client_model)
    return _finalize_responses_response(response_data, turn=turn, debug_label='转换为 Responses 后')


def _handle_openai_stream(
    ctx: RouteContext,
    cc_payload: dict[str, Any],
    url: str,
    headers: dict[str, str],
    turn: dict[str, Any] | None,
):
    """处理 OpenAI 兼容后端的流式 Responses 返回。"""
    cc_payload['stream'] = True
    converter = ResponsesStreamConverter(model=ctx.client_model)

    def generate():
        """消费 OpenAI 聊天补全流，并实时改写为 Responses SSE。"""
        yield from converter.start_events()

        attach_upstream_request(turn, cc_payload, headers)
        resp, err = forward_request(url, headers, cc_payload, stream=True)
        if err:
            attach_error(turn, {'stage': 'forward_request', 'message': str(err)})
            set_stream_summary(turn, {'status': 'error'})
            finalize_turn(turn)
            yield responses_error_event(str(err))
            return

        think_extractor = ThinkTagExtractor()
        chunk_count = 0
        client_events: list[str] = []

        for chunk in iter_openai_sse(resp):
            if chunk is None:
                _dbg(f'流式响应结束，共 {chunk_count} 个数据片段')
                finalized_events = converter.finalize()
                for item in finalized_events:
                    client_events.append(item)
                    append_client_event(turn, {'type': 'responses_event', 'data': item})
                    yield item
                usage_tracker.record(ctx.client_model)
                set_stream_summary(turn, {
                    'chunk_count': chunk_count,
                    'client_event_count': len(client_events),
                })
                attach_client_response(turn, {
                    'type': 'responses.stream.summary',
                    'model': ctx.client_model,
                    'events': client_events,
                })
                finalize_turn(turn)
                return

            append_upstream_event(turn, {'type': 'openai_chunk', 'data': chunk})
            if chunk_count < 10:
                _dbg(
                    f'上游原始片段#{chunk_count}='
                    + json.dumps(chunk, ensure_ascii=False, default=str)[:500]
                )

            chunk = fix_stream_chunk(chunk)
            for out in think_extractor.process_chunk(chunk):
                for evt in converter.process_cc_chunk(out):
                    client_events.append(evt)
                    append_client_event(turn, {'type': 'responses_event', 'data': evt})
                    if chunk_count < 10:
                        _dbg(
                            f'转换后片段#{chunk_count}='
                            + json.dumps(out, ensure_ascii=False, default=str)[:500]
                        )
                    yield evt

            chunk_count += 1

    return sse_response(generate())


def _handle_responses_backend(ctx: RouteContext, payload: dict[str, Any], turn: dict[str, Any] | None):
    """处理走原生 Responses 后端的请求。

    当中转站本身就只支持 `/v1/responses` 时，不需要再绕到聊天补全中间协议，
    直接转发原生 Responses 请求即可。
    """
    payload = dict(payload)
    payload['model'] = ctx.upstream_model
    payload = inject_instructions_responses(payload, ctx.custom_instructions, ctx.instructions_position)
    url, headers = build_responses_target(ctx)
    payload = apply_body_modifications(payload, ctx.body_modifications)
    headers = apply_header_modifications(headers, ctx.header_modifications)

    if ctx.is_stream:
        return _handle_responses_stream(ctx, payload, url, headers, turn)
    return _handle_responses_non_stream(ctx, payload, url, headers, turn)


def _handle_responses_non_stream(
    ctx: RouteContext,
    payload: dict[str, Any],
    url: str,
    headers: dict[str, str],
    turn: dict[str, Any] | None,
):
    """处理原生 Responses 后端的非流式返回。"""
    payload['stream'] = False
    attach_upstream_request(turn, payload, headers)
    resp, err = forward_request(url, headers, payload)
    if err:
        attach_error(turn, {'stage': 'forward_request', 'message': 'upstream request failed'})
        finalize_turn(turn)
        return err

    response_data = resp.json()
    attach_upstream_response(turn, response_data)
    response_data['model'] = ctx.client_model
    return _finalize_responses_response(response_data, turn=turn, debug_label='原生 Responses 返回后')


def _handle_responses_stream(
    ctx: RouteContext,
    payload: dict[str, Any],
    url: str,
    headers: dict[str, str],
    turn: dict[str, Any] | None,
):
    """处理原生 Responses 后端的流式返回。"""
    payload['stream'] = True
    converter = ResponsesStreamConverter(model=ctx.client_model)

    def generate():
        """透传上游原生 Responses 流，并做轻量模型名改写。"""
        attach_upstream_request(turn, payload, headers)
        resp, err = forward_request(url, headers, payload, stream=True)
        if err:
            attach_error(turn, {'stage': 'forward_request', 'message': str(err)})
            set_stream_summary(turn, {'status': 'error'})
            finalize_turn(turn)
            yield responses_error_event(str(err))
            return

        event_count = 0
        client_events: list[str] = []
        for event_type, event_data in iter_responses_sse(resp):
            append_upstream_event(turn, {'type': event_type, 'data': event_data})
            if event_count < 10:
                _dbg(
                    f'上游事件#{event_count} 类型={event_type} 数据='
                    + json.dumps(event_data, ensure_ascii=False, default=str)[:500]
                )
            produced = converter.process_responses_event(event_type, event_data)
            for evt in produced:
                client_events.append(evt)
                append_client_event(turn, {'type': 'responses_event', 'data': evt})
                yield evt
            event_count += 1

        _dbg(f'流式响应结束，共 {event_count} 个事件')
        usage_tracker.record(ctx.client_model)
        set_stream_summary(turn, {
            'event_count': event_count,
            'client_event_count': len(client_events),
        })
        attach_client_response(turn, {
            'type': 'responses.stream.summary',
            'model': ctx.client_model,
            'events': client_events,
        })
        finalize_turn(turn)

    return sse_response(generate())


def _handle_gemini_backend(ctx: RouteContext, cc_payload: dict[str, Any], turn: dict[str, Any] | None):
    """处理走 Gemini Contents 后端的 Responses 请求。"""
    gemini_payload = cc_to_gemini_request(cc_payload)
    _dbg(
        '已转换为 Gemini 请求：字段=' + str(list(gemini_payload.keys()))
        + f' 内容数={len(gemini_payload.get("contents", []))}'
    )

    url, headers = build_gemini_target(ctx, stream=ctx.is_stream)
    gemini_payload = apply_body_modifications(gemini_payload, ctx.body_modifications)
    headers = apply_header_modifications(headers, ctx.header_modifications)

    if ctx.is_stream:
        return _handle_gemini_stream(ctx, gemini_payload, url, headers, turn)
    return _handle_gemini_non_stream(ctx, gemini_payload, url, headers, turn)


def _handle_gemini_non_stream(
    ctx: RouteContext,
    payload: dict[str, Any],
    url: str,
    headers: dict[str, str],
    turn: dict[str, Any] | None,
):
    """处理 Gemini 后端的非流式 Responses 返回。"""
    attach_upstream_request(turn, payload, headers)
    resp, err = forward_request(url, headers, payload)
    if err:
        attach_error(turn, {'stage': 'forward_request', 'message': 'upstream request failed'})
        finalize_turn(turn)
        return err

    raw = resp.json()
    attach_upstream_response(turn, raw)
    _dbg('上游原始响应=' + json.dumps(raw, ensure_ascii=False, default=str)[:1000])

    cc_data = gemini_to_cc_response(raw)
    response_data = cc_to_responses(cc_data, ctx.client_model)
    return _finalize_responses_response(response_data, turn=turn, debug_label='Gemini 转回 Responses 后')


def _handle_gemini_stream(
    ctx: RouteContext,
    payload: dict[str, Any],
    url: str,
    headers: dict[str, str],
    turn: dict[str, Any] | None,
):
    """处理 Gemini 后端的流式 Responses 返回。"""
    converter = ResponsesStreamConverter(model=ctx.client_model)
    gemini_converter = GeminiStreamConverter()

    def generate():
        yield from converter.start_events()

        attach_upstream_request(turn, payload, headers)
        resp, err = forward_request(url, headers, payload, stream=True)
        if err:
            attach_error(turn, {'stage': 'forward_request', 'message': str(err)})
            set_stream_summary(turn, {'status': 'error'})
            finalize_turn(turn)
            yield responses_error_event(str(err))
            return

        chunk_count = 0
        client_events: list[str] = []
        for gemini_chunk in iter_gemini_sse(resp):
            append_upstream_event(turn, {'type': 'gemini_chunk', 'data': gemini_chunk})
            if chunk_count < 10:
                _dbg(
                    f'上游 Gemini 片段#{chunk_count}='
                    + json.dumps(gemini_chunk, ensure_ascii=False, default=str)[:500]
                )

            for cc_chunk in gemini_converter.process_chunk(gemini_chunk):
                for evt in converter.process_cc_chunk(cc_chunk):
                    client_events.append(evt)
                    append_client_event(turn, {'type': 'responses_event', 'data': evt})
                    yield evt

            chunk_count += 1

        _dbg(f'流式响应结束，共 {chunk_count} 个数据片段')
        finalized_events = converter.finalize()
        for evt in finalized_events:
            client_events.append(evt)
            append_client_event(turn, {'type': 'responses_event', 'data': evt})
            yield evt
        usage_tracker.record(ctx.client_model)
        set_stream_summary(turn, {
            'chunk_count': chunk_count,
            'client_event_count': len(client_events),
        })
        attach_client_response(turn, {
            'type': 'responses.stream.summary',
            'model': ctx.client_model,
            'events': client_events,
        })
        finalize_turn(turn)

    return sse_response(generate())


def _handle_anthropic_backend(ctx: RouteContext, cc_payload: dict[str, Any], turn: dict[str, Any] | None):
    """处理走 Anthropic 后端的 Responses 请求。"""
    anthropic_payload = cc_to_messages_request(cc_payload)
    _dbg(
        '已转换为 Messages 请求：字段=' + str(list(anthropic_payload.keys()))
        + f' 消息数={len(anthropic_payload.get("messages", []))}'
    )

    url, headers = build_anthropic_target(ctx)
    anthropic_payload = apply_body_modifications(anthropic_payload, ctx.body_modifications)
    headers = apply_header_modifications(headers, ctx.header_modifications)

    if ctx.is_stream:
        return _handle_anthropic_stream(ctx, anthropic_payload, url, headers, turn)
    return _handle_anthropic_non_stream(ctx, anthropic_payload, url, headers, turn)


def _handle_anthropic_non_stream(
    ctx: RouteContext,
    anthropic_payload: dict[str, Any],
    url: str,
    headers: dict[str, str],
    turn: dict[str, Any] | None,
):
    """处理 Anthropic 后端的非流式 Responses 返回。"""
    anthropic_payload['stream'] = False
    attach_upstream_request(turn, anthropic_payload, headers)
    resp, err = forward_request(url, headers, anthropic_payload)
    if err:
        attach_error(turn, {'stage': 'forward_request', 'message': 'upstream request failed'})
        finalize_turn(turn)
        return err

    raw = resp.json()
    attach_upstream_response(turn, raw)
    _dbg('上游原始响应=' + json.dumps(raw, ensure_ascii=False, default=str)[:1000])

    cc_data = messages_to_cc_response(raw)
    response_data = cc_to_responses(cc_data, ctx.client_model)
    return _finalize_responses_response(response_data, turn=turn, debug_label='Messages 转回 Responses 后')


def _handle_anthropic_stream(
    ctx: RouteContext,
    anthropic_payload: dict[str, Any],
    url: str,
    headers: dict[str, str],
    turn: dict[str, Any] | None,
):
    """处理 Anthropic 后端的流式 Responses 返回。

    这里直接将 Anthropic SSE 事件映射到 Responses SSE，故意跳过 CC 流式中间态，
    这样可以减少一次事件重组，降低流式转换复杂度，也更容易保留原始时序。
    """
    anthropic_payload['stream'] = True
    converter = ResponsesStreamConverter(model=ctx.client_model)

    def generate():
        """消费 Anthropic SSE，并直接映射为 Responses 事件序列。"""
        yield from converter.start_events()

        attach_upstream_request(turn, anthropic_payload, headers)
        resp, err = forward_request(url, headers, anthropic_payload, stream=True)
        if err:
            attach_error(turn, {'stage': 'forward_request', 'message': str(err)})
            set_stream_summary(turn, {'status': 'error'})
            finalize_turn(turn)
            yield responses_error_event(str(err))
            return

        event_count = 0
        client_events: list[str] = []
        for event_type, event_data in iter_anthropic_sse(resp):
            append_upstream_event(turn, {'type': event_type, 'data': event_data})
            if event_count < 10:
                _dbg(
                    f'上游事件#{event_count} 类型={event_type} 数据='
                    + json.dumps(event_data, ensure_ascii=False, default=str)[:500]
                )

            produced = converter.process_anthropic_event(event_type, event_data)
            for evt in produced:
                client_events.append(evt)
                append_client_event(turn, {'type': 'responses_event', 'data': evt})
                yield evt
            event_count += 1

        _dbg(f'流式响应结束，共 {event_count} 个事件')
        finalized_events = converter.finalize()
        for evt in finalized_events:
            client_events.append(evt)
            append_client_event(turn, {'type': 'responses_event', 'data': evt})
            yield evt
        usage_tracker.record(ctx.client_model)
        set_stream_summary(turn, {
            'event_count': event_count,
            'client_event_count': len(client_events),
        })
        attach_client_response(turn, {
            'type': 'responses.stream.summary',
            'model': ctx.client_model,
            'events': client_events,
        })
        finalize_turn(turn)

    return sse_response(generate())


def _finalize_responses_response(response_data: dict[str, Any], *, turn: dict[str, Any], debug_label: str):
    """统一收尾非流式 Responses 响应。

    两条转换链路和一条原生 Responses 链路最终都会回到 Responses 对象，因此这里集中
    处理调试日志、回填展示模型名以及 usage 日志。
    """
    response_data['model'] = response_data.get('model') or ''
    _dbg(debug_label + '=' + json.dumps(response_data, ensure_ascii=False, default=str)[:1000])
    log_usage('响应生成', response_data.get('usage', {}), input_key='input_tokens', output_key='output_tokens')

    usage_tracker.record(
        response_data.get('model', ''),
        response_data.get('usage'),
        input_key='input_tokens',
        output_key='output_tokens',
    )

    attach_client_response(turn, response_data)
    finalize_turn(turn, usage=response_data.get('usage'))

    return jsonify(response_data)
