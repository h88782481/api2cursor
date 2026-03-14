"""路由: /v1/chat/completions

处理 Cursor 发来的 OpenAI Chat Completions 格式请求。
根据模型映射的后端类型，转发到 OpenAI 兼容接口、Anthropic Messages 接口，
或原生 OpenAI Responses 接口。
"""

from __future__ import annotations

import json
import logging
from typing import Any

from flask import Blueprint, jsonify, request

from adapters.cc_anthropic_adapter import (
    AnthropicStreamConverter,
    cc_to_messages_request,
    messages_to_cc_response,
)
from adapters.cc_gemini_adapter import (
    GeminiStreamConverter,
    cc_to_gemini_request,
    gemini_to_cc_response,
)
from adapters.openai_compat_fixer import fix_response, fix_stream_chunk, normalize_request
from adapters.responses_cc_adapter import (
    ResponsesToCCStreamConverter,
    cc_to_responses_request,
    responses_to_cc,
    responses_to_cc_response,
)
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
    chat_error_chunk,
    inject_instructions_anthropic,
    inject_instructions_cc,
    inject_instructions_responses,
    log_route_context,
    log_usage,
    sse_data_message,
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

bp = Blueprint('chat', __name__)


def _dbg(message: str) -> None:
    """仅在调试模式下输出详细日志。"""
    if Config.DEBUG:
        logger.info('[聊天补全调试] %s', message)


@bp.route('/v1/chat/completions', methods=['POST'])
def chat_completions():
    """处理聊天补全请求并按模型映射分发到不同后端。"""
    original_payload = request.get_json(force=True)
    payload, message_count = _normalize_chat_payload(json.loads(json.dumps(original_payload, ensure_ascii=False, default=str)))

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

    payload['messages'] = thinking_cache.inject(payload.get('messages', []))

    if ctx.backend == 'openai':
        return _handle_openai_backend(ctx, payload, turn)
    if ctx.backend == 'responses':
        return _handle_responses_backend(ctx, payload, turn)
    if ctx.backend == 'gemini':
        return _handle_gemini_backend(ctx, payload, turn)
    return _handle_anthropic_backend(ctx, payload, turn)


def _normalize_chat_payload(payload: dict[str, Any]) -> tuple[dict[str, Any], int]:
    """整理聊天补全入口的请求体。

    这里保留了一层兼容逻辑：当 Cursor 或调用方把 Responses 格式误发到
    `/v1/chat/completions` 时，先降级转换成 Chat Completions，再进入统一主流程。
    """
    message_count = len(payload.get('messages', []))

    if message_count == 0 and 'input' in payload:
        logger.info('检测到 Responses 格式误入聊天补全接口，已自动转换为 Chat Completions 格式')
        payload = responses_to_cc(payload)
        message_count = len(payload.get('messages', []))
    elif message_count == 0:
        logger.warning('消息列表为空，请求字段=%s', list(payload.keys()))

    return payload, message_count


def _handle_openai_backend(ctx: RouteContext, payload: dict[str, Any], turn: dict[str, Any]):
    """处理走 OpenAI 兼容后端的聊天补全请求。"""
    _dbg(
        '原始请求字段=' + str(list(payload.keys())) + ' '
        + '附加字段='
        + json.dumps(
            {k: v for k, v in payload.items() if k != 'messages'},
            ensure_ascii=False,
            default=str,
        )[:500]
    )

    payload = normalize_request(payload, ctx.upstream_model)
    payload = inject_instructions_cc(payload, ctx.custom_instructions, ctx.instructions_position)
    _dbg(
        f'标准化完成：模型={payload.get("model")} '
        f'工具数={len(payload.get("tools", []))}'
    )

    url, headers = build_openai_target(ctx)
    payload = apply_body_modifications(payload, ctx.body_modifications)
    headers = apply_header_modifications(headers, ctx.header_modifications)

    if ctx.is_stream:
        return _handle_openai_stream(ctx, payload, url, headers, turn)
    return _handle_openai_non_stream(ctx, payload, url, headers, turn)


def _handle_openai_non_stream(
    ctx: RouteContext,
    payload: dict[str, Any],
    url: str,
    headers: dict[str, str],
    turn: dict[str, Any],
):
    """处理 OpenAI 兼容后端的非流式返回。"""
    payload['stream'] = False
    attach_upstream_request(turn, payload, headers)
    resp, err = forward_request(url, headers, payload)
    if err:
        attach_error(turn, {'stage': 'forward_request', 'message': 'upstream request failed'})
        finalize_turn(turn)
        return err

    raw = resp.json()
    attach_upstream_response(turn, raw)
    _dbg('上游原始响应=' + json.dumps(raw, ensure_ascii=False, default=str)[:1000])

    data = fix_response(raw)
    return _finalize_chat_response(ctx, data, turn=turn, debug_label='修复后响应')


def _handle_openai_stream(
    ctx: RouteContext,
    payload: dict[str, Any],
    url: str,
    headers: dict[str, str],
    turn: dict[str, Any],
):
    """处理 OpenAI 兼容后端的流式返回。"""
    payload['stream'] = True

    def generate():
        """消费上游 OpenAI SSE，并逐段产出给 Cursor 的聊天补全流。"""
        attach_upstream_request(turn, payload, headers)
        resp, err = forward_request(url, headers, payload, stream=True)
        if err:
            attach_error(turn, {'stage': 'forward_request', 'message': str(err)})
            set_stream_summary(turn, {'status': 'error'})
            finalize_turn(turn)
            yield chat_error_chunk(str(err))
            return

        think_extractor = ThinkTagExtractor()
        chunk_count = 0
        last_usage = None
        client_chunks: list[dict[str, Any]] = []

        for chunk in iter_openai_sse(resp):
            if chunk is None:
                _dbg(f'流式响应结束，共 {chunk_count} 个数据片段')
                close_chunk = think_extractor.finalize()
                if close_chunk:
                    client_chunks.append(close_chunk)
                    append_client_event(turn, {'type': 'chat_chunk', 'data': close_chunk})
                    yield sse_data_message(close_chunk)
                append_client_event(turn, {'type': 'done'})
                yield sse_data_message('[DONE]')
                usage_tracker.record(ctx.client_model, last_usage)
                set_stream_summary(turn, {
                    'chunk_count': chunk_count,
                    'client_chunk_count': len(client_chunks),
                    'usage': last_usage,
                })
                attach_client_response(turn, {
                    'type': 'chat.completion.stream.summary',
                    'model': ctx.client_model,
                    'chunks': client_chunks,
                    'usage': last_usage,
                })
                finalize_turn(turn, usage=last_usage)
                return

            append_upstream_event(turn, {'type': 'openai_chunk', 'data': chunk})
            if chunk.get('usage'):
                last_usage = chunk['usage']

            if chunk_count < 10:
                _dbg(
                    f'上游原始片段#{chunk_count}='
                    + json.dumps(chunk, ensure_ascii=False, default=str)[:500]
                )

            chunk = fix_stream_chunk(chunk)
            chunk['model'] = ctx.client_model

            for out in think_extractor.process_chunk(chunk):
                client_chunks.append(out)
                append_client_event(turn, {'type': 'chat_chunk', 'data': out})
                if chunk_count < 10:
                    _dbg(
                        f'返回片段#{chunk_count}='
                        + json.dumps(out, ensure_ascii=False, default=str)[:500]
                    )
                yield sse_data_message(out)

            chunk_count += 1

        usage_tracker.record(ctx.client_model, last_usage)
        set_stream_summary(turn, {
            'chunk_count': chunk_count,
            'client_chunk_count': len(client_chunks),
            'usage': last_usage,
            'ended_without_done': True,
        })
        attach_client_response(turn, {
            'type': 'chat.completion.stream.summary',
            'model': ctx.client_model,
            'chunks': client_chunks,
            'usage': last_usage,
        })
        finalize_turn(turn, usage=last_usage)

    return sse_response(generate())


def _handle_responses_backend(ctx: RouteContext, payload: dict[str, Any], turn: dict[str, Any] | None):
    """处理走原生 Responses 后端的聊天补全请求。

    当上游只支持 `/v1/responses` 时，需要先把聊天补全请求转换为 Responses 请求，
    返回时再转换回聊天补全协议。
    """
    responses_payload = cc_to_responses_request(payload)
    responses_payload['model'] = ctx.upstream_model
    responses_payload = inject_instructions_responses(responses_payload, ctx.custom_instructions, ctx.instructions_position)
    _dbg(
        '已转换为 Responses 请求：字段=' + str(list(responses_payload.keys()))
        + f' 输入项数={len(responses_payload.get("input", []))}'
    )

    url, headers = build_responses_target(ctx)
    responses_payload = apply_body_modifications(responses_payload, ctx.body_modifications)
    headers = apply_header_modifications(headers, ctx.header_modifications)

    if ctx.is_stream:
        return _handle_responses_stream(ctx, responses_payload, url, headers, turn)
    return _handle_responses_non_stream(ctx, responses_payload, url, headers, turn)


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

    raw = resp.json()
    attach_upstream_response(turn, raw)
    _dbg('上游原始响应=' + json.dumps(raw, ensure_ascii=False, default=str)[:1000])

    data = responses_to_cc_response(raw, ctx.client_model)
    return _finalize_chat_response(ctx, data, turn=turn, debug_label='Responses 转回聊天补全后')


def _handle_responses_stream(
    ctx: RouteContext,
    payload: dict[str, Any],
    url: str,
    headers: dict[str, str],
    turn: dict[str, Any] | None,
):
    """处理原生 Responses 后端的流式返回。"""
    payload['stream'] = True
    converter = ResponsesToCCStreamConverter(model=ctx.client_model)

    def generate():
        """消费上游 Responses 事件，并实时转换成聊天补全 chunk。"""
        attach_upstream_request(turn, payload, headers)
        resp, err = forward_request(url, headers, payload, stream=True)
        if err:
            attach_error(turn, {'stage': 'forward_request', 'message': str(err)})
            set_stream_summary(turn, {'status': 'error'})
            finalize_turn(turn)
            yield chat_error_chunk(str(err))
            return

        event_count = 0
        client_chunks: list[Any] = []
        for event_type, event_data in iter_responses_sse(resp):
            append_upstream_event(turn, {'type': event_type, 'data': event_data})
            if event_count < 10:
                _dbg(
                    f'上游事件#{event_count} 类型={event_type} 数据='
                    + json.dumps(event_data, ensure_ascii=False, default=str)[:500]
                )

            for chunk in converter.process_event(event_type, event_data):
                client_chunks.append(chunk)
                append_client_event(turn, {'type': 'chat_chunk', 'data': chunk})
                if event_count < 10:
                    _dbg(
                        f'返回片段#{event_count}='
                        + json.dumps(chunk, ensure_ascii=False, default=str)[:500]
                    )
                yield sse_data_message(chunk)

            event_count += 1

        _dbg(f'流式响应结束，共 {event_count} 个事件')
        append_client_event(turn, {'type': 'done'})
        yield sse_data_message('[DONE]')
        usage_tracker.record(ctx.client_model)
        set_stream_summary(turn, {
            'event_count': event_count,
            'client_chunk_count': len(client_chunks),
        })
        attach_client_response(turn, {
            'type': 'chat.completion.stream.summary',
            'model': ctx.client_model,
            'chunks': client_chunks,
        })
        finalize_turn(turn)

    return sse_response(generate())


def _handle_gemini_backend(ctx: RouteContext, payload: dict[str, Any], turn: dict[str, Any] | None):
    """处理走 Gemini Contents 后端的聊天补全请求。"""
    payload = inject_instructions_cc(payload, ctx.custom_instructions, ctx.instructions_position)
    gemini_payload = cc_to_gemini_request(payload)
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
    """处理 Gemini 后端的非流式返回。"""
    attach_upstream_request(turn, payload, headers)
    resp, err = forward_request(url, headers, payload)
    if err:
        attach_error(turn, {'stage': 'forward_request', 'message': 'upstream request failed'})
        finalize_turn(turn)
        return err

    raw = resp.json()
    attach_upstream_response(turn, raw)
    _dbg('上游原始响应=' + json.dumps(raw, ensure_ascii=False, default=str)[:1000])

    data = gemini_to_cc_response(raw)
    return _finalize_chat_response(ctx, data, turn=turn, debug_label='Gemini 转回聊天补全后')


def _handle_gemini_stream(
    ctx: RouteContext,
    payload: dict[str, Any],
    url: str,
    headers: dict[str, str],
    turn: dict[str, Any] | None,
):
    """处理 Gemini 后端的流式返回。"""
    converter = GeminiStreamConverter()

    def generate():
        attach_upstream_request(turn, payload, headers)
        resp, err = forward_request(url, headers, payload, stream=True)
        if err:
            attach_error(turn, {'stage': 'forward_request', 'message': str(err)})
            set_stream_summary(turn, {'status': 'error'})
            finalize_turn(turn)
            yield chat_error_chunk(str(err))
            return

        chunk_count = 0
        client_chunks: list[Any] = []
        for gemini_chunk in iter_gemini_sse(resp):
            append_upstream_event(turn, {'type': 'gemini_chunk', 'data': gemini_chunk})
            if chunk_count < 10:
                _dbg(
                    f'上游 Gemini 片段#{chunk_count}='
                    + json.dumps(gemini_chunk, ensure_ascii=False, default=str)[:500]
                )

            for cc_chunk in converter.process_chunk(gemini_chunk):
                cc_chunk['model'] = ctx.client_model
                client_chunks.append(cc_chunk)
                append_client_event(turn, {'type': 'chat_chunk', 'data': cc_chunk})
                if chunk_count < 10:
                    _dbg(
                        f'返回片段#{chunk_count}='
                        + json.dumps(cc_chunk, ensure_ascii=False, default=str)[:500]
                    )
                yield sse_data_message(cc_chunk)

            chunk_count += 1

        _dbg(f'流式响应结束，共 {chunk_count} 个数据片段')
        append_client_event(turn, {'type': 'done'})
        yield sse_data_message('[DONE]')
        usage_tracker.record(ctx.client_model)
        set_stream_summary(turn, {
            'chunk_count': chunk_count,
            'client_chunk_count': len(client_chunks),
        })
        attach_client_response(turn, {
            'type': 'chat.completion.stream.summary',
            'model': ctx.client_model,
            'chunks': client_chunks,
        })
        finalize_turn(turn)

    return sse_response(generate())


def _handle_anthropic_backend(ctx: RouteContext, payload: dict[str, Any], turn: dict[str, Any] | None):
    """处理走 Anthropic Messages 后端的聊天补全请求。"""
    payload['model'] = ctx.upstream_model
    anthropic_payload = cc_to_messages_request(payload)
    anthropic_payload = inject_instructions_anthropic(anthropic_payload, ctx.custom_instructions, ctx.instructions_position)
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
    payload: dict[str, Any],
    url: str,
    headers: dict[str, str],
    turn: dict[str, Any] | None,
):
    """处理 Anthropic 后端的非流式返回。"""
    payload['stream'] = False
    attach_upstream_request(turn, payload, headers)
    resp, err = forward_request(url, headers, payload)
    if err:
        attach_error(turn, {'stage': 'forward_request', 'message': 'upstream request failed'})
        finalize_turn(turn)
        return err

    raw = resp.json()
    attach_upstream_response(turn, raw)
    _dbg('上游原始响应=' + json.dumps(raw, ensure_ascii=False, default=str)[:1000])

    data = messages_to_cc_response(raw)
    return _finalize_chat_response(ctx, data, turn=turn, debug_label='Messages 转回聊天补全后')


def _handle_anthropic_stream(
    ctx: RouteContext,
    payload: dict[str, Any],
    url: str,
    headers: dict[str, str],
    turn: dict[str, Any] | None,
):
    """处理 Anthropic 后端的流式返回。

    这里仍然保留独立的事件级转换器，而不是先落成完整响应再回放，
    是为了尽量保持 Cursor 端的流式体验和工具调用时序。
    """
    payload['stream'] = True
    converter = AnthropicStreamConverter()

    def generate():
        """消费上游 Anthropic 事件流，并逐步映射为聊天补全 SSE。"""
        attach_upstream_request(turn, payload, headers)
        resp, err = forward_request(url, headers, payload, stream=True)
        if err:
            attach_error(turn, {'stage': 'forward_request', 'message': str(err)})
            set_stream_summary(turn, {'status': 'error'})
            finalize_turn(turn)
            yield chat_error_chunk(str(err))
            return

        event_count = 0
        client_chunks: list[Any] = []
        for event_type, event_data in iter_anthropic_sse(resp):
            append_upstream_event(turn, {'type': event_type, 'data': event_data})
            if event_count < 10:
                _dbg(
                    f'上游事件#{event_count} 类型={event_type} 数据='
                    + json.dumps(event_data, ensure_ascii=False, default=str)[:500]
                )

            for chunk_str in converter.process_event(event_type, event_data):
                try:
                    chunk_obj = json.loads(chunk_str)
                    chunk_obj['model'] = ctx.client_model
                    chunk_str = json.dumps(chunk_obj, ensure_ascii=False)
                except (json.JSONDecodeError, TypeError):
                    pass

                client_chunks.append(chunk_str)
                append_client_event(turn, {'type': 'chat_chunk', 'data': chunk_str})
                if event_count < 10:
                    _dbg(f'返回片段#{event_count}={chunk_str[:500]}')
                yield sse_data_message(chunk_str)

            event_count += 1

        _dbg(f'流式响应结束，共 {event_count} 个事件')
        append_client_event(turn, {'type': 'done'})
        yield sse_data_message('[DONE]')
        usage_tracker.record(ctx.client_model)
        set_stream_summary(turn, {
            'event_count': event_count,
            'client_chunk_count': len(client_chunks),
        })
        attach_client_response(turn, {
            'type': 'chat.completion.stream.summary',
            'model': ctx.client_model,
            'chunks': client_chunks,
        })
        finalize_turn(turn)

    return sse_response(generate())


def _finalize_chat_response(
    ctx: RouteContext,
    data: dict[str, Any],
    *,
    turn: dict[str, Any] | None,
    debug_label: str,
):
    """统一收尾非流式聊天补全响应。

    三条后端链路最终都会回到 Chat Completions 格式，因此这里集中做：
    - 回填给 Cursor 展示的模型名
    - 输出统一调试日志
    - 输出统一令牌统计日志
    """
    data['model'] = ctx.client_model
    _dbg(debug_label + '=' + json.dumps(data, ensure_ascii=False, default=str)[:1000])
    log_usage('聊天补全', data.get('usage', {}), input_key='prompt_tokens', output_key='completion_tokens')

    usage_tracker.record(ctx.client_model, data.get('usage'))
    attach_client_response(turn, data)
    finalize_turn(turn, usage=data.get('usage'))

    for choice in data.get('choices', []):
        msg = choice.get('message', {})
        if msg.get('reasoning_content'):
            thinking_cache.store_from_response(
                request.get_json(silent=True, force=True).get('messages', []),
                msg['reasoning_content'],
            )
            break

    return jsonify(data)


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
