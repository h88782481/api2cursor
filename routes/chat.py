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
    build_anthropic_target,
    build_openai_target,
    build_responses_target,
    build_route_context,
    chat_error_chunk,
    log_route_context,
    log_usage,
    sse_data_message,
)
from utils.http import (
    forward_request,
    iter_anthropic_sse,
    iter_openai_sse,
    iter_responses_sse,
    sse_response,
)
from utils.think_tag import ThinkTagExtractor

logger = logging.getLogger(__name__)

bp = Blueprint('chat', __name__)


def _dbg(message: str) -> None:
    """仅在调试模式下输出详细日志。"""
    if Config.DEBUG:
        logger.info('[聊天补全调试] %s', message)


@bp.route('/v1/chat/completions', methods=['POST'])
def chat_completions():
    """处理聊天补全请求并按模型映射分发到不同后端。"""
    payload = request.get_json(force=True)
    payload, message_count = _normalize_chat_payload(payload)

    client_model = payload.get('model', 'unknown')
    is_stream = payload.get('stream', False)
    ctx = build_route_context(client_model, is_stream)

    log_route_context('聊天补全', ctx, extra=f'消息数={message_count}')
    _log_messages(payload)

    if ctx.backend == 'openai':
        return _handle_openai_backend(ctx, payload)
    if ctx.backend == 'responses':
        return _handle_responses_backend(ctx, payload)
    return _handle_anthropic_backend(ctx, payload)


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


def _handle_openai_backend(ctx: RouteContext, payload: dict[str, Any]):
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
    _dbg(
        f'标准化完成：模型={payload.get("model")} '
        f'工具数={len(payload.get("tools", []))}'
    )

    url, headers = build_openai_target(ctx)

    if ctx.is_stream:
        return _handle_openai_stream(ctx, payload, url, headers)
    return _handle_openai_non_stream(ctx, payload, url, headers)


def _handle_openai_non_stream(
    ctx: RouteContext,
    payload: dict[str, Any],
    url: str,
    headers: dict[str, str],
):
    """处理 OpenAI 兼容后端的非流式返回。"""
    payload['stream'] = False
    resp, err = forward_request(url, headers, payload)
    if err:
        return err

    raw = resp.json()
    _dbg('上游原始响应=' + json.dumps(raw, ensure_ascii=False, default=str)[:1000])

    data = fix_response(raw)
    return _finalize_chat_response(ctx, data, debug_label='修复后响应')


def _handle_openai_stream(
    ctx: RouteContext,
    payload: dict[str, Any],
    url: str,
    headers: dict[str, str],
):
    """处理 OpenAI 兼容后端的流式返回。"""
    payload['stream'] = True

    def generate():
        """消费上游 OpenAI SSE，并逐段产出给 Cursor 的聊天补全流。"""
        resp, err = forward_request(url, headers, payload, stream=True)
        if err:
            yield chat_error_chunk(str(err))
            return

        think_extractor = ThinkTagExtractor()
        chunk_count = 0

        for chunk in iter_openai_sse(resp):
            if chunk is None:
                _dbg(f'流式响应结束，共 {chunk_count} 个数据片段')
                yield sse_data_message('[DONE]')
                return

            if chunk_count < 10:
                _dbg(
                    f'上游原始片段#{chunk_count}='
                    + json.dumps(chunk, ensure_ascii=False, default=str)[:500]
                )

            chunk = fix_stream_chunk(chunk)
            chunk['model'] = ctx.client_model

            for out in think_extractor.process_chunk(chunk):
                if chunk_count < 10:
                    _dbg(
                        f'返回片段#{chunk_count}='
                        + json.dumps(out, ensure_ascii=False, default=str)[:500]
                    )
                yield sse_data_message(out)

            chunk_count += 1

    return sse_response(generate())


def _handle_responses_backend(ctx: RouteContext, payload: dict[str, Any]):
    """处理走原生 Responses 后端的聊天补全请求。

    当上游只支持 `/v1/responses` 时，需要先把聊天补全请求转换为 Responses 请求，
    返回时再转换回聊天补全协议。
    """
    responses_payload = cc_to_responses_request(payload)
    responses_payload['model'] = ctx.upstream_model
    _dbg(
        '已转换为 Responses 请求：字段=' + str(list(responses_payload.keys()))
        + f' 输入项数={len(responses_payload.get("input", []))}'
    )

    url, headers = build_responses_target(ctx)

    if ctx.is_stream:
        return _handle_responses_stream(ctx, responses_payload, url, headers)
    return _handle_responses_non_stream(ctx, responses_payload, url, headers)


def _handle_responses_non_stream(
    ctx: RouteContext,
    payload: dict[str, Any],
    url: str,
    headers: dict[str, str],
):
    """处理原生 Responses 后端的非流式返回。"""
    payload['stream'] = False
    resp, err = forward_request(url, headers, payload)
    if err:
        return err

    raw = resp.json()
    _dbg('上游原始响应=' + json.dumps(raw, ensure_ascii=False, default=str)[:1000])

    data = responses_to_cc_response(raw, ctx.client_model)
    return _finalize_chat_response(ctx, data, debug_label='Responses 转回聊天补全后')


def _handle_responses_stream(
    ctx: RouteContext,
    payload: dict[str, Any],
    url: str,
    headers: dict[str, str],
):
    """处理原生 Responses 后端的流式返回。"""
    payload['stream'] = True
    converter = ResponsesToCCStreamConverter(model=ctx.client_model)

    def generate():
        """消费上游 Responses 事件，并实时转换成聊天补全 chunk。"""
        resp, err = forward_request(url, headers, payload, stream=True)
        if err:
            yield chat_error_chunk(str(err))
            return

        event_count = 0
        for event_type, event_data in iter_responses_sse(resp):
            if event_count < 10:
                _dbg(
                    f'上游事件#{event_count} 类型={event_type} 数据='
                    + json.dumps(event_data, ensure_ascii=False, default=str)[:500]
                )

            for chunk in converter.process_event(event_type, event_data):
                if event_count < 10:
                    _dbg(
                        f'返回片段#{event_count}='
                        + json.dumps(chunk, ensure_ascii=False, default=str)[:500]
                    )
                yield sse_data_message(chunk)

            event_count += 1

        _dbg(f'流式响应结束，共 {event_count} 个事件')
        yield sse_data_message('[DONE]')

    return sse_response(generate())


def _handle_anthropic_backend(ctx: RouteContext, payload: dict[str, Any]):
    """处理走 Anthropic Messages 后端的聊天补全请求。"""
    payload['model'] = ctx.upstream_model
    anthropic_payload = cc_to_messages_request(payload)
    _dbg(
        '已转换为 Messages 请求：字段=' + str(list(anthropic_payload.keys()))
        + f' 消息数={len(anthropic_payload.get("messages", []))}'
    )

    url, headers = build_anthropic_target(ctx)

    if ctx.is_stream:
        return _handle_anthropic_stream(ctx, anthropic_payload, url, headers)
    return _handle_anthropic_non_stream(ctx, anthropic_payload, url, headers)


def _handle_anthropic_non_stream(
    ctx: RouteContext,
    payload: dict[str, Any],
    url: str,
    headers: dict[str, str],
):
    """处理 Anthropic 后端的非流式返回。"""
    payload['stream'] = False
    resp, err = forward_request(url, headers, payload)
    if err:
        return err

    raw = resp.json()
    _dbg('上游原始响应=' + json.dumps(raw, ensure_ascii=False, default=str)[:1000])

    data = messages_to_cc_response(raw)
    return _finalize_chat_response(ctx, data, debug_label='Messages 转回聊天补全后')


def _handle_anthropic_stream(
    ctx: RouteContext,
    payload: dict[str, Any],
    url: str,
    headers: dict[str, str],
):
    """处理 Anthropic 后端的流式返回。

    这里仍然保留独立的事件级转换器，而不是先落成完整响应再回放，
    是为了尽量保持 Cursor 端的流式体验和工具调用时序。
    """
    payload['stream'] = True
    converter = AnthropicStreamConverter()

    def generate():
        """消费上游 Anthropic 事件流，并逐步映射为聊天补全 SSE。"""
        resp, err = forward_request(url, headers, payload, stream=True)
        if err:
            yield chat_error_chunk(str(err))
            return

        event_count = 0
        for event_type, event_data in iter_anthropic_sse(resp):
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

                if event_count < 10:
                    _dbg(f'返回片段#{event_count}={chunk_str[:500]}')
                yield sse_data_message(chunk_str)

            event_count += 1

        _dbg(f'流式响应结束，共 {event_count} 个事件')
        yield sse_data_message('[DONE]')

    return sse_response(generate())


def _finalize_chat_response(
    ctx: RouteContext,
    data: dict[str, Any],
    *,
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
