"""请求编排管线

一个请求的完整旅程：

    入口路由 → 请求体格式检测（BYOK bug 兼容）→ 路由决策
      → 请求准备（原生透传 或 解析为 IR 再构建上游格式）
      → 转发中转站
      → 响应处理（原生透传 或 解析为 IR 再编码回 Cursor 格式）

两条路径：
- 原生路径：解析格式 == 上游格式时请求侧直接透传（保留未知字段）；
  responses→responses / messages→messages 的响应侧也透传（只做模型名改写和兼容注入）
- IR 路径：其余组合走 `解析 → IR → 构建`，流式为 `解码 → IR 事件 → 过滤器 → 编码`
"""

from __future__ import annotations

import logging
from typing import Any, AsyncIterator

import httpx
from fastapi import Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

from ..compat.detect import looks_like_responses_payload, sanitize_responses_payload
from ..compat.thinking import (
    MessagesThinkingStreamFilter,
    ThinkTagFilter,
    extract_think_from_response,
    inject_thinking_into_messages_response,
    thinking_cache,
)
from ..compat.tools import fix_tool_call_block
from ..protocols import get_codec
from ..protocols.anthropic import parse_messages_usage
from ..protocols.base import parse_json, sse_data, sse_event
from ..protocols.chat_completions import normalize_cc_request
from ..protocols.responses_api import ensure_prompt_cache_key, parse_responses_usage
from ..services import request_log
from ..services.usage import usage_tracker
from .ir import IRRequest, IRUsage, StreamEnd, StreamError, StreamEvent, ThinkingDelta
from .routing import RouteDecision, resolve_route
from .upstream import iter_sse, post_json, post_stream

logger = logging.getLogger(__name__)

_SSE_HEADERS = {'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'}

# 响应侧可以原生透传的格式（协议事件种类多，IR 往返会丢失未建模的事件/字段）
_RAW_RESPONSE_FORMATS = ('responses', 'messages')


async def handle_entry(request: Request, entry_format: str) -> Response:
    """三个数据面入口的统一处理函数。"""
    try:
        payload = await request.json()
    except Exception:
        return _error_json('无效的 JSON 请求体', 'invalid_request_error', 400)
    if not isinstance(payload, dict):
        return _error_json('请求体必须是 JSON 对象', 'invalid_request_error', 400)

    # ─── BYOK bug 检测：chat 入口收到 Responses 风格请求体 ───
    parse_format = entry_format
    if entry_format == 'chat' and looks_like_responses_payload(payload):
        parse_format = 'responses'
        logger.info('检测到 Responses 风格请求体进入 /v1/chat/completions（Cursor BYOK 兼容），'
                    '将按 Responses 解析、按 Chat Completions 返回')

    client_model = str(payload.get('model', 'unknown'))
    stream = bool(payload.get('stream', False))
    decision = resolve_route(entry_format, client_model, stream)

    logger.info(
        '[%s] 模型=%s 上游模型=%s 解析格式=%s 上游格式=%s 流式=%s',
        entry_format, client_model, decision.upstream_model,
        parse_format, decision.upstream_format, stream,
    )

    turn = request_log.start_turn(
        route=entry_format,
        client_model=client_model,
        client_format=decision.client_format,
        upstream_format=decision.upstream_format,
        stream=stream,
        client_request=payload,
        request_headers=dict(request.headers),
        target_url=decision.target_url,
        upstream_model=decision.upstream_model,
        metadata={'parse_format': parse_format},
    )

    upstream_codec = get_codec(decision.upstream_format)
    client_codec = get_codec(decision.client_format)

    # ─── 请求准备 ───
    ir_request: IRRequest | None = None
    if parse_format == decision.upstream_format:
        body = _prepare_native_request(parse_format, payload, decision)
    else:
        ir_request = get_codec(parse_format).parse_request(payload)
        ir_request.model = decision.upstream_model
        ir_request.stream = stream
        if decision.custom_instructions:
            ir_request.system = _merge_text(
                decision.custom_instructions, ir_request.system, decision.instructions_position,
            )
        # Responses 上游通过自身的 reasoning 项管理思考内容，不做缓存注入
        if decision.upstream_format != 'responses':
            thinking_cache.inject_ir(ir_request)
        body = upstream_codec.build_request(ir_request, decision.upstream_model)

    _apply_body_modifications(body, decision.body_modifications)
    headers = upstream_codec.build_headers(decision.api_key)
    _apply_header_modifications(headers, decision.header_modifications)
    url = upstream_codec.upstream_url(decision.target_url, decision.upstream_model, stream)

    raw_response = (
        decision.client_format == decision.upstream_format
        and decision.client_format in _RAW_RESPONSE_FORMATS
    )

    client: httpx.AsyncClient = request.app.state.http_client
    request_log.attach_upstream_request(turn, body, headers)
    request_log.debug_log(entry_format, f'上游地址={url} 请求字段={list(body.keys())}')

    if stream:
        return await _handle_stream(
            client, url, headers, body, decision, parse_format,
            raw_response, ir_request, payload, turn,
        )
    return await _handle_non_stream(
        client, url, headers, body, decision, parse_format,
        raw_response, ir_request, payload, turn,
    )


# ═══════════════════════════════════════════════════════════
#  非流式
# ═══════════════════════════════════════════════════════════


async def _handle_non_stream(
    client: httpx.AsyncClient,
    url: str,
    headers: dict[str, str],
    body: dict[str, Any],
    decision: RouteDecision,
    parse_format: str,
    raw_response: bool,
    ir_request: IRRequest | None,
    original_payload: dict[str, Any],
    turn: dict[str, Any] | None,
) -> Response:
    try:
        resp = await post_json(client, url, headers, body)
    except httpx.HTTPError as e:
        logger.error('请求上游失败: %s', e)
        request_log.attach_error(turn, {'stage': 'forward_request', 'message': str(e)})
        request_log.finalize_turn(turn)
        return _error_json(f'请求上游失败: {e}', 'proxy_error', 502)

    if resp.status_code >= 400:
        logger.warning('上游返回 %s: %s', resp.status_code, resp.text[:300])
        request_log.attach_error(turn, {
            'stage': 'upstream_status',
            'status_code': resp.status_code,
            'message': resp.text[:2000],
        })
        request_log.finalize_turn(turn)
        return Response(
            content=resp.content,
            status_code=resp.status_code,
            media_type=resp.headers.get('content-type', 'application/json'),
        )

    try:
        data = resp.json()
    except ValueError:
        request_log.attach_error(turn, {'stage': 'parse_response', 'message': '上游返回非 JSON 响应'})
        request_log.finalize_turn(turn)
        return _error_json('上游返回非 JSON 响应', 'upstream_error', 502)

    request_log.attach_upstream_response(turn, data)

    reasoning = ''
    if raw_response:
        result, usage = _finalize_raw_response(decision, data)
    else:
        upstream_codec = get_codec(decision.upstream_format)
        ir_response = upstream_codec.parse_response(data)
        if decision.upstream_format == 'chat':
            extract_think_from_response(ir_response)
        for block in ir_response.tool_calls():
            fix_tool_call_block(block)
        result = get_codec(decision.client_format).build_response(ir_response, decision.client_model)
        usage = ir_response.usage
        reasoning = ir_response.thinking()

    usage_tracker.record(decision.client_model, usage)
    _store_thinking(reasoning, ir_request, parse_format, original_payload)
    logger.info('[%s] 请求完成 输入令牌=%s 输出令牌=%s',
                decision.client_format, usage.input_tokens, usage.output_tokens)

    request_log.attach_client_response(turn, result)
    request_log.finalize_turn(turn, usage=_usage_dict(usage))
    return JSONResponse(result)


def _finalize_raw_response(
    decision: RouteDecision,
    data: dict[str, Any],
) -> tuple[dict[str, Any], IRUsage]:
    """responses→responses / messages→messages 的非流式透传收尾。"""
    if decision.upstream_format == 'messages':
        inject_thinking_into_messages_response(data)
        usage = parse_messages_usage(data.get('usage'))
    else:
        usage = parse_responses_usage(data.get('usage'))

    if data.get('model'):
        data['model'] = decision.client_model
    return data, usage


# ═══════════════════════════════════════════════════════════
#  流式
# ═══════════════════════════════════════════════════════════


async def _handle_stream(
    client: httpx.AsyncClient,
    url: str,
    headers: dict[str, str],
    body: dict[str, Any],
    decision: RouteDecision,
    parse_format: str,
    raw_response: bool,
    ir_request: IRRequest | None,
    original_payload: dict[str, Any],
    turn: dict[str, Any] | None,
) -> Response:
    try:
        resp = await post_stream(client, url, headers, body)
    except httpx.HTTPError as e:
        logger.error('请求上游失败: %s', e)
        request_log.attach_error(turn, {'stage': 'forward_request', 'message': str(e)})
        request_log.finalize_turn(turn)
        return _error_json(f'请求上游失败: {e}', 'proxy_error', 502)

    if resp.status_code >= 400:
        error_body = await resp.aread()
        await resp.aclose()
        logger.warning('上游返回 %s: %s', resp.status_code, error_body[:300])
        request_log.attach_error(turn, {
            'stage': 'upstream_status',
            'status_code': resp.status_code,
            'message': error_body.decode('utf-8', errors='replace')[:2000],
        })
        request_log.finalize_turn(turn)
        return Response(
            content=error_body,
            status_code=resp.status_code,
            media_type=resp.headers.get('content-type', 'application/json'),
        )

    if raw_response:
        generator = _raw_stream(resp, decision, turn)
    else:
        generator = _convert_stream(
            resp, decision, parse_format, ir_request, original_payload, turn,
        )
    return StreamingResponse(generator, media_type='text/event-stream', headers=_SSE_HEADERS)


async def _convert_stream(
    resp: httpx.Response,
    decision: RouteDecision,
    parse_format: str,
    ir_request: IRRequest | None,
    original_payload: dict[str, Any],
    turn: dict[str, Any] | None,
) -> AsyncIterator[str]:
    """IR 转换流：上游 SSE → 解码 → 过滤器 → 编码 → 客户端 SSE。"""
    upstream_codec = get_codec(decision.upstream_format)
    client_codec = get_codec(decision.client_format)
    decoder = upstream_codec.stream_decoder()
    encoder = client_codec.stream_encoder(decision.client_model)
    filters = [ThinkTagFilter()] if decision.upstream_format == 'chat' else []

    usage: IRUsage | None = None
    thinking_parts: list[str] = []
    upstream_count = 0
    client_count = 0
    status = 'ok'

    def track(event: StreamEvent) -> None:
        nonlocal usage, status
        if isinstance(event, StreamEnd) and event.usage is not None:
            usage = event.usage
        elif isinstance(event, ThinkingDelta):
            thinking_parts.append(event.text)
        elif isinstance(event, StreamError):
            status = 'error'

    try:
        for message in encoder.start():
            client_count += 1
            request_log.append_client_event(turn, {'raw': message})
            yield message

        try:
            async for event_type, data in iter_sse(resp):
                upstream_count += 1
                request_log.append_upstream_event(turn, {'type': event_type, 'data': data})
                events = _run_filters(filters, decoder.decode(event_type, data))
                for event in events:
                    track(event)
                    for message in encoder.encode(event):
                        client_count += 1
                        request_log.append_client_event(turn, {'raw': message})
                        yield message

            tail = _run_filters(filters, decoder.finalize()) + _finalize_filters(filters)
            for event in tail:
                track(event)
                for message in encoder.encode(event):
                    client_count += 1
                    yield message
            for message in encoder.finalize():
                client_count += 1
                yield message
        except httpx.HTTPError as e:
            logger.error('读取上游流失败: %s', e)
            status = 'error'
            request_log.attach_error(turn, {'stage': 'stream_read', 'message': str(e)})
            yield client_codec.error_event(f'读取上游流失败: {e}', 'proxy_error')
    finally:
        await resp.aclose()
        usage_tracker.record(decision.client_model, usage)
        if thinking_parts:
            _store_thinking(''.join(thinking_parts), ir_request, parse_format, original_payload)
        request_log.set_stream_summary(turn, {
            'status': status,
            'upstream_event_count': upstream_count,
            'client_event_count': client_count,
            'usage': _usage_dict(usage) if usage else None,
        })
        request_log.finalize_turn(turn, usage=_usage_dict(usage) if usage else None)
        if usage is not None:
            logger.info('[%s] 流式完成 输入令牌=%s 输出令牌=%s',
                        decision.client_format, usage.input_tokens, usage.output_tokens)


async def _raw_stream(
    resp: httpx.Response,
    decision: RouteDecision,
    turn: dict[str, Any] | None,
) -> AsyncIterator[str]:
    """responses→responses / messages→messages 的流式透传。

    保留上游全部事件结构（包括 IR 未建模的事件类型），只做：
    - 模型名改写（Cursor 侧展示映射前的模型名）
    - messages 链路的 reasoning_content → thinking block 注入
    """
    is_messages = decision.upstream_format == 'messages'
    messages_filter = MessagesThinkingStreamFilter() if is_messages else None

    usage_input = 0
    usage_output = 0
    usage_seen = False
    upstream_count = 0
    client_count = 0

    try:
        try:
            async for event_type, data in iter_sse(resp):
                upstream_count += 1
                request_log.append_upstream_event(turn, {'type': event_type, 'data': data})

                payload = parse_json(data)
                if payload is None:
                    # 非 JSON 数据（如 [DONE]）原样透传
                    message = sse_event(event_type, data) if event_type else sse_data(data)
                    client_count += 1
                    yield message
                    continue

                etype = event_type or payload.get('type', '')

                if is_messages:
                    if etype == 'message_start' and isinstance(payload.get('message'), dict):
                        message_obj = payload['message']
                        if message_obj.get('model'):
                            message_obj['model'] = decision.client_model
                        start_usage = message_obj.get('usage')
                        if isinstance(start_usage, dict):
                            # 含 cache_read / cache_creation 的总输入口径
                            usage_input = parse_messages_usage(start_usage).input_tokens
                            usage_seen = True
                    elif etype == 'message_delta' and isinstance(payload.get('usage'), dict):
                        usage_output = payload['usage'].get('output_tokens', 0) or 0
                        usage_seen = True

                    for out_type, out_data in messages_filter.process(etype, payload):
                        message = sse_event(out_type, out_data) if out_type else sse_data(out_data)
                        client_count += 1
                        request_log.append_client_event(turn, {'raw': message})
                        yield message
                    continue

                # responses 透传：改写 response 对象中的模型名并提取 usage
                response_obj = payload.get('response')
                if isinstance(response_obj, dict):
                    if response_obj.get('model'):
                        response_obj['model'] = decision.client_model
                    if etype in ('response.completed', 'response.incomplete'):
                        raw_usage = response_obj.get('usage')
                        if isinstance(raw_usage, dict):
                            parsed = parse_responses_usage(raw_usage)
                            usage_input, usage_output = parsed.input_tokens, parsed.output_tokens
                            usage_seen = True
                if payload.get('model'):
                    payload['model'] = decision.client_model

                message = sse_event(etype, payload) if etype else sse_data(payload)
                client_count += 1
                request_log.append_client_event(turn, {'raw': message})
                yield message
        except httpx.HTTPError as e:
            logger.error('读取上游流失败: %s', e)
            request_log.attach_error(turn, {'stage': 'stream_read', 'message': str(e)})
            yield get_codec(decision.client_format).error_event(f'读取上游流失败: {e}', 'proxy_error')
    finally:
        await resp.aclose()
        usage = IRUsage(input_tokens=usage_input, output_tokens=usage_output) if usage_seen else None
        usage_tracker.record(decision.client_model, usage)
        request_log.set_stream_summary(turn, {
            'upstream_event_count': upstream_count,
            'client_event_count': client_count,
            'usage': _usage_dict(usage) if usage else None,
        })
        request_log.finalize_turn(turn, usage=_usage_dict(usage) if usage else None)


# ═══════════════════════════════════════════════════════════
#  请求准备
# ═══════════════════════════════════════════════════════════


def _prepare_native_request(
    parse_format: str,
    payload: dict[str, Any],
    decision: RouteDecision,
) -> dict[str, Any]:
    """解析格式 == 上游格式时的原生请求准备（保留未知字段）。"""
    payload['model'] = decision.upstream_model
    payload['stream'] = decision.stream
    instructions = decision.custom_instructions
    position = decision.instructions_position

    if parse_format == 'chat':
        normalize_cc_request(payload, decision.upstream_model)
        if instructions:
            _inject_instructions_chat(payload, instructions, position)
        payload['messages'] = thinking_cache.inject_cc(payload.get('messages') or [])
        return payload

    if parse_format == 'responses':
        sanitize_responses_payload(payload)
        if instructions:
            payload['instructions'] = _merge_text(
                instructions, str(payload.get('instructions') or ''), position,
            )
        ensure_prompt_cache_key(payload)
        return payload

    # messages
    if instructions:
        existing = payload.get('system') or ''
        if isinstance(existing, list):
            existing = '\n'.join(
                block.get('text', '') for block in existing
                if isinstance(block, dict) and block.get('type') == 'text'
            )
        payload['system'] = _merge_text(instructions, existing, position)
    return payload


def _inject_instructions_chat(payload: dict[str, Any], instructions: str, position: str) -> None:
    """向 CC 请求注入自定义指令（写入首条 system 消息）。"""
    messages = payload.get('messages')
    if not isinstance(messages, list):
        messages = []
        payload['messages'] = messages
    if messages and isinstance(messages[0], dict) and messages[0].get('role') == 'system':
        first = messages[0]
        original = first.get('content') or ''
        if isinstance(original, list):
            original = '\n'.join(
                part.get('text', '') for part in original
                if isinstance(part, dict) and part.get('type') == 'text'
            )
        elif not isinstance(original, str):
            original = str(original)
        first['content'] = _merge_text(instructions, original, position)
    else:
        messages.insert(0, {'role': 'system', 'content': instructions})


def _merge_text(custom: str, existing: str, position: str) -> str:
    """根据 position 决定自定义指令与原有内容的拼接顺序。"""
    if not existing:
        return custom
    if position == 'append':
        return existing + '\n\n' + custom
    return custom + '\n\n' + existing


def _apply_body_modifications(payload: dict[str, Any], modifications: dict[str, Any]) -> None:
    """对转发请求体应用字段级修改：值为 null 删除，其余设置/覆盖。"""
    if not modifications:
        return
    for key, value in modifications.items():
        if value is None:
            payload.pop(key, None)
        else:
            payload[key] = value
    logger.info('已应用 body_modifications: %s', list(modifications.keys()))


def _apply_header_modifications(headers: dict[str, str], modifications: dict[str, Any]) -> None:
    """对转发请求头应用字段级修改：值为 null 删除，其余设置/覆盖。"""
    if not modifications:
        return
    for key, value in modifications.items():
        if value is None:
            headers.pop(key, None)
        else:
            headers[key] = str(value)
    logger.info('已应用 header_modifications: %s', list(modifications.keys()))


# ═══════════════════════════════════════════════════════════
#  内部辅助
# ═══════════════════════════════════════════════════════════


def _run_filters(filters: list[Any], events: list[StreamEvent]) -> list[StreamEvent]:
    for f in filters:
        out: list[StreamEvent] = []
        for event in events:
            out.extend(f.process(event))
        events = out
    return events


def _finalize_filters(filters: list[Any]) -> list[StreamEvent]:
    """依次收尾过滤器，尾部事件继续流经后续过滤器。"""
    events: list[StreamEvent] = []
    for i, f in enumerate(filters):
        tail = f.finalize()
        for g in filters[i + 1:]:
            passed: list[StreamEvent] = []
            for event in tail:
                passed.extend(g.process(event))
            tail = passed
        events.extend(tail)
    return events


def _store_thinking(
    reasoning: str,
    ir_request: IRRequest | None,
    parse_format: str,
    original_payload: dict[str, Any],
) -> None:
    """把本次响应的思考内容写入 thinking 缓存。"""
    if not reasoning:
        return
    if ir_request is not None:
        thinking_cache.store_ir(ir_request, reasoning)
    elif parse_format == 'chat':
        messages = original_payload.get('messages')
        if isinstance(messages, list):
            thinking_cache.store_cc(messages, reasoning)


def _usage_dict(usage: IRUsage | None) -> dict[str, int]:
    if usage is None:
        return {'input_tokens': 0, 'output_tokens': 0}
    return {'input_tokens': usage.input_tokens, 'output_tokens': usage.output_tokens}


def _error_json(message: str, error_type: str, status: int) -> JSONResponse:
    return JSONResponse(
        {'error': {'message': message, 'type': error_type}},
        status_code=status,
    )
