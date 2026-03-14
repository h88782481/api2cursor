"""路由: /v1/messages

Anthropic Messages API 透传。当 Cursor 直接发送 Anthropic 格式请求时，
直接转发到上游并原样返回。处理非标准的 reasoning_content 字段，
将其注入为标准的 thinking blocks。
"""

import json
import logging

import requests as req_lib
from flask import Blueprint, request, jsonify

import settings
from config import Config
from routes.common import apply_body_modifications, apply_header_modifications, inject_instructions_anthropic
from utils.http import build_anthropic_headers, forward_request, sse_response
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

logger = logging.getLogger(__name__)

bp = Blueprint('messages', __name__)


@bp.route('/v1/messages', methods=['POST'])
def messages_passthrough():
    """透传 Anthropic Messages 请求，并在必要时补齐 thinking 兼容层。"""
    original_payload = request.get_json(force=True)
    payload = json.loads(json.dumps(original_payload, ensure_ascii=False, default=str))
    model = payload.get('model', 'unknown')
    is_stream = payload.get('stream', False)

    logger.info(f'[透传] model={model} 流式={is_stream}')

    mapping = settings.resolve_model(model)
    url_base = mapping['target_url']
    api_key = mapping['api_key']
    custom_instructions = mapping.get('custom_instructions', '')
    instructions_position = mapping.get('instructions_position', 'prepend')
    body_mods = mapping.get('body_modifications', {})
    header_mods = mapping.get('header_modifications', {})
    headers = build_anthropic_headers(api_key)
    headers = apply_header_modifications(headers, header_mods)
    url = f'{url_base.rstrip("/")}/v1/messages'

    turn = start_turn(
        route='messages',
        client_model=model,
        backend='anthropic',
        stream=is_stream,
        client_request=original_payload,
        request_headers=dict(request.headers),
        target_url=url_base,
        upstream_model=model,
    )

    payload = inject_instructions_anthropic(payload, custom_instructions, instructions_position)
    payload = apply_body_modifications(payload, body_mods)

    if not is_stream:
        attach_upstream_request(turn, payload, headers)
        resp, err = forward_request(url, headers, payload)
        if err:
            attach_error(turn, {'stage': 'forward_request', 'message': 'upstream request failed'})
            finalize_turn(turn)
            return err
        data = resp.json()
        attach_upstream_response(turn, data)
        _inject_thinking(data)
        attach_client_response(turn, data)
        finalize_turn(turn)
        return jsonify(data)

    def generate():
        """建立上游流式连接并逐段回传处理后的 SSE 数据。"""
        attach_upstream_request(turn, payload, headers)
        try:
            resp = req_lib.post(
                url, headers=headers, json=payload,
                timeout=Config.API_TIMEOUT, stream=True,
            )
            if resp.status_code != 200:
                body = resp.content.decode('utf-8', errors='replace')
                logger.warning(f'上游返回 {resp.status_code}: {body[:300]}')
                attach_error(turn, {'stage': 'upstream_status', 'status_code': resp.status_code, 'message': body})
                set_stream_summary(turn, {'status': 'error'})
                finalize_turn(turn)
                yield f'data: {json.dumps({"error": {"message": body, "type": "upstream_error"}})}\n\n'
                return

            summary = {'upstream_event_count': 0, 'client_event_count': 0}
            client_events = []
            for out in _process_stream(resp, turn=turn, summary=summary):
                client_events.append(out)
                yield out
            set_stream_summary(turn, summary)
            attach_client_response(turn, {
                'type': 'messages.stream.summary',
                'events': client_events,
            })
            finalize_turn(turn)
        except req_lib.RequestException as e:
            logger.error(f'请求上游失败: {e}')
            attach_error(turn, {'stage': 'request_exception', 'message': str(e)})
            set_stream_summary(turn, {'status': 'error'})
            finalize_turn(turn)
            yield f'data: {json.dumps({"error": {"message": str(e), "type": "proxy_error"}})}\n\n'

    return sse_response(generate())


# ─── 内部辅助 ─────────────────────────────────────


def _inject_thinking(data):
    """将非标准 reasoning_content 字段注入为 Anthropic thinking block"""
    rc = data.pop('reasoning_content', None) or data.pop('reasoningContent', None)
    if not rc:
        return

    content = data.get('content')
    if not isinstance(content, list):
        content = []

    # 避免重复注入
    if any(isinstance(b, dict) and b.get('type') == 'thinking' for b in content):
        return

    content.insert(0, {'type': 'thinking', 'thinking': rc})
    data['content'] = content
    logger.info(f'已注入 thinking block ({len(rc)} 字符)')


def _process_stream(resp, *, turn=None, summary: dict[str, int] | None = None):
    """处理 /v1/messages 流式响应，检测并注入 thinking 事件

    追踪上游 content block 的 index，在注入 thinking blocks 时使用独立的 index，
    并将后续上游 block 的 index 偏移，避免冲突。
    """
    reasoning_buf = ''
    injected = False
    index_offset = 0
    if summary is None:
        summary = {'upstream_event_count': 0, 'client_event_count': 0}

    for line in resp.iter_lines():
        if not line:
            continue
        decoded = line.decode('utf-8', errors='replace')
        append_upstream_event(turn, {'raw': decoded})
        summary['upstream_event_count'] += 1

        if not decoded.startswith('data:'):
            yield decoded + '\n\n'
            continue

        data_str = decoded[5:].strip()
        if not data_str:
            yield decoded + '\n\n'
            continue

        try:
            event_data = json.loads(data_str)
        except json.JSONDecodeError:
            yield decoded + '\n\n'
            continue

        modified = False

        for container_key in ('message', 'delta'):
            container = event_data.get(container_key)
            if not container:
                continue
            rc = container.pop('reasoning_content', None) or container.pop('reasoningContent', None)
            if rc:
                reasoning_buf += rc
                modified = True

        if reasoning_buf and not injected:
            if event_data.get('delta', {}).get('type') == 'text_delta':
                injected = True
                for injected_event in _emit_thinking_blocks(reasoning_buf):
                    append_client_event(turn, {'raw': injected_event})
                    summary['client_event_count'] += 1
                    yield injected_event
                index_offset = 1
                reasoning_buf = ''

        if index_offset and 'index' in event_data:
            event_data['index'] = event_data['index'] + index_offset
            modified = True

        output = f'data: {json.dumps(event_data)}\n\n' if modified else decoded + '\n\n'
        append_client_event(turn, {'raw': output})
        summary['client_event_count'] += 1
        yield output


def _emit_thinking_blocks(text):
    """生成一组等价的 Anthropic thinking block SSE 事件。"""
    yield (
        f'event: content_block_start\n'
        f'data: {json.dumps({"type": "content_block_start", "index": 0, "content_block": {"type": "thinking", "thinking": ""}})}\n\n'
    )
    yield (
        f'event: content_block_delta\n'
        f'data: {json.dumps({"type": "content_block_delta", "index": 0, "delta": {"type": "thinking_delta", "thinking": text}})}\n\n'
    )
    yield (
        f'event: content_block_stop\n'
        f'data: {json.dumps({"type": "content_block_stop", "index": 0})}\n\n'
    )
