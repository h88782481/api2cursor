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
from utils.http import build_anthropic_headers, forward_request, sse_response

logger = logging.getLogger(__name__)

bp = Blueprint('messages', __name__)


@bp.route('/v1/messages', methods=['POST'])
def messages_passthrough():
    """透传 Anthropic Messages 请求，并在必要时补齐 thinking 兼容层。"""
    payload = request.get_json(force=True)
    model = payload.get('model', 'unknown')
    is_stream = payload.get('stream', False)

    logger.info(f'[透传] model={model} 流式={is_stream}')

    url_base = settings.get_url()
    api_key = settings.get_key()
    headers = build_anthropic_headers(api_key)
    url = f'{url_base.rstrip("/")}/v1/messages'

    if not is_stream:
        resp, err = forward_request(url, headers, payload)
        if err:
            return err
        data = resp.json()
        _inject_thinking(data)
        return jsonify(data)

    # 流式透传
    def generate():
        """建立上游流式连接并逐段回传处理后的 SSE 数据。"""
        try:
            resp = req_lib.post(
                url, headers=headers, json=payload,
                timeout=Config.API_TIMEOUT, stream=True,
            )
            if resp.status_code != 200:
                body = resp.content.decode('utf-8', errors='replace')
                logger.warning(f'上游返回 {resp.status_code}: {body[:300]}')
                yield f'data: {json.dumps({"error": {"message": body, "type": "upstream_error"}})}\n\n'
                return

            yield from _process_stream(resp)
        except req_lib.RequestException as e:
            logger.error(f'请求上游失败: {e}')
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


def _process_stream(resp):
    """处理 /v1/messages 流式响应，检测并注入 thinking 事件"""
    reasoning_buf = ''
    injected = False

    for line in resp.iter_lines():
        if not line:
            continue
        decoded = line.decode('utf-8', errors='replace')

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

        # 提取 reasoning_content
        for container_key in ('message', 'delta'):
            container = event_data.get(container_key)
            if not container:
                continue
            rc = container.pop('reasoning_content', None) or container.pop('reasoningContent', None)
            if rc:
                reasoning_buf += rc
                modified = True

        # 在首个 text_delta 前注入 thinking blocks
        if reasoning_buf and not injected:
            if event_data.get('delta', {}).get('type') == 'text_delta':
                injected = True
                yield from _emit_thinking_blocks(reasoning_buf)
                reasoning_buf = ''

        yield f'data: {json.dumps(event_data)}\n\n' if modified else decoded + '\n\n'


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
