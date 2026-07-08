"""协议编解码器接口

每种协议实现一个 Codec，分为两个方向：
- 客户端方向（Cursor 一侧）: parse_request / build_response / stream_encoder / error_event
- 上游方向（中转站一侧）: build_request / parse_response / stream_decoder / upstream_url / build_headers

chat / responses / messages 三种协议实现双向，gemini 只实现上游方向。
"""

from __future__ import annotations

import json
from typing import Any

from ..core.ir import IRRequest, IRResponse, StreamEvent


def sse_data(data: Any) -> str:
    """构造仅包含 data 的 SSE 消息。"""
    payload = data if isinstance(data, str) else json.dumps(data, ensure_ascii=False)
    return f'data: {payload}\n\n'


def sse_event(event_type: str, data: Any) -> str:
    """构造带 event 名称的 SSE 消息。"""
    payload = data if isinstance(data, str) else json.dumps(data, ensure_ascii=False)
    return f'event: {event_type}\ndata: {payload}\n\n'


def parse_json(data: str) -> dict[str, Any] | None:
    """宽松解析 SSE data 段，失败返回 None。"""
    try:
        parsed = json.loads(data)
    except (json.JSONDecodeError, ValueError):
        return None
    return parsed if isinstance(parsed, dict) else None


def strip_version_suffix(base_url: str) -> str:
    """剥离上游地址尾部的版本段，容忍用户把 https://relay.com/v1 填进来。

    各编解码器总是重新拼接自己需要的规范版本段（/v1 或 /v1beta），
    因此这里的剥离是无损的：https://relay.com/openai/v1 → https://relay.com/openai。
    """
    base = base_url.rstrip('/')
    for suffix in ('/v1beta', '/v1'):
        if base.endswith(suffix):
            return base[:-len(suffix)]
    return base


class StreamDecoder:
    """上游 SSE → IR 事件。"""

    def decode(self, event_type: str, data: str) -> list[StreamEvent]:
        """处理一条上游 SSE 消息，产出零或多个 IR 事件。"""
        raise NotImplementedError

    def finalize(self) -> list[StreamEvent]:
        """上游流结束时调用，补齐未发出的收尾事件（如 StreamEnd）。"""
        return []


class StreamEncoder:
    """IR 事件 → 客户端 SSE 字符串。"""

    def start(self) -> list[str]:
        """流开始前需要先发给客户端的事件（如 Responses 的 response.created）。"""
        return []

    def encode(self, event: StreamEvent) -> list[str]:
        """处理一个 IR 事件，产出零或多条 SSE 消息。"""
        raise NotImplementedError

    def finalize(self) -> list[str]:
        """流结束时调用，补齐协议要求的终止消息（如 chat 的 [DONE]）。"""
        return []


class Codec:
    """单个协议的编解码器。"""

    format: str = ''

    # ─── 客户端方向 ──────────────────────────────

    def parse_request(self, payload: dict[str, Any]) -> IRRequest:
        raise NotImplementedError(f'{self.format} 不支持作为客户端格式')

    def build_response(self, response: IRResponse, client_model: str) -> dict[str, Any]:
        raise NotImplementedError(f'{self.format} 不支持作为客户端格式')

    def stream_encoder(self, client_model: str) -> StreamEncoder:
        raise NotImplementedError(f'{self.format} 不支持作为客户端格式')

    def error_event(self, message: str, error_type: str = 'upstream_error') -> str:
        """构造客户端流式链路使用的错误消息。"""
        return sse_data({'error': {'message': message, 'type': error_type}})

    # ─── 上游方向 ────────────────────────────────

    def build_request(self, request: IRRequest, upstream_model: str) -> dict[str, Any]:
        raise NotImplementedError

    def parse_response(self, payload: dict[str, Any]) -> IRResponse:
        raise NotImplementedError

    def stream_decoder(self) -> StreamDecoder:
        raise NotImplementedError

    def upstream_url(self, base_url: str, model: str, stream: bool) -> str:
        raise NotImplementedError

    def build_headers(self, api_key: str) -> dict[str, str]:
        headers = {'Content-Type': 'application/json'}
        if api_key:
            headers['Authorization'] = f'Bearer {api_key}'
        return headers
