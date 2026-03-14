"""HTTP 工具 - 请求头构建、上游转发、SSE 流解析、响应构建"""

from __future__ import annotations

import json
import logging
import uuid
from typing import Any, Iterator

import requests
from flask import Response, jsonify

from config import Config

logger = logging.getLogger(__name__)


def gen_id(prefix: str = '') -> str:
    """生成带可选前缀的短随机 ID，用于请求和工具调用标识。"""
    return f'{prefix}{uuid.uuid4().hex[:24]}'


# ─── 请求头构建 ────────────────────────────────────


def build_openai_headers(api_key: str) -> dict[str, str]:
    """构建 OpenAI 兼容请求头"""
    return {
        'Authorization': f'Bearer {api_key}',
        'Content-Type': 'application/json',
    }


def build_anthropic_headers(api_key: str) -> dict[str, str]:
    """构建 Anthropic 请求头，根据密钥前缀自动选择鉴权方式"""
    headers = {
        'anthropic-version': '2023-06-01',
        'Content-Type': 'application/json',
    }
    if api_key.startswith('sk-'):
        headers['x-api-key'] = api_key
    else:
        headers['Authorization'] = f'Bearer {api_key}'
    return headers


def build_gemini_headers(api_key: str) -> dict[str, str]:
    """构建 Gemini 请求头，根据密钥前缀选择鉴权方式"""
    headers = {'Content-Type': 'application/json'}
    if api_key.startswith('AIza'):
        headers['x-goog-api-key'] = api_key
    else:
        headers['Authorization'] = f'Bearer {api_key}'
    return headers


# ─── 响应构建 ──────────────────────────────────────


def sse_response(generator):
    """将生成器包装为 SSE 流式响应"""
    return Response(
        generator,
        content_type='text/event-stream',
        headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'},
    )


def error_json(message, error_type='proxy_error', status=502):
    """构造统一的 JSON 错误响应，供非流式链路直接返回。"""
    return jsonify({'error': {'message': str(message), 'type': error_type}}), status


# ─── 上游请求转发 ──────────────────────────────────


def forward_request(url, headers, payload, stream=False):
    """转发请求到上游 API

    返回值:
        成功: (response, None)
        失败（流式）: (None, error_body_str)
        失败（非流式）: (None, Flask Response)
    """
    try:
        resp = requests.post(
            url, headers=headers, json=payload,
            timeout=Config.API_TIMEOUT, stream=stream,
        )
        if resp.status_code != 200:
            body = resp.content.decode('utf-8', errors='replace')
            logger.warning(f'上游返回 {resp.status_code}: {body[:300]}')
            if stream:
                return None, f'上游错误 {resp.status_code}: {body}'
            return None, Response(
                resp.content, status=resp.status_code,
                content_type=resp.headers.get('Content-Type', 'application/json'),
            )
        return resp, None
    except requests.RequestException as e:
        logger.error(f'请求上游失败: {e}')
        if stream:
            return None, str(e)
        return None, error_json(str(e))


# ─── SSE 流解析 ───────────────────────────────────


def iter_openai_sse(response) -> Iterator[dict[str, Any] | None]:
    """解析 OpenAI SSE 流，yield chunk 字典；yield None 表示 [DONE]"""
    for line in response.iter_lines():
        if not line:
            continue
        decoded = line.decode('utf-8', errors='replace')
        if not decoded.startswith('data:'):
            continue
        data_str = decoded[5:].strip()
        if data_str == '[DONE]':
            yield None
            return
        try:
            yield json.loads(data_str)
        except json.JSONDecodeError:
            continue


def iter_anthropic_sse(response) -> Iterator[tuple[str, dict[str, Any]]]:
    """解析 Anthropic SSE 流，yield (event_type, data_dict) 元组"""
    yield from _iter_event_sse(response)


def iter_responses_sse(response) -> Iterator[tuple[str, dict[str, Any]]]:
    """解析 OpenAI Responses SSE 流，yield (event_type, data_dict) 元组"""
    yield from _iter_event_sse(response)


def iter_gemini_sse(response) -> Iterator[dict[str, Any]]:
    """解析 Gemini SSE 流，yield 完整的 GenerateContentResponse 字典。

    Gemini 流式使用 ?alt=sse，每个 data: 行是一个完整的 JSON 响应。
    """
    for line in response.iter_lines():
        if not line:
            continue
        decoded = line.decode('utf-8', errors='replace')
        if not decoded.startswith('data:'):
            continue
        data_str = decoded[5:].strip()
        if not data_str:
            continue
        try:
            yield json.loads(data_str)
        except json.JSONDecodeError:
            continue


def _iter_event_sse(response) -> Iterator[tuple[str, dict[str, Any]]]:
    """解析带 event/data 的通用 SSE 流。

    SSE 规范中空行是事件分隔符，遇到空行时重置 event_type，
    避免前一个事件的类型泄漏到下一个事件。
    """
    event_type = ''
    for line in response.iter_lines():
        if not line:
            event_type = ''
            continue
        decoded = line.decode('utf-8', errors='replace')
        if decoded.startswith('event:'):
            event_type = decoded[6:].strip()
        elif decoded.startswith('data:'):
            data_str = decoded[5:].strip()
            if not data_str:
                continue
            try:
                yield event_type, json.loads(data_str)
            except json.JSONDecodeError:
                continue
