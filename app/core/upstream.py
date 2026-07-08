"""上游请求转发与 SSE 流解析"""

from __future__ import annotations

from typing import AsyncIterator

import httpx


async def post_json(
    client: httpx.AsyncClient,
    url: str,
    headers: dict[str, str],
    payload: dict,
) -> httpx.Response:
    """发送非流式上游请求。"""
    return await client.post(url, json=payload, headers=headers)


async def post_stream(
    client: httpx.AsyncClient,
    url: str,
    headers: dict[str, str],
    payload: dict,
) -> httpx.Response:
    """发送流式上游请求，返回未读取 body 的响应对象。"""
    request = client.build_request('POST', url, json=payload, headers=headers)
    return await client.send(request, stream=True)


async def iter_sse(response: httpx.Response) -> AsyncIterator[tuple[str, str]]:
    """解析 SSE 流，逐条产出 (event_type, data)。

    - 空行是事件分隔符，遇到时重置 event_type，避免类型泄漏到下一个事件
    - data 为原始字符串（可能是 JSON，也可能是 [DONE] 之类的哨兵值）
    """
    event_type = ''
    async for line in response.aiter_lines():
        if not line:
            event_type = ''
            continue
        if line.startswith('event:'):
            event_type = line[6:].strip()
        elif line.startswith('data:'):
            data = line[5:].strip()
            if data:
                yield event_type, data
