"""上游 HTTP 与 SSE 传输。"""

from __future__ import annotations

from collections.abc import AsyncIterator

import httpx


class UpstreamClient:
    def __init__(self, client: httpx.AsyncClient):
        self.client = client

    async def post(
        self,
        url: str,
        headers: dict[str, str],
        body: dict,
    ) -> httpx.Response:
        return await self.client.post(
            url,
            headers=httpx.Headers(headers, encoding='latin-1'),
            json=body,
        )

    async def stream(
        self,
        url: str,
        headers: dict[str, str],
        body: dict,
    ) -> httpx.Response:
        request = self.client.build_request(
            'POST',
            url,
            headers=httpx.Headers(headers, encoding='latin-1'),
            json=body,
        )
        return await self.client.send(request, stream=True)


async def iter_sse(response: httpx.Response) -> AsyncIterator[tuple[str, str]]:
    event_type = ''
    data: list[str] = []

    async for line in response.aiter_lines():
        if not line:
            if data:
                yield event_type, '\n'.join(data)
            event_type = ''
            data = []
        elif line.startswith('event:'):
            event_type = line[6:].strip()
        elif line.startswith('data:'):
            data.append(line[5:].lstrip())

    if data:
        yield event_type, '\n'.join(data)
