"""单一 Chat 请求用例。"""

from __future__ import annotations

import logging
from typing import Any

import httpx
from fastapi.responses import JSONResponse, Response, StreamingResponse

from ..errors import ApiError
from ..observability import RequestLogger, UsageTracker
from ..settings import RouteResolver
from ..upstream import UpstreamClient
from .builder import PreparedRequest, UpstreamRequestBuilder
from .cursor import CursorAdapter
from .exchange import Exchange
from .rosetta import Rosetta
from .streaming import StreamBridge

logger = logging.getLogger(__name__)
_SSE_HEADERS = {'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'}


class ChatGateway:
    def __init__(
        self,
        resolver: RouteResolver,
        upstream: UpstreamClient,
        rosetta: Rosetta,
        request_log: RequestLogger,
        usage: UsageTracker,
    ):
        self.upstream = upstream
        self.rosetta = rosetta
        self.cursor = CursorAdapter(rosetta)
        self.builder = UpstreamRequestBuilder(resolver, self.cursor, rosetta)
        self.streams = StreamBridge(rosetta, self.cursor)
        self.request_log = request_log
        self.usage = usage

    async def handle(
        self,
        payload: dict[str, Any],
        request_headers: dict[str, Any],
    ) -> Response:
        prepared = self.builder.build(payload)
        exchange = self._start_exchange(payload, request_headers, prepared)
        if exchange.stream:
            return await self._stream(exchange, prepared.body, prepared.headers, prepared.url)
        return await self._complete(exchange, prepared.body, prepared.headers, prepared.url)

    def _start_exchange(
        self,
        payload: dict[str, Any],
        request_headers: dict[str, Any],
        prepared: PreparedRequest,
    ) -> Exchange:
        exchange = Exchange(
            route=prepared.route,
            stream=prepared.stream,
            custom_tools=prepared.custom_tools,
        )
        exchange.log = self.request_log.start(
            payload=payload,
            request_headers=request_headers,
            upstream_protocol=prepared.route.protocol,
            upstream_model=prepared.route.upstream_model,
            target_url=prepared.url,
            dialect=prepared.dialect,
        )
        self.request_log.warnings(exchange.log, prepared.warnings)
        self.request_log.upstream_request(exchange.log, prepared.body, prepared.headers)
        self.request_log.debug(
            'chat',
            f'{prepared.route.client_model} → '
            f'{prepared.route.protocol}:{prepared.route.upstream_model}',
        )
        return exchange

    async def _complete(
        self,
        exchange: Exchange,
        body: dict[str, Any],
        headers: dict[str, str],
        url: str,
    ) -> Response:
        try:
            response = await self.upstream.post(url, headers, body)
        except httpx.HTTPError as exc:
            return self._proxy_error(exchange, exc)

        if response.status_code >= 400:
            return await self._upstream_failure(exchange, response)

        try:
            upstream_body = response.json()
            self.request_log.upstream_response(exchange.log, upstream_body)
            ir_response = self.rosetta.response_from(exchange.route.protocol, upstream_body)
            usage = ir_response.get('usage')
            result = self.cursor.response(
                ir_response,
                exchange.route.client_model,
                exchange.custom_tools,
            )
        except (ValueError, ApiError) as exc:
            error = exc if isinstance(exc, ApiError) else ApiError(
                '上游返回了无效 JSON',
                error_type='upstream_response_error',
                status_code=502,
            )
            self.request_log.error(exchange.log, {
                'stage': 'response_conversion',
                'message': str(exc),
            })
            self.request_log.finish(exchange.log)
            return JSONResponse(error.body(), status_code=error.status_code)

        self.usage.record(exchange.route.client_model, usage)
        self.request_log.client_response(exchange.log, result)
        self.request_log.finish(exchange.log, dict(usage) if usage else None)
        return JSONResponse(result)

    async def _stream(
        self,
        exchange: Exchange,
        body: dict[str, Any],
        headers: dict[str, str],
        url: str,
    ) -> Response:
        try:
            response = await self.upstream.stream(url, headers, body)
        except httpx.HTTPError as exc:
            return self._proxy_error(exchange, exc)

        if response.status_code >= 400:
            return await self._upstream_failure(exchange, response)

        usage: dict[str, Any] = {}

        async def generate():
            try:
                async for message in self.streams.run(
                    response,
                    exchange.route.protocol,
                    exchange.route.client_model,
                    custom_tools=exchange.custom_tools,
                    on_upstream=lambda event: self.request_log.upstream_event(exchange.log, event),
                    on_client=lambda event: self.request_log.client_event(exchange.log, event),
                    on_usage=lambda value: usage.update(value),
                ):
                    yield message
            finally:
                self.usage.record(exchange.route.client_model, usage or None)
                self.request_log.finish(exchange.log, usage or None)

        return StreamingResponse(generate(), media_type='text/event-stream', headers=_SSE_HEADERS)

    async def _upstream_failure(
        self,
        exchange: Exchange,
        response: httpx.Response,
    ) -> Response:
        content = await response.aread()
        await response.aclose()
        self.request_log.error(exchange.log, {
            'stage': 'upstream_status',
            'status_code': response.status_code,
            'message': content.decode(errors='replace')[:2000],
        })
        self.request_log.finish(exchange.log)
        return Response(
            content,
            status_code=response.status_code,
            media_type=response.headers.get('content-type', 'application/json'),
        )

    def _proxy_error(self, exchange: Exchange, exc: Exception) -> JSONResponse:
        logger.error('请求上游失败: %s', exc)
        error = ApiError(
            f'请求上游失败: {exc}',
            error_type='proxy_error',
            status_code=502,
        )
        self.request_log.error(exchange.log, {'stage': 'forward_request', 'message': str(exc)})
        self.request_log.finish(exchange.log)
        return JSONResponse(error.body(), status_code=error.status_code)
