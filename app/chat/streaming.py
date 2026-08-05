"""上游流转 Cursor Chat SSE。"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Callable, Iterator
from typing import Any
from uuid import uuid4

import httpx

from ..errors import ApiError
from ..protocol import WireProtocol
from ..upstream import iter_sse
from .cursor import CursorAdapter
from .reasoning_display import CursorReasoningDisplay
from .rosetta import Rosetta


def sse_data(value: Any) -> str:
    data = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
    return f'data: {data}\n\n'


class StreamBridge:
    def __init__(self, rosetta: Rosetta, cursor: CursorAdapter):
        self.rosetta = rosetta
        self.cursor = cursor

    async def run(
        self,
        response: httpx.Response,
        protocol: WireProtocol,
        client_model: str,
        *,
        custom_tools: set[str] | None = None,
        reasoning_conversion: bool = False,
        on_upstream: Callable[[dict[str, Any]], None] | None = None,
        on_client: Callable[[str], None] | None = None,
        on_usage: Callable[[dict[str, Any]], None] | None = None,
    ) -> AsyncIterator[str]:
        source = self.rosetta.stream_context(protocol)
        target = self.rosetta.stream_context('chat')
        tool_indexes: dict[str, int] = {}
        usage: dict[str, Any] = {}
        pending_stream_end: dict[str, Any] | None = None
        stream_failed = False
        fallback_id = f'chatcmpl-{uuid4().hex}'
        custom_tools = custom_tools or set()
        custom_calls: set[str] = set()
        custom_arguments: dict[str, str] = {}
        reasoning_display = CursorReasoningDisplay() if reasoning_conversion else None

        try:
            async for event_type, raw in iter_sse(response):
                if on_upstream:
                    on_upstream({'type': event_type, 'data': raw})
                if raw == '[DONE]':
                    continue
                try:
                    chunk = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                if not isinstance(chunk, dict):
                    continue
                if reasoning_display is not None:
                    reasoning_display.capture_upstream(protocol, chunk)

                for event in self.rosetta.stream_from(protocol, chunk, source):
                    event_type = event.get('type')
                    if (
                        event_type == 'reasoning_delta'
                        and event.get('signature') is not None
                        and not event.get('reasoning')
                    ):
                        # Chat SSE 无法表达 provider 签名；启用转换时原值由状态载体保存。
                        continue
                    if event_type == 'stream_start':
                        event['response_id'] = event.get('response_id') or fallback_id
                        event['model'] = event.get('model') or client_model
                    elif event_type == 'tool_call_start':
                        call_id = event['tool_call_id']
                        tool_indexes.setdefault(call_id, len(tool_indexes))
                        event['tool_call_index'] = tool_indexes[call_id]
                        if (
                            protocol != 'responses'
                            and event.get('tool_name') in custom_tools
                        ):
                            custom_calls.add(call_id)
                            custom_arguments[call_id] = ''
                    elif event_type == 'tool_call_delta':
                        call_id = event['tool_call_id']
                        if call_id in tool_indexes:
                            event['tool_call_index'] = tool_indexes[call_id]
                        if call_id in custom_calls:
                            custom_arguments[call_id] += event.get('arguments_delta', '')
                            continue
                    elif event_type == 'finish' and tool_indexes:
                        event['finish_reason'] = {'reason': 'tool_calls'}
                        for call_id in custom_calls:
                            custom_event = {
                                'type': 'tool_call_delta',
                                'tool_call_id': call_id,
                                'tool_call_index': tool_indexes[call_id],
                                'arguments_delta': self.cursor.unwrap_custom_arguments(
                                    custom_arguments[call_id],
                                ),
                            }
                            for message in self._encode(
                                custom_event,
                                target,
                                client_model,
                                on_client,
                                reasoning_display,
                            ):
                                yield message
                    if event_type == 'usage':
                        incoming_usage = event.get('usage')
                        if isinstance(incoming_usage, dict):
                            _merge_usage(usage, incoming_usage)
                            if on_usage:
                                on_usage(dict(usage))
                        # 部分 Chat 上游会在每个 delta 中返回累计 usage。
                        # 这些快照若逐个交给目标转换器，会被当作增量相加。
                        continue
                    if event_type == 'stream_end':
                        # Defer termination until all cumulative usage snapshots
                        # have been collected, including snapshots sent after
                        # the provider's own end marker.
                        pending_stream_end = pending_stream_end or event
                        continue
                    for message in self._encode(
                        event,
                        target,
                        client_model,
                        on_client,
                        reasoning_display,
                    ):
                        yield message
        except (httpx.HTTPError, ApiError) as exc:
            stream_failed = True
            error = exc if isinstance(exc, ApiError) else ApiError(
                f'读取上游流失败: {exc}',
                error_type='proxy_error',
                status_code=502,
            )
            message = sse_data(error.body())
            if on_client:
                on_client(message)
            yield message
        finally:
            await response.aclose()

        if not stream_failed:
            if usage:
                normalized_usage = {
                    'type': 'usage',
                    'usage': dict(usage),
                }
                for message in self._encode(
                    normalized_usage,
                    target,
                    client_model,
                    on_client,
                    reasoning_display,
                ):
                    yield message
            final_stream_end = pending_stream_end or {'type': 'stream_end'}
            for message in self._encode(
                final_stream_end,
                target,
                client_model,
                on_client,
                reasoning_display,
            ):
                yield message

        if reasoning_display is not None:
            if closing := reasoning_display.flush_chunk(client_model):
                message = sse_data(closing)
                if on_client:
                    on_client(message)
                yield message

        done = sse_data('[DONE]')
        if on_client:
            on_client(done)
        yield done

    def _encode(
        self,
        event: dict[str, Any],
        target: Any,
        client_model: str,
        on_client: Callable[[str], None] | None,
        reasoning_display: CursorReasoningDisplay | None,
    ) -> Iterator[str]:
        for output in self.rosetta.stream_to_chat(event, target):
            chunk = self.cursor.stream_chunk(output, client_model)
            visible_chunks = (
                reasoning_display.rewrite_chunk(chunk)
                if reasoning_display is not None
                else [chunk]
            )
            for visible_chunk in visible_chunks:
                message = sse_data(visible_chunk)
                if on_client:
                    on_client(message)
                yield message


def _merge_usage(total: dict[str, Any], incoming: dict[str, Any]) -> None:
    for key in ('prompt_tokens', 'completion_tokens'):
        total[key] = max(int(total.get(key) or 0), int(incoming.get(key) or 0))
    total['total_tokens'] = total.get('prompt_tokens', 0) + total.get('completion_tokens', 0)
    for key in ('reasoning_tokens', 'cache_read_tokens', 'cache_creation_tokens'):
        if key in incoming:
            total[key] = max(int(total.get(key) or 0), int(incoming[key] or 0))
