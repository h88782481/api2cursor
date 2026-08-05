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
        on_upstream: Callable[[dict[str, Any]], None] | None = None,
        on_client: Callable[[str], None] | None = None,
        on_usage: Callable[[dict[str, Any]], None] | None = None,
    ) -> AsyncIterator[str]:
        source = self.rosetta.stream_context(protocol)
        target = self.rosetta.stream_context('chat')
        tool_indexes: dict[str, int] = {}
        usage: dict[str, Any] = {}
        fallback_id = f'chatcmpl-{uuid4().hex}'
        custom_tools = custom_tools or set()
        custom_calls: set[str] = set()
        custom_arguments: dict[str, str] = {}
        reasoning_display = CursorReasoningDisplay()

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

                for event in self.rosetta.stream_from(protocol, chunk, source):
                    event_type = event.get('type')
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
                    if event.get('type') == 'usage':
                        _merge_usage(usage, event['usage'])
                        if on_usage:
                            on_usage(usage)
                    for message in self._encode(
                        event,
                        target,
                        client_model,
                        on_client,
                        reasoning_display,
                    ):
                        yield message
        except (httpx.HTTPError, ApiError) as exc:
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
        reasoning_display: CursorReasoningDisplay,
    ) -> Iterator[str]:
        for output in self.rosetta.stream_to_chat(event, target):
            chunk = self.cursor.stream_chunk(output, client_model)
            for visible_chunk in reasoning_display.rewrite_chunk(chunk):
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
