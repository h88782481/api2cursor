"""OpenAI Responses API 协议编解码器

客户端方向：解析 Cursor 发来的 /v1/responses 请求（含 BYOK bug 场景下发到
/v1/chat/completions 的 Responses 风格请求体）/ 把 IR 编码回 Responses 响应与事件流。
上游方向：把 IR 构建为发往中转站 /v1/responses 的请求 / 解析其响应与 SSE 流。

流式编码按真实 OpenAI Responses 事件结构输出（response.created →
output_item.added → *.delta → *.done → response.completed），并维护
reasoning / message / function_call 三类输出项的生命周期。
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from typing import Any

from ..compat.tools import dump_arguments, parse_tool_choice, parse_tool_definitions
from ..core.ir import (
    Block,
    IRMessage,
    IRRequest,
    IRResponse,
    IRUsage,
    ImageBlock,
    StreamEnd,
    StreamError,
    StreamEvent,
    StreamStart,
    TextBlock,
    TextDelta,
    ThinkingBlock,
    ThinkingDelta,
    ToolCallBlock,
    ToolCallDelta,
    ToolCallEnd,
    ToolCallStart,
    ToolResultBlock,
    gen_id,
)
from .base import Codec, StreamDecoder, StreamEncoder, parse_json, sse_event, strip_version_suffix


class ResponsesCodec(Codec):
    format = 'responses'

    # ═══════════════════════════════════════════
    #  客户端方向
    # ═══════════════════════════════════════════

    def parse_request(self, payload: dict[str, Any]) -> IRRequest:
        request = IRRequest(
            model=payload.get('model', ''),
            stream=bool(payload.get('stream', False)),
            system=str(payload.get('instructions') or ''),
        )

        input_data = payload.get('input', [])
        if isinstance(input_data, str):
            if input_data:
                request.messages.append(IRMessage('user', [TextBlock(text=input_data)]))
        elif isinstance(input_data, list):
            _parse_input_items(input_data, request.messages)

        request.tools = parse_tool_definitions(payload.get('tools'))
        request.tool_choice = parse_tool_choice(payload.get('tool_choice'))

        if isinstance(payload.get('max_output_tokens'), int):
            request.max_tokens = payload['max_output_tokens']
        if isinstance(payload.get('temperature'), (int, float)):
            request.temperature = payload['temperature']
        if isinstance(payload.get('top_p'), (int, float)):
            request.top_p = payload['top_p']
        return request

    def build_response(self, response: IRResponse, client_model: str) -> dict[str, Any]:
        output: list[dict[str, Any]] = []
        thinking = response.thinking()
        if thinking:
            output.append(_reasoning_item(gen_id('rs_'), thinking))
        text = response.text()
        if text:
            output.append(_message_item(gen_id('msg_'), text))
        for block in response.tool_calls():
            output.append(_function_call_item(gen_id('fc_'), block.id, block.name, block.arguments))

        return {
            'id': response.id or gen_id('resp_'),
            'object': 'response',
            'created_at': int(time.time()),
            'status': 'incomplete' if response.finish_reason == 'length' else 'completed',
            'model': client_model,
            'output': output,
            'usage': _build_usage(response.usage),
        }

    def stream_encoder(self, client_model: str) -> StreamEncoder:
        return ResponsesStreamEncoder(client_model)

    def error_event(self, message: str, error_type: str = 'upstream_error') -> str:
        return sse_event('error', {'type': 'error', 'error': {'type': error_type, 'message': message}})

    # ═══════════════════════════════════════════
    #  上游方向
    # ═══════════════════════════════════════════

    def build_request(self, request: IRRequest, upstream_model: str) -> dict[str, Any]:
        input_items: list[dict[str, Any]] = []
        for message in request.messages:
            _append_input_items(message, input_items)

        payload: dict[str, Any] = {
            'model': upstream_model,
            'input': input_items,
            'stream': request.stream,
        }
        if request.system:
            payload['instructions'] = request.system
        if request.tools:
            payload['tools'] = [
                {
                    'type': 'function',
                    'name': t.name,
                    'description': t.description,
                    'parameters': t.parameters,
                }
                for t in request.tools
            ]
        tool_choice = _build_tool_choice(request.tool_choice)
        if tool_choice is not None:
            payload['tool_choice'] = tool_choice
        if request.max_tokens:
            payload['max_output_tokens'] = request.max_tokens
        if request.temperature is not None:
            payload['temperature'] = request.temperature
        if request.top_p is not None:
            payload['top_p'] = request.top_p

        ensure_prompt_cache_key(payload)
        return payload

    def parse_response(self, payload: dict[str, Any]) -> IRResponse:
        blocks: list[Block] = []
        for item in payload.get('output') or []:
            if not isinstance(item, dict):
                continue
            item_type = item.get('type', '')
            if item_type == 'reasoning':
                text = _extract_reasoning_text(item)
                if text:
                    blocks.append(ThinkingBlock(text=text))
            elif item_type == 'message':
                text = _extract_output_text(item.get('content', []))
                if text:
                    blocks.append(TextBlock(text=text))
            elif item_type in ('function_call', 'custom_tool_call'):
                blocks.append(_parse_tool_call_item(item))

        has_tools = any(isinstance(b, ToolCallBlock) for b in blocks)
        if has_tools:
            finish_reason = 'tool_calls'
        elif payload.get('status') == 'incomplete':
            finish_reason = 'length'
        else:
            finish_reason = 'stop'

        return IRResponse(
            id=payload.get('id') or gen_id('resp_'),
            model=payload.get('model', ''),
            blocks=blocks,
            finish_reason=finish_reason,
            usage=parse_responses_usage(payload.get('usage')),
        )

    def stream_decoder(self) -> StreamDecoder:
        return ResponsesStreamDecoder()

    def upstream_url(self, base_url: str, model: str, stream: bool) -> str:
        return f'{strip_version_suffix(base_url)}/v1/responses'


def ensure_prompt_cache_key(payload: dict[str, Any]) -> dict[str, Any]:
    """确保 Responses 请求携带 prompt_cache_key 以启用上游提示缓存。

    部分中转站对原生 /v1/responses 请求不会自动生成 prompt_cache_key，
    导致提示缓存无法命中。这里根据模型名 + instructions 生成稳定的 cache key。
    """
    if payload.get('prompt_cache_key'):
        return payload
    seed = f"{payload.get('model', '')}|{payload.get('instructions', '')}"
    payload['prompt_cache_key'] = hashlib.sha256(seed.encode()).hexdigest()[:32]
    return payload


# ═══════════════════════════════════════════════════════════
#  流式解码器: Responses SSE → IR 事件
# ═══════════════════════════════════════════════════════════


class ResponsesStreamDecoder(StreamDecoder):
    def __init__(self):
        self._ended = False
        # 上游 output_index → 工具信息
        self._tools: dict[int, dict[str, Any]] = {}
        self._tool_count = 0

    def decode(self, event_type: str, data: str) -> list[StreamEvent]:
        payload = parse_json(data)
        if payload is None:
            return []
        event_type = event_type or payload.get('type', '')

        if event_type == 'response.created':
            response = payload.get('response') or {}
            return [StreamStart(id=response.get('id', ''), model=response.get('model', ''))]

        if event_type == 'response.output_item.added':
            return self._handle_item_added(payload)

        if event_type == 'response.output_text.delta':
            delta = payload.get('delta', '')
            return [TextDelta(delta)] if delta else []

        if event_type == 'response.refusal.delta':
            delta = payload.get('delta', '')
            return [TextDelta(delta)] if delta else []

        if event_type in ('response.reasoning_summary_text.delta', 'response.reasoning_text.delta'):
            delta = payload.get('delta', '')
            return [ThinkingDelta(delta)] if delta else []

        if event_type in ('response.function_call_arguments.delta', 'response.custom_tool_call_input.delta'):
            return self._handle_arguments_delta(payload)

        if event_type in ('response.function_call_arguments.done', 'response.custom_tool_call_input.done'):
            return self._handle_arguments_done(payload)

        if event_type in ('response.completed', 'response.incomplete'):
            return self._handle_completed(payload, incomplete=event_type == 'response.incomplete')

        if event_type in ('response.failed', 'error'):
            error = payload.get('error') or payload.get('response', {}).get('error') or {}
            message = error.get('message') if isinstance(error, dict) else str(error)
            return [StreamError(message=str(message or '上游返回失败'), error_type='upstream_error')]

        return []

    def finalize(self) -> list[StreamEvent]:
        if self._ended:
            return []
        self._ended = True
        events: list[StreamEvent] = [
            ToolCallEnd(
                index=tool['index'],
                id=tool['call_id'],
                name=tool['name'],
                arguments=tool['args'],
            )
            for tool in self._tools.values()
        ]
        self._tools.clear()
        events.append(StreamEnd(
            finish_reason='tool_calls' if self._tool_count else 'stop',
        ))
        return events

    def _handle_item_added(self, payload: dict[str, Any]) -> list[StreamEvent]:
        item = payload.get('item') or {}
        item_type = item.get('type', '')
        if item_type not in ('function_call', 'custom_tool_call'):
            return []

        output_index = payload.get('output_index', 0)
        our_index = self._tool_count
        self._tool_count += 1

        if item_type == 'custom_tool_call':
            call_id = item.get('call_id') or item.get('id') or gen_id('call_')
            name = item.get('name') or item.get('namespace') or 'custom_tool'
            initial_args = item.get('input', '') or ''
        else:
            call_id = item.get('call_id') or item.get('id') or gen_id('call_')
            name = item.get('name', '')
            initial_args = item.get('arguments', '') or ''

        self._tools[output_index] = {
            'index': our_index,
            'call_id': call_id,
            'name': name,
            'args': initial_args,
        }
        events: list[StreamEvent] = [ToolCallStart(index=our_index, id=call_id, name=name)]
        if initial_args:
            events.append(ToolCallDelta(index=our_index, arguments=initial_args))
        return events

    def _handle_arguments_delta(self, payload: dict[str, Any]) -> list[StreamEvent]:
        delta = payload.get('delta', '')
        if not delta:
            return []
        tool = self._find_tool(payload.get('output_index'))
        if tool is None:
            return []
        tool['args'] += delta
        return [ToolCallDelta(index=tool['index'], arguments=delta)]

    def _handle_arguments_done(self, payload: dict[str, Any]) -> list[StreamEvent]:
        output_index = payload.get('output_index')
        tool = None
        if isinstance(output_index, int):
            tool = self._tools.pop(output_index, None)
        elif self._tools:
            tool = self._tools.pop(max(self._tools), None)
        if tool is None:
            return []
        arguments = payload.get('arguments') or payload.get('input') or tool['args']
        name = payload.get('name') or tool['name']
        return [ToolCallEnd(
            index=tool['index'],
            id=tool['call_id'],
            name=name,
            arguments=arguments,
        )]

    def _handle_completed(self, payload: dict[str, Any], *, incomplete: bool) -> list[StreamEvent]:
        if self._ended:
            return []
        self._ended = True

        response = payload.get('response') or {}
        events: list[StreamEvent] = [
            ToolCallEnd(
                index=tool['index'],
                id=tool['call_id'],
                name=tool['name'],
                arguments=tool['args'],
            )
            for tool in self._tools.values()
        ]
        self._tools.clear()

        has_tools = self._tool_count > 0 or any(
            isinstance(item, dict) and item.get('type') in ('function_call', 'custom_tool_call')
            for item in response.get('output') or []
        )
        if has_tools:
            finish_reason = 'tool_calls'
        elif incomplete or response.get('status') == 'incomplete':
            finish_reason = 'length'
        else:
            finish_reason = 'stop'

        events.append(StreamEnd(
            finish_reason=finish_reason,
            usage=parse_responses_usage(response.get('usage') or payload.get('usage')),
        ))
        return events

    def _find_tool(self, output_index: Any) -> dict[str, Any] | None:
        if isinstance(output_index, int) and output_index in self._tools:
            return self._tools[output_index]
        if self._tools:
            return self._tools[max(self._tools)]
        return None


# ═══════════════════════════════════════════════════════════
#  流式编码器: IR 事件 → Responses SSE
# ═══════════════════════════════════════════════════════════


@dataclass
class _ToolItem:
    """function_call 输出项的流式缓冲。"""

    output_index: int
    fc_id: str
    call_id: str
    name: str
    args: str = ''
    closed: bool = False


class ResponsesStreamEncoder(StreamEncoder):
    """维护 reasoning / message / function_call 输出项生命周期的状态机。"""

    def __init__(self, client_model: str):
        self._model = client_model
        self._resp_id = gen_id('resp_')
        self._created_at = int(time.time())

        self._reasoning_buf = ''
        self._reasoning_open = False
        self._reasoning_closed = False
        self._rs_id = gen_id('rs_')
        self._rs_index: int | None = None

        self._text_buf = ''
        self._text_open = False
        self._text_closed = False
        self._msg_id = gen_id('msg_')
        self._msg_index: int | None = None

        self._tools: dict[int, _ToolItem] = {}  # IR 工具 index → 输出项缓冲
        self._output_items: list[dict[str, Any]] = []
        self._next_output_index = 0
        self._finished = False
        self._sequence_number = 0

    def _emit(self, event_type: str, data: dict[str, Any]) -> str:
        """按 Responses 规范为每个事件附加递增的 sequence_number。"""
        data['sequence_number'] = self._sequence_number
        self._sequence_number += 1
        return sse_event(event_type, data)

    def start(self) -> list[str]:
        return [self._emit('response.created', {
            'type': 'response.created',
            'response': {
                'id': self._resp_id,
                'object': 'response',
                'created_at': self._created_at,
                'status': 'in_progress',
                'model': self._model,
                'output': [],
            },
        })]

    def encode(self, event: StreamEvent) -> list[str]:
        if isinstance(event, StreamStart):
            return []
        if isinstance(event, ThinkingDelta):
            out = self._ensure_reasoning()
            self._reasoning_buf += event.text
            out.append(self._emit('response.reasoning_summary_text.delta', {
                'type': 'response.reasoning_summary_text.delta',
                'item_id': self._rs_id,
                'output_index': self._rs_index,
                'summary_index': 0,
                'delta': event.text,
            }))
            return out
        if isinstance(event, TextDelta):
            out = self._ensure_text()
            self._text_buf += event.text
            out.append(self._emit('response.output_text.delta', {
                'type': 'response.output_text.delta',
                'item_id': self._msg_id,
                'output_index': self._msg_index,
                'content_index': 0,
                'delta': event.text,
            }))
            return out
        if isinstance(event, ToolCallStart):
            return self._start_tool(event)
        if isinstance(event, ToolCallDelta):
            tool = self._tools.get(event.index)
            if tool is None or tool.closed:
                return []
            tool.args += event.arguments
            return [self._emit('response.function_call_arguments.delta', {
                'type': 'response.function_call_arguments.delta',
                'item_id': tool.fc_id,
                'output_index': tool.output_index,
                'delta': event.arguments,
            })]
        if isinstance(event, ToolCallEnd):
            tool = self._tools.get(event.index)
            if tool is None:
                return []
            if event.arguments:
                tool.args = event.arguments
            if event.name:
                tool.name = event.name
            return self._close_tool(tool)
        if isinstance(event, StreamEnd):
            return self._finish(event.finish_reason, event.usage)
        if isinstance(event, StreamError):
            return [self._emit('error', {
                'type': 'error',
                'error': {'type': event.error_type, 'message': event.message},
            })]
        return []

    def finalize(self) -> list[str]:
        if self._finished:
            return []
        return self._finish('stop', None)

    # ─── 输出项生命周期 ──────────────────────────

    def _ensure_reasoning(self) -> list[str]:
        if self._reasoning_open:
            return []
        self._reasoning_open = True
        self._rs_index = self._alloc_index()
        return [self._emit('response.output_item.added', {
            'type': 'response.output_item.added',
            'output_index': self._rs_index,
            'item': {'id': self._rs_id, 'type': 'reasoning', 'summary': []},
        })]

    def _close_reasoning(self) -> list[str]:
        if not self._reasoning_open or self._reasoning_closed:
            return []
        self._reasoning_closed = True
        item = _reasoning_item(self._rs_id, self._reasoning_buf)
        self._output_items.append(item)
        return [
            self._emit('response.reasoning_summary_text.done', {
                'type': 'response.reasoning_summary_text.done',
                'item_id': self._rs_id,
                'output_index': self._rs_index,
                'summary_index': 0,
                'text': self._reasoning_buf,
            }),
            self._emit('response.output_item.done', {
                'type': 'response.output_item.done',
                'output_index': self._rs_index,
                'item': item,
            }),
        ]

    def _ensure_text(self) -> list[str]:
        out = self._close_reasoning()
        if self._text_open:
            return out
        self._text_open = True
        self._msg_index = self._alloc_index()
        out.append(self._emit('response.output_item.added', {
            'type': 'response.output_item.added',
            'output_index': self._msg_index,
            'item': {
                'id': self._msg_id,
                'type': 'message',
                'status': 'in_progress',
                'role': 'assistant',
                'content': [],
            },
        }))
        out.append(self._emit('response.content_part.added', {
            'type': 'response.content_part.added',
            'item_id': self._msg_id,
            'output_index': self._msg_index,
            'content_index': 0,
            'part': {'type': 'output_text', 'annotations': [], 'text': ''},
        }))
        return out

    def _close_text(self) -> list[str]:
        if not self._text_open or self._text_closed:
            return []
        self._text_closed = True
        item = _message_item(self._msg_id, self._text_buf)
        self._output_items.append(item)
        return [
            self._emit('response.output_text.done', {
                'type': 'response.output_text.done',
                'item_id': self._msg_id,
                'output_index': self._msg_index,
                'content_index': 0,
                'text': self._text_buf,
            }),
            self._emit('response.output_item.done', {
                'type': 'response.output_item.done',
                'output_index': self._msg_index,
                'item': item,
            }),
        ]

    def _start_tool(self, event: ToolCallStart) -> list[str]:
        out = self._close_reasoning() + self._close_text()
        tool = _ToolItem(
            output_index=self._alloc_index(),
            fc_id=gen_id('fc_'),
            call_id=event.id or gen_id('call_'),
            name=event.name,
        )
        self._tools[event.index] = tool
        out.append(self._emit('response.output_item.added', {
            'type': 'response.output_item.added',
            'output_index': tool.output_index,
            'item': {
                'id': tool.fc_id,
                'type': 'function_call',
                'status': 'in_progress',
                'call_id': tool.call_id,
                'name': tool.name,
                'arguments': '',
            },
        }))
        return out

    def _close_tool(self, tool: _ToolItem) -> list[str]:
        if tool.closed:
            return []
        tool.closed = True
        item = _function_call_item(tool.fc_id, tool.call_id, tool.name, tool.args)
        self._output_items.append(item)
        return [
            self._emit('response.function_call_arguments.done', {
                'type': 'response.function_call_arguments.done',
                'item_id': tool.fc_id,
                'output_index': tool.output_index,
                'name': tool.name,
                'arguments': tool.args,
            }),
            self._emit('response.output_item.done', {
                'type': 'response.output_item.done',
                'output_index': tool.output_index,
                'item': item,
            }),
        ]

    def _finish(self, finish_reason: str, usage: IRUsage | None) -> list[str]:
        if self._finished:
            return []
        self._finished = True

        out = self._close_reasoning() + self._close_text()
        for index in sorted(self._tools):
            out.extend(self._close_tool(self._tools[index]))

        out.append(self._emit('response.completed', {
            'type': 'response.completed',
            'response': {
                'id': self._resp_id,
                'object': 'response',
                'created_at': self._created_at,
                'status': 'incomplete' if finish_reason == 'length' else 'completed',
                'model': self._model,
                'output': self._output_items,
                'usage': _build_usage(usage or IRUsage()),
            },
        }))
        return out

    def _alloc_index(self) -> int:
        index = self._next_output_index
        self._next_output_index += 1
        return index


# ═══════════════════════════════════════════════════════════
#  请求辅助
# ═══════════════════════════════════════════════════════════


def _parse_input_items(items: list[Any], messages: list[IRMessage]) -> None:
    """将 Responses `input` 数组解析为 IR 消息列表。

    reasoning 项先挂起，遇到相邻的 assistant 消息或 function_call 时归属过去，
    与 Cursor 发送的 `reasoning → message → function_call` 顺序保持一致。
    """
    pending_thinking: str | None = None

    for item in items:
        if isinstance(item, str):
            messages.append(IRMessage('user', [TextBlock(text=item)]))
            continue
        if not isinstance(item, dict):
            continue

        item_type = item.get('type', '')
        role = item.get('role', '')

        if item_type == 'reasoning':
            pending_thinking = _extract_reasoning_text(item)
            continue

        if item_type == 'function_call':
            message = _last_assistant_or_new(messages)
            if pending_thinking and not message.has_thinking():
                message.blocks.insert(0, ThinkingBlock(text=pending_thinking))
                pending_thinking = None
            message.blocks.append(ToolCallBlock(
                id=item.get('call_id') or gen_id('call_'),
                name=item.get('name', ''),
                arguments=dump_arguments(item.get('arguments', '{}')),
            ))
            continue

        if item_type == 'function_call_output':
            output = item.get('output', '')
            messages.append(IRMessage('user', [ToolResultBlock(
                call_id=item.get('call_id', ''),
                content=output if isinstance(output, str) else dump_arguments(output),
            )]))
            continue

        if role and (not item_type or item_type == 'message'):
            ir_role = 'assistant' if role == 'assistant' else 'user'
            blocks = _parse_message_content(item.get('content', ''))
            if ir_role == 'assistant' and pending_thinking:
                blocks.insert(0, ThinkingBlock(text=pending_thinking))
                pending_thinking = None
            if blocks:
                messages.append(IRMessage(ir_role, blocks))


def _last_assistant_or_new(messages: list[IRMessage]) -> IRMessage:
    if messages and messages[-1].role == 'assistant':
        return messages[-1]
    message = IRMessage('assistant', [])
    messages.append(message)
    return message


def _parse_message_content(content: Any) -> list[Block]:
    if content is None:
        return []
    if isinstance(content, str):
        return [TextBlock(text=content)] if content else []
    if not isinstance(content, list):
        return [TextBlock(text=str(content))]

    blocks: list[Block] = []
    for part in content:
        if isinstance(part, str):
            if part:
                blocks.append(TextBlock(text=part))
            continue
        if not isinstance(part, dict):
            continue
        part_type = part.get('type', '')
        if part_type in ('input_text', 'output_text', 'text'):
            if part.get('text'):
                blocks.append(TextBlock(text=part['text']))
        elif part_type == 'refusal':
            if part.get('refusal'):
                blocks.append(TextBlock(text=part['refusal']))
        elif part_type == 'input_image':
            image_url = part.get('image_url', '')
            url = image_url.get('url', '') if isinstance(image_url, dict) else str(image_url or '')
            if url.startswith('data:'):
                media_type, _, base64_data = url.partition(';base64,')
                blocks.append(ImageBlock(
                    media_type=media_type.replace('data:', '') or 'image/png',
                    data=base64_data,
                ))
            elif url:
                blocks.append(ImageBlock(url=url))
    return blocks


def _append_input_items(message: IRMessage, items: list[dict[str, Any]]) -> None:
    """将单条 IR 消息编码为 Responses `input` 项。

    普通用户消息优先使用 EasyInputMessage（{role, content}）以减少 token 开销，
    提高上游 prompt caching 的前缀匹配命中率。
    """
    if message.role == 'assistant':
        text = '\n'.join(b.text for b in message.blocks if isinstance(b, TextBlock))
        if text:
            items.append({
                'type': 'message',
                'role': 'assistant',
                'content': [{'type': 'output_text', 'text': text}],
            })
        for block in message.blocks:
            if isinstance(block, ToolCallBlock):
                items.append({
                    'type': 'function_call',
                    'call_id': block.id or gen_id('call_'),
                    'name': block.name,
                    'arguments': block.arguments,
                })
        # thinking 块不回传：Responses 上游通过自身的 reasoning 项管理思考内容
        return

    for block in message.blocks:
        if isinstance(block, ToolResultBlock):
            items.append({
                'type': 'function_call_output',
                'call_id': block.call_id,
                'output': block.content,
            })

    text_blocks = [b for b in message.blocks if isinstance(b, TextBlock)]
    image_blocks = [b for b in message.blocks if isinstance(b, ImageBlock)]
    if image_blocks:
        content: list[dict[str, Any]] = [
            {'type': 'input_text', 'text': b.text} for b in text_blocks if b.text
        ]
        content.extend(
            {'type': 'input_image', 'image_url': b.to_data_url()} for b in image_blocks
        )
        items.append({'role': 'user', 'content': content})
    elif text_blocks:
        items.append({
            'role': 'user',
            'content': '\n'.join(b.text for b in text_blocks),
        })


def _build_tool_choice(tool_choice: Any) -> Any:
    if tool_choice in ('auto', 'required', 'none'):
        return tool_choice
    if isinstance(tool_choice, dict) and tool_choice.get('name'):
        return {'type': 'function', 'name': tool_choice['name']}
    return None


# ═══════════════════════════════════════════════════════════
#  响应辅助
# ═══════════════════════════════════════════════════════════


def _reasoning_item(item_id: str, text: str) -> dict[str, Any]:
    return {
        'id': item_id,
        'type': 'reasoning',
        'summary': [{'type': 'summary_text', 'text': text}],
    }


def _message_item(item_id: str, text: str) -> dict[str, Any]:
    return {
        'id': item_id,
        'type': 'message',
        'status': 'completed',
        'role': 'assistant',
        'content': [{'type': 'output_text', 'annotations': [], 'text': text}],
    }


def _function_call_item(item_id: str, call_id: str, name: str, arguments: str) -> dict[str, Any]:
    return {
        'id': item_id,
        'type': 'function_call',
        'status': 'completed',
        'call_id': call_id or gen_id('call_'),
        'name': name,
        'arguments': arguments,
    }


def _parse_tool_call_item(item: dict[str, Any]) -> ToolCallBlock:
    if item.get('type') == 'custom_tool_call':
        return ToolCallBlock(
            id=item.get('call_id') or item.get('id') or gen_id('call_'),
            name=item.get('name') or item.get('namespace') or 'custom_tool',
            arguments=dump_arguments(item.get('input', '')),
        )
    return ToolCallBlock(
        id=item.get('call_id') or item.get('id') or gen_id('call_'),
        name=item.get('name', ''),
        arguments=dump_arguments(item.get('arguments', '{}')),
    )


def _extract_reasoning_text(item: dict[str, Any]) -> str:
    texts: list[str] = []
    summary = item.get('summary', [])
    if isinstance(summary, list):
        for part in summary:
            if isinstance(part, dict) and part.get('type') == 'summary_text':
                texts.append(part.get('text', ''))
    content = item.get('content', [])
    if isinstance(content, list):
        for part in content:
            if isinstance(part, dict) and part.get('type') in ('reasoning_text', 'text'):
                texts.append(part.get('text', ''))
    return ''.join(texts)


def _extract_output_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ''
    texts: list[str] = []
    for part in content:
        if isinstance(part, dict):
            if part.get('type') in ('output_text', 'text'):
                texts.append(part.get('text', ''))
            elif part.get('type') == 'refusal':
                texts.append(part.get('refusal', ''))
    return ''.join(texts)


def parse_responses_usage(usage: Any) -> IRUsage:
    """解析 Responses usage 结构（供本模块与透传路径共用）。"""
    if not isinstance(usage, dict):
        return IRUsage()
    input_details = usage.get('input_tokens_details') or {}
    output_details = usage.get('output_tokens_details') or {}
    return IRUsage(
        input_tokens=usage.get('input_tokens', 0) or 0,
        output_tokens=usage.get('output_tokens', 0) or 0,
        cached_tokens=(input_details.get('cached_tokens', 0) or 0) if isinstance(input_details, dict) else 0,
        reasoning_tokens=(output_details.get('reasoning_tokens', 0) or 0) if isinstance(output_details, dict) else 0,
    )


def _build_usage(usage: IRUsage) -> dict[str, Any]:
    return {
        'input_tokens': usage.input_tokens,
        'input_tokens_details': {'cached_tokens': usage.cached_tokens},
        'output_tokens': usage.output_tokens,
        'output_tokens_details': {'reasoning_tokens': usage.reasoning_tokens},
        'total_tokens': usage.total_tokens,
    }
