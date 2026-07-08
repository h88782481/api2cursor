"""Anthropic Messages 协议编解码器

客户端方向：解析 Cursor 发来的 /v1/messages 请求 / 把 IR 编码回 Messages 响应与 SSE 流。
上游方向：把 IR 构建为发往中转站 /v1/messages 的请求 / 解析其响应与 SSE 流。

包含 Anthropic 特有的处理：
  - max_tokens 最小 8192 兜底
  - 顶层 cache_control 自动提示缓存
  - 相邻同角色消息合并（Anthropic 要求角色严格交替）
  - tool_use 块 ID 补全与 stop_reason 修正
"""

from __future__ import annotations

from typing import Any

from ..compat.tools import dump_arguments, parse_arguments_dict, parse_tool_choice, parse_tool_definitions
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
from .base import Codec, StreamDecoder, StreamEncoder, parse_json, sse_event

# Anthropic stop_reason ↔ IR finish_reason
_STOP_TO_FINISH = {
    'end_turn': 'stop',
    'max_tokens': 'length',
    'tool_use': 'tool_calls',
    'stop_sequence': 'stop',
    'refusal': 'content_filter',
}
_FINISH_TO_STOP = {
    'stop': 'end_turn',
    'length': 'max_tokens',
    'tool_calls': 'tool_use',
    'content_filter': 'end_turn',
}


class AnthropicCodec(Codec):
    format = 'messages'

    # ═══════════════════════════════════════════
    #  客户端方向
    # ═══════════════════════════════════════════

    def parse_request(self, payload: dict[str, Any]) -> IRRequest:
        request = IRRequest(
            model=payload.get('model', ''),
            stream=bool(payload.get('stream', False)),
            system=_flatten_system(payload.get('system')),
        )

        for message in payload.get('messages') or []:
            if not isinstance(message, dict):
                continue
            role = 'assistant' if message.get('role') == 'assistant' else 'user'
            blocks = _parse_content_blocks(message.get('content'))
            if blocks:
                request.messages.append(IRMessage(role, blocks))

        request.tools = parse_tool_definitions(payload.get('tools'))
        request.tool_choice = parse_tool_choice(payload.get('tool_choice'))

        if isinstance(payload.get('max_tokens'), int):
            request.max_tokens = payload['max_tokens']
        if isinstance(payload.get('temperature'), (int, float)):
            request.temperature = payload['temperature']
        if isinstance(payload.get('top_p'), (int, float)):
            request.top_p = payload['top_p']
        stop = payload.get('stop_sequences')
        if isinstance(stop, list):
            request.stop = [s for s in stop if isinstance(s, str)]
        return request

    def build_response(self, response: IRResponse, client_model: str) -> dict[str, Any]:
        content: list[dict[str, Any]] = []
        for block in response.blocks:
            if isinstance(block, ThinkingBlock):
                item: dict[str, Any] = {'type': 'thinking', 'thinking': block.text}
                if block.signature:
                    item['signature'] = block.signature
                content.append(item)
            elif isinstance(block, TextBlock):
                if block.text:
                    content.append({'type': 'text', 'text': block.text})
            elif isinstance(block, ToolCallBlock):
                content.append({
                    'type': 'tool_use',
                    'id': block.id or gen_id('toolu_'),
                    'name': block.name,
                    'input': parse_arguments_dict(block.arguments),
                })

        usage = response.usage
        return {
            'id': response.id or gen_id('msg_'),
            'type': 'message',
            'role': 'assistant',
            'model': client_model,
            'content': content,
            'stop_reason': _FINISH_TO_STOP.get(response.finish_reason, 'end_turn'),
            'stop_sequence': None,
            'usage': {
                'input_tokens': usage.input_tokens,
                'output_tokens': usage.output_tokens,
            },
        }

    def stream_encoder(self, client_model: str) -> StreamEncoder:
        return MessagesStreamEncoder(client_model)

    def error_event(self, message: str, error_type: str = 'upstream_error') -> str:
        return sse_event('error', {
            'type': 'error',
            'error': {'type': error_type, 'message': message},
        })

    # ═══════════════════════════════════════════
    #  上游方向
    # ═══════════════════════════════════════════

    def build_request(self, request: IRRequest, upstream_model: str) -> dict[str, Any]:
        messages: list[dict[str, Any]] = []
        for message in request.messages:
            content = _build_content_blocks(message)
            if content:
                messages.append({'role': message.role, 'content': content})
        messages = _merge_same_role(messages)

        payload: dict[str, Any] = {
            'model': upstream_model,
            'messages': messages,
            # 未设置或设置过小都兜底到 8192，避免上游因默认值过小过早截断
            'max_tokens': max(request.max_tokens or 8192, 8192),
            'stream': request.stream,
        }
        if request.system:
            payload['system'] = request.system
        if request.tools:
            payload['tools'] = [
                {
                    'name': t.name,
                    'description': t.description,
                    'input_schema': t.parameters,
                }
                for t in request.tools
            ]
        if request.temperature is not None:
            payload['temperature'] = request.temperature
        if request.top_p is not None:
            payload['top_p'] = request.top_p
        if request.stop:
            payload['stop_sequences'] = request.stop

        optimize_cache_control(payload)
        return payload

    def parse_response(self, payload: dict[str, Any]) -> IRResponse:
        fix_anthropic_tool_use(payload)

        blocks: list[Block] = []
        for block in payload.get('content') or []:
            if not isinstance(block, dict):
                continue
            block_type = block.get('type', '')
            if block_type == 'text':
                blocks.append(TextBlock(text=block.get('text', '')))
            elif block_type == 'thinking':
                blocks.append(ThinkingBlock(
                    text=block.get('thinking', ''),
                    signature=block.get('signature', ''),
                ))
            elif block_type == 'tool_use':
                blocks.append(ToolCallBlock(
                    id=block.get('id') or gen_id('toolu_'),
                    name=block.get('name', ''),
                    arguments=dump_arguments(block.get('input', {})),
                ))

        usage = payload.get('usage') or {}
        return IRResponse(
            id=payload.get('id') or gen_id('msg_'),
            model=payload.get('model', ''),
            blocks=blocks,
            finish_reason=_STOP_TO_FINISH.get(payload.get('stop_reason', 'end_turn'), 'stop'),
            usage=IRUsage(
                input_tokens=usage.get('input_tokens', 0) or 0,
                output_tokens=usage.get('output_tokens', 0) or 0,
                cached_tokens=usage.get('cache_read_input_tokens', 0) or 0,
            ),
        )

    def stream_decoder(self) -> StreamDecoder:
        return MessagesStreamDecoder()

    def upstream_url(self, base_url: str, model: str, stream: bool) -> str:
        return f'{base_url.rstrip("/")}/v1/messages'

    def build_headers(self, api_key: str) -> dict[str, str]:
        headers = {
            'anthropic-version': '2023-06-01',
            'Content-Type': 'application/json',
        }
        if api_key.startswith('sk-'):
            headers['x-api-key'] = api_key
        elif api_key:
            headers['Authorization'] = f'Bearer {api_key}'
        return headers


# ═══════════════════════════════════════════════════════════
#  流式解码器: Anthropic SSE → IR 事件
# ═══════════════════════════════════════════════════════════


class MessagesStreamDecoder(StreamDecoder):
    def __init__(self):
        self._input_tokens = 0
        self._ended = False
        # Anthropic block index → 工具信息
        self._tools: dict[int, dict[str, Any]] = {}
        self._tool_count = 0

    def decode(self, event_type: str, data: str) -> list[StreamEvent]:
        payload = parse_json(data)
        if payload is None:
            return []
        event_type = event_type or payload.get('type', '')

        if event_type == 'message_start':
            message = payload.get('message', {})
            usage = message.get('usage', {}) if isinstance(message, dict) else {}
            self._input_tokens = usage.get('input_tokens', 0) or 0
            return [StreamStart(id=message.get('id', ''), model=message.get('model', ''))]

        if event_type == 'content_block_start':
            block = payload.get('content_block', {})
            if isinstance(block, dict) and block.get('type') == 'tool_use':
                index = payload.get('index', 0)
                our_index = self._tool_count
                self._tool_count += 1
                self._tools[index] = {
                    'index': our_index,
                    'id': block.get('id') or gen_id('toolu_'),
                    'name': block.get('name', ''),
                    'args': '',
                }
                return [ToolCallStart(
                    index=our_index,
                    id=self._tools[index]['id'],
                    name=self._tools[index]['name'],
                )]
            return []

        if event_type == 'content_block_delta':
            delta = payload.get('delta', {})
            delta_type = delta.get('type', '')
            if delta_type == 'text_delta' and delta.get('text'):
                return [TextDelta(delta['text'])]
            if delta_type == 'thinking_delta' and delta.get('thinking'):
                return [ThinkingDelta(delta['thinking'])]
            if delta_type == 'input_json_delta' and delta.get('partial_json'):
                index = payload.get('index', 0)
                tool = self._tools.get(index)
                if tool is None and self._tools:
                    tool = self._tools[max(self._tools)]
                if tool is not None:
                    tool['args'] += delta['partial_json']
                    return [ToolCallDelta(index=tool['index'], arguments=delta['partial_json'])]
            return []

        if event_type == 'content_block_stop':
            index = payload.get('index', 0)
            tool = self._tools.pop(index, None)
            if tool is not None:
                return [ToolCallEnd(
                    index=tool['index'],
                    id=tool['id'],
                    name=tool['name'],
                    arguments=tool['args'],
                )]
            return []

        if event_type == 'message_delta':
            if self._ended:
                return []
            self._ended = True
            delta = payload.get('delta', {})
            usage = payload.get('usage', {})
            output_tokens = usage.get('output_tokens', 0) or 0 if isinstance(usage, dict) else 0
            return [StreamEnd(
                finish_reason=_STOP_TO_FINISH.get(delta.get('stop_reason', 'end_turn'), 'stop'),
                usage=IRUsage(input_tokens=self._input_tokens, output_tokens=output_tokens),
            )]

        if event_type == 'error':
            error = payload.get('error', {})
            return [StreamError(
                message=str(error.get('message', payload)),
                error_type=error.get('type', 'upstream_error'),
            )]

        return []

    def finalize(self) -> list[StreamEvent]:
        if self._ended:
            return []
        self._ended = True
        return [StreamEnd(
            finish_reason='tool_calls' if self._tool_count else 'stop',
            usage=IRUsage(input_tokens=self._input_tokens),
        )]


# ═══════════════════════════════════════════════════════════
#  流式编码器: IR 事件 → Anthropic SSE
# ═══════════════════════════════════════════════════════════


class MessagesStreamEncoder(StreamEncoder):
    def __init__(self, client_model: str):
        self._model = client_model
        self._id = gen_id('msg_')
        self._started = False
        self._ended = False
        self._index = -1
        self._block_type: str | None = None  # thinking / text / tool_use

    def encode(self, event: StreamEvent) -> list[str]:
        if isinstance(event, StreamStart):
            return self._ensure_started()
        if isinstance(event, ThinkingDelta):
            out = self._ensure_block('thinking')
            out.append(self._delta({'type': 'thinking_delta', 'thinking': event.text}))
            return out
        if isinstance(event, TextDelta):
            out = self._ensure_block('text')
            out.append(self._delta({'type': 'text_delta', 'text': event.text}))
            return out
        if isinstance(event, ToolCallStart):
            out = self._ensure_started() + self._close_block()
            self._index += 1
            self._block_type = 'tool_use'
            out.append(sse_event('content_block_start', {
                'type': 'content_block_start',
                'index': self._index,
                'content_block': {
                    'type': 'tool_use',
                    'id': event.id or gen_id('toolu_'),
                    'name': event.name,
                    'input': {},
                },
            }))
            return out
        if isinstance(event, ToolCallDelta):
            if self._block_type != 'tool_use':
                return []
            return [self._delta({'type': 'input_json_delta', 'partial_json': event.arguments})]
        if isinstance(event, ToolCallEnd):
            if self._block_type == 'tool_use':
                return self._close_block()
            return []
        if isinstance(event, StreamEnd):
            if self._ended:
                return []
            self._ended = True
            usage = event.usage or IRUsage()
            out = self._ensure_started() + self._close_block()
            out.append(sse_event('message_delta', {
                'type': 'message_delta',
                'delta': {
                    'stop_reason': _FINISH_TO_STOP.get(event.finish_reason, 'end_turn'),
                    'stop_sequence': None,
                },
                'usage': {
                    'input_tokens': usage.input_tokens,
                    'output_tokens': usage.output_tokens,
                },
            }))
            out.append(sse_event('message_stop', {'type': 'message_stop'}))
            return out
        if isinstance(event, StreamError):
            return [sse_event('error', {
                'type': 'error',
                'error': {'type': event.error_type, 'message': event.message},
            })]
        return []

    def finalize(self) -> list[str]:
        if not self._started or self._ended:
            return []
        return self.encode(StreamEnd(finish_reason='stop'))

    def _ensure_started(self) -> list[str]:
        if self._started:
            return []
        self._started = True
        return [sse_event('message_start', {
            'type': 'message_start',
            'message': {
                'id': self._id,
                'type': 'message',
                'role': 'assistant',
                'model': self._model,
                'content': [],
                'stop_reason': None,
                'stop_sequence': None,
                'usage': {'input_tokens': 0, 'output_tokens': 0},
            },
        })]

    def _ensure_block(self, block_type: str) -> list[str]:
        out = self._ensure_started()
        if self._block_type == block_type:
            return out
        out.extend(self._close_block())
        self._index += 1
        self._block_type = block_type
        content_block = (
            {'type': 'thinking', 'thinking': ''}
            if block_type == 'thinking'
            else {'type': 'text', 'text': ''}
        )
        out.append(sse_event('content_block_start', {
            'type': 'content_block_start',
            'index': self._index,
            'content_block': content_block,
        }))
        return out

    def _close_block(self) -> list[str]:
        if self._block_type is None:
            return []
        self._block_type = None
        return [sse_event('content_block_stop', {
            'type': 'content_block_stop',
            'index': self._index,
        })]

    def _delta(self, delta: dict[str, Any]) -> str:
        return sse_event('content_block_delta', {
            'type': 'content_block_delta',
            'index': self._index,
            'delta': delta,
        })


# ═══════════════════════════════════════════════════════════
#  请求 / 响应辅助
# ═══════════════════════════════════════════════════════════


def fix_anthropic_tool_use(payload: dict[str, Any]) -> dict[str, Any]:
    """修复 Anthropic 响应中的 tool_use 块（补全 ID、修正 stop_reason）。"""
    content = payload.get('content')
    if not isinstance(content, list):
        return payload

    has_tool_use = False
    for block in content:
        if isinstance(block, dict) and block.get('type') == 'tool_use':
            has_tool_use = True
            if not block.get('id'):
                block['id'] = gen_id('toolu_')

    if has_tool_use and payload.get('stop_reason') != 'tool_use':
        payload['stop_reason'] = 'tool_use'
    return payload


def optimize_cache_control(payload: dict[str, Any]) -> None:
    """为 Anthropic Messages 请求启用顶层自动 prompt caching。

    2026 版 Claude API 支持在请求顶层使用 `cache_control` 开启自动缓存，
    由上游自动把断点放到最后一个可缓存块并随多轮对话前移。相比手动在嵌套
    content blocks 上打断点，这种方式对 Anthropic 兼容中转站更稳定。
    """
    for message in payload.get('messages', []):
        content = message.get('content')
        if isinstance(content, str):
            message['content'] = [{'type': 'text', 'text': content}]
        elif content is None:
            message['content'] = []

    payload.pop('cache_control', None)
    for tool in payload.get('tools', []):
        if isinstance(tool, dict):
            tool.pop('cache_control', None)
    system = payload.get('system')
    if isinstance(system, list):
        for block in system:
            if isinstance(block, dict):
                block.pop('cache_control', None)
    for message in payload.get('messages', []):
        content = message.get('content')
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict):
                    block.pop('cache_control', None)

    payload['cache_control'] = {'type': 'ephemeral'}


def _flatten_system(system: Any) -> str:
    if isinstance(system, str):
        return system
    if isinstance(system, list):
        return '\n'.join(
            block.get('text', '') for block in system
            if isinstance(block, dict) and block.get('type') == 'text'
        )
    return str(system) if system else ''


def _parse_content_blocks(content: Any) -> list[Block]:
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
        if part_type == 'text':
            if part.get('text'):
                blocks.append(TextBlock(text=part['text']))
        elif part_type == 'thinking':
            blocks.append(ThinkingBlock(
                text=part.get('thinking', ''),
                signature=part.get('signature', ''),
            ))
        elif part_type == 'image':
            source = part.get('source', {})
            if isinstance(source, dict) and source.get('type') == 'base64':
                blocks.append(ImageBlock(
                    media_type=source.get('media_type', 'image/png'),
                    data=source.get('data', ''),
                ))
            elif isinstance(source, dict):
                blocks.append(ImageBlock(url=source.get('url', '')))
        elif part_type == 'tool_use':
            blocks.append(ToolCallBlock(
                id=part.get('id') or gen_id('toolu_'),
                name=part.get('name', ''),
                arguments=dump_arguments(part.get('input', {})),
            ))
        elif part_type == 'tool_result':
            blocks.append(ToolResultBlock(
                call_id=part.get('tool_use_id', ''),
                content=_flatten_tool_result(part.get('content', '')),
            ))
    return blocks


def _flatten_tool_result(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [
            block.get('text', '')
            for block in content
            if isinstance(block, dict) and block.get('type') == 'text'
        ]
        return '\n'.join(parts)
    return str(content) if content else ''


def _build_content_blocks(message: IRMessage) -> list[dict[str, Any]]:
    """将 IR 消息编码为 Anthropic content 块，thinking 块保持在最前。"""
    thinking: list[dict[str, Any]] = []
    others: list[dict[str, Any]] = []

    for block in message.blocks:
        if isinstance(block, ThinkingBlock):
            item: dict[str, Any] = {'type': 'thinking', 'thinking': block.text}
            if block.signature:
                item['signature'] = block.signature
            thinking.append(item)
        elif isinstance(block, TextBlock):
            if block.text:
                others.append({'type': 'text', 'text': block.text})
        elif isinstance(block, ImageBlock):
            if block.data:
                others.append({
                    'type': 'image',
                    'source': {
                        'type': 'base64',
                        'media_type': block.media_type,
                        'data': block.data,
                    },
                })
            elif block.url:
                others.append({
                    'type': 'image',
                    'source': {'type': 'url', 'url': block.url},
                })
        elif isinstance(block, ToolCallBlock):
            others.append({
                'type': 'tool_use',
                'id': block.id or gen_id('toolu_'),
                'name': block.name,
                'input': parse_arguments_dict(block.arguments),
            })
        elif isinstance(block, ToolResultBlock):
            others.append({
                'type': 'tool_result',
                'tool_use_id': block.call_id,
                'content': block.content,
            })

    return thinking + others


def _merge_same_role(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """合并相邻同角色消息（Anthropic 要求消息角色严格交替）。"""
    if not messages:
        return messages
    merged = [messages[0]]
    for message in messages[1:]:
        if message['role'] == merged[-1]['role']:
            merged[-1]['content'] = list(merged[-1]['content']) + list(message['content'])
        else:
            merged.append(message)
    return merged
