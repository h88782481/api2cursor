"""OpenAI Chat Completions 协议编解码器

客户端方向：解析 Cursor 发来的 CC 请求 / 把 IR 编码回 CC 响应与 chunk 流。
上游方向：把 IR 构建为发往中转站 /v1/chat/completions 的请求 / 解析其响应与 SSE 流。

同时收口旧版 openai_compat_fixer 的"协议清洗"逻辑：
  - Cursor 混入的 Anthropic 风格 tool_use/tool_result 内容块
  - 扁平工具定义、对象式 tool_choice
  - reasoningContent 命名差异、旧版 function_call、tool_calls 元数据补全
  - 流式 tool_calls 空白字段清理（防止 Cursor 用空值覆盖真实值）
"""

from __future__ import annotations

import json
import time
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
from .base import Codec, StreamDecoder, StreamEncoder, parse_json, sse_data


class ChatCompletionsCodec(Codec):
    format = 'chat'

    # ═══════════════════════════════════════════
    #  客户端方向
    # ═══════════════════════════════════════════

    def parse_request(self, payload: dict[str, Any]) -> IRRequest:
        request = IRRequest(
            model=payload.get('model', ''),
            stream=bool(payload.get('stream', False)),
        )

        system_parts: list[str] = []
        for message in payload.get('messages') or []:
            if not isinstance(message, dict):
                continue
            role = message.get('role', '')

            if role in ('system', 'developer'):
                text = _flatten_text(message.get('content'))
                if text:
                    system_parts.append(text)
                continue

            if role == 'tool':
                request.messages.append(IRMessage('user', [ToolResultBlock(
                    call_id=message.get('tool_call_id', ''),
                    content=_flatten_tool_content(message.get('content')),
                    name=message.get('name', ''),
                )]))
                continue

            ir_role = 'assistant' if role == 'assistant' else 'user'
            blocks = _parse_content_blocks(message.get('content'))

            if role == 'assistant':
                reasoning = message.get('reasoning_content') or message.get('reasoningContent')
                if reasoning:
                    blocks.insert(0, ThinkingBlock(text=str(reasoning)))
                for tool_call in message.get('tool_calls') or []:
                    if not isinstance(tool_call, dict):
                        continue
                    func = tool_call.get('function') or {}
                    blocks.append(ToolCallBlock(
                        id=tool_call.get('id') or gen_id('call_'),
                        name=func.get('name', ''),
                        arguments=dump_arguments(func.get('arguments', '{}')),
                    ))

            if blocks:
                request.messages.append(IRMessage(ir_role, blocks))

        request.system = '\n\n'.join(system_parts)
        request.tools = parse_tool_definitions(payload.get('tools'))
        request.tool_choice = parse_tool_choice(payload.get('tool_choice'))

        max_tokens = payload.get('max_tokens') or payload.get('max_completion_tokens')
        if isinstance(max_tokens, int) and max_tokens > 0:
            request.max_tokens = max_tokens
        if isinstance(payload.get('temperature'), (int, float)):
            request.temperature = payload['temperature']
        if isinstance(payload.get('top_p'), (int, float)):
            request.top_p = payload['top_p']
        request.stop = _parse_stop(payload.get('stop'))
        return request

    def build_response(self, response: IRResponse, client_model: str) -> dict[str, Any]:
        message: dict[str, Any] = {
            'role': 'assistant',
            'content': response.text() or None,
        }
        thinking = response.thinking()
        if thinking:
            message['reasoning_content'] = thinking

        tool_calls = [
            {
                'index': index,
                'id': block.id or gen_id('call_'),
                'type': 'function',
                'function': {'name': block.name, 'arguments': block.arguments},
            }
            for index, block in enumerate(response.tool_calls())
        ]
        if tool_calls:
            message['tool_calls'] = tool_calls

        usage = response.usage
        return {
            'id': response.id or gen_id('chatcmpl-'),
            'object': 'chat.completion',
            'created': int(time.time()),
            'model': client_model,
            'choices': [{
                'index': 0,
                'message': message,
                'finish_reason': response.finish_reason,
            }],
            'usage': {
                'prompt_tokens': usage.input_tokens,
                'completion_tokens': usage.output_tokens,
                'total_tokens': usage.total_tokens,
            },
        }

    def stream_encoder(self, client_model: str) -> StreamEncoder:
        return ChatStreamEncoder(client_model)

    # ═══════════════════════════════════════════
    #  上游方向
    # ═══════════════════════════════════════════

    def build_request(self, request: IRRequest, upstream_model: str) -> dict[str, Any]:
        messages: list[dict[str, Any]] = []
        if request.system:
            messages.append({'role': 'system', 'content': request.system})

        for message in request.messages:
            if message.role == 'assistant':
                messages.append(_build_assistant_message(message))
                continue
            # user 消息：tool_result 块拆分为独立的 tool 消息
            for block in message.blocks:
                if isinstance(block, ToolResultBlock):
                    messages.append({
                        'role': 'tool',
                        'tool_call_id': block.call_id,
                        'content': block.content,
                    })
            rest = [b for b in message.blocks if not isinstance(b, ToolResultBlock)]
            if rest:
                messages.append(_build_user_message(rest))

        payload: dict[str, Any] = {
            'model': upstream_model,
            'messages': messages,
            'stream': request.stream,
        }
        if request.tools:
            payload['tools'] = [
                {
                    'type': 'function',
                    'function': {
                        'name': t.name,
                        'description': t.description,
                        'parameters': t.parameters,
                    },
                }
                for t in request.tools
            ]
        tool_choice = _build_tool_choice(request.tool_choice)
        if tool_choice is not None:
            payload['tool_choice'] = tool_choice
        if request.max_tokens:
            payload['max_tokens'] = request.max_tokens
        if request.temperature is not None:
            payload['temperature'] = request.temperature
        if request.top_p is not None:
            payload['top_p'] = request.top_p
        if request.stop:
            payload['stop'] = request.stop
        return payload

    def parse_response(self, payload: dict[str, Any]) -> IRResponse:
        choice = (payload.get('choices') or [{}])[0]
        if not isinstance(choice, dict):
            choice = {}
        message = choice.get('message') or {}
        if not isinstance(message, dict):
            message = {}

        blocks: list[Block] = []
        reasoning = message.get('reasoning_content') or message.get('reasoningContent')
        if reasoning:
            blocks.append(ThinkingBlock(text=str(reasoning)))

        text = _flatten_text(message.get('content'))
        if text:
            blocks.append(TextBlock(text=text))

        tool_calls = message.get('tool_calls')
        if not tool_calls and isinstance(message.get('function_call'), dict):
            # 旧版 function_call 升级为 tool_calls
            legacy = message['function_call']
            tool_calls = [{'id': gen_id('call_'), 'function': legacy}]
        for tool_call in tool_calls or []:
            if not isinstance(tool_call, dict):
                continue
            func = tool_call.get('function') or {}
            blocks.append(ToolCallBlock(
                id=tool_call.get('id') or gen_id('call_'),
                name=func.get('name', ''),
                arguments=dump_arguments(func.get('arguments', '{}')),
            ))

        finish_reason = choice.get('finish_reason') or 'stop'
        if finish_reason == 'function_call':
            finish_reason = 'tool_calls'
        if any(isinstance(b, ToolCallBlock) for b in blocks):
            finish_reason = 'tool_calls'

        return IRResponse(
            id=payload.get('id') or gen_id('chatcmpl-'),
            model=payload.get('model', ''),
            blocks=blocks,
            finish_reason=finish_reason,
            usage=parse_cc_usage(payload.get('usage')),
        )

    def stream_decoder(self) -> StreamDecoder:
        return ChatStreamDecoder()

    def upstream_url(self, base_url: str, model: str, stream: bool) -> str:
        return f'{base_url.rstrip("/")}/v1/chat/completions'


# ═══════════════════════════════════════════════════════════
#  流式解码器: CC SSE → IR 事件
# ═══════════════════════════════════════════════════════════


class ChatStreamDecoder(StreamDecoder):
    def __init__(self):
        self._started = False
        self._ended = False
        self._finish_reason: str | None = None
        self._usage: IRUsage | None = None
        # 上游 tool_calls index → 顺序化的 IR index
        self._slots: dict[int, int] = {}
        self._meta: dict[int, dict[str, str]] = {}
        self._current_upstream_index: int | None = None

    def decode(self, event_type: str, data: str) -> list[StreamEvent]:
        if data == '[DONE]':
            return self._emit_end()
        chunk = parse_json(data)
        if chunk is None:
            return []

        if isinstance(chunk.get('error'), dict):
            error = chunk['error']
            return [StreamError(
                message=str(error.get('message', error)),
                error_type=error.get('type', 'upstream_error'),
            )]

        events: list[StreamEvent] = []
        if isinstance(chunk.get('usage'), dict) and chunk['usage']:
            self._usage = parse_cc_usage(chunk['usage'])

        for choice in chunk.get('choices') or []:
            if not isinstance(choice, dict):
                continue
            delta = choice.get('delta') or {}
            if not isinstance(delta, dict):
                delta = {}

            if not self._started:
                self._started = True
                events.append(StreamStart(id=chunk.get('id', ''), model=chunk.get('model', '')))

            reasoning = delta.get('reasoning_content') or delta.get('reasoningContent')
            if reasoning:
                events.append(ThinkingDelta(str(reasoning)))

            content = delta.get('content')
            if isinstance(content, str) and content:
                events.append(TextDelta(content))

            tool_calls = delta.get('tool_calls')
            if not tool_calls and isinstance(delta.get('function_call'), dict):
                tool_calls = [_legacy_function_call_to_tool_call(delta['function_call'])]
            for tool_call in tool_calls or []:
                if isinstance(tool_call, dict):
                    events.extend(self._handle_tool_delta(tool_call))

            finish = choice.get('finish_reason')
            if finish:
                self._finish_reason = 'tool_calls' if finish == 'function_call' else finish

        return events

    def finalize(self) -> list[StreamEvent]:
        return self._emit_end()

    def _handle_tool_delta(self, tool_call: dict[str, Any]) -> list[StreamEvent]:
        events: list[StreamEvent] = []
        func = tool_call.get('function') or {}
        if not isinstance(func, dict):
            func = {}

        # 部分兼容提供商会发送空字符串的 id/name，需要当作缺失处理
        name = func.get('name') or ''
        if not str(name).strip():
            name = ''
        call_id = tool_call.get('id') or ''
        if not str(call_id).strip():
            call_id = ''

        upstream_index = tool_call.get('index')
        if not isinstance(upstream_index, int):
            upstream_index = (
                self._current_upstream_index
                if self._current_upstream_index is not None
                else 0
            )
        self._current_upstream_index = upstream_index

        if upstream_index not in self._slots:
            our_index = len(self._slots)
            self._slots[upstream_index] = our_index
            self._meta[our_index] = {
                'id': call_id or gen_id('call_'),
                'name': name,
                'args': '',
            }
            events.append(ToolCallStart(
                index=our_index,
                id=self._meta[our_index]['id'],
                name=name,
            ))

        our_index = self._slots[upstream_index]
        meta = self._meta[our_index]
        if name and not meta['name']:
            meta['name'] = name

        arguments = func.get('arguments')
        if isinstance(arguments, str) and arguments:
            meta['args'] += arguments
            events.append(ToolCallDelta(index=our_index, arguments=arguments))

        return events

    def _emit_end(self) -> list[StreamEvent]:
        if self._ended:
            return []
        self._ended = True
        events: list[StreamEvent] = []
        for our_index in sorted(self._meta):
            meta = self._meta[our_index]
            events.append(ToolCallEnd(
                index=our_index,
                id=meta['id'],
                name=meta['name'],
                arguments=meta['args'],
            ))
        finish = self._finish_reason or ('tool_calls' if self._meta else 'stop')
        events.append(StreamEnd(finish_reason=finish, usage=self._usage))
        return events


# ═══════════════════════════════════════════════════════════
#  流式编码器: IR 事件 → CC chunk SSE
# ═══════════════════════════════════════════════════════════


class ChatStreamEncoder(StreamEncoder):
    def __init__(self, client_model: str):
        self._model = client_model
        self._id = gen_id('chatcmpl-')
        self._created = int(time.time())
        self._role_sent = False
        self._done_sent = False

    def encode(self, event: StreamEvent) -> list[str]:
        if isinstance(event, StreamStart):
            if event.id:
                self._id = event.id
            return self._ensure_role()
        if isinstance(event, TextDelta):
            return self._ensure_role() + [self._chunk({'content': event.text})]
        if isinstance(event, ThinkingDelta):
            return self._ensure_role() + [self._chunk({'reasoning_content': event.text})]
        if isinstance(event, ToolCallStart):
            return self._ensure_role() + [self._chunk({'tool_calls': [{
                'index': event.index,
                'id': event.id,
                'type': 'function',
                'function': {'name': event.name, 'arguments': ''},
            }]})]
        if isinstance(event, ToolCallDelta):
            return [self._chunk({'tool_calls': [{
                'index': event.index,
                'function': {'arguments': event.arguments},
            }]})]
        if isinstance(event, StreamEnd):
            return self._ensure_role() + [self._chunk(
                {}, finish_reason=event.finish_reason, usage=event.usage,
            )]
        if isinstance(event, StreamError):
            return [sse_data({'error': {'message': event.message, 'type': event.error_type}})]
        return []

    def finalize(self) -> list[str]:
        if self._done_sent:
            return []
        self._done_sent = True
        return [sse_data('[DONE]')]

    def _ensure_role(self) -> list[str]:
        if self._role_sent:
            return []
        self._role_sent = True
        return [self._chunk({'role': 'assistant', 'content': ''})]

    def _chunk(
        self,
        delta: dict[str, Any],
        finish_reason: str | None = None,
        usage: IRUsage | None = None,
    ) -> str:
        choice: dict[str, Any] = {'index': 0, 'delta': delta}
        if finish_reason:
            choice['finish_reason'] = finish_reason
        chunk: dict[str, Any] = {
            'id': self._id,
            'object': 'chat.completion.chunk',
            'created': self._created,
            'model': self._model,
            'choices': [choice],
        }
        if usage is not None:
            chunk['usage'] = {
                'prompt_tokens': usage.input_tokens,
                'completion_tokens': usage.output_tokens,
                'total_tokens': usage.total_tokens,
            }
        return sse_data(chunk)


# ═══════════════════════════════════════════════════════════
#  chat → chat 透传路径的请求清洗（保留未知字段）
# ═══════════════════════════════════════════════════════════


def normalize_cc_request(payload: dict[str, Any], upstream_model: str) -> dict[str, Any]:
    """就地清洗 Cursor 发来的 CC 请求，供 chat → chat 透传使用。

    与走 IR 的转换路径不同，这里保留请求体中的未知字段（如 stream_options），
    只做最小标准化：
    1. Anthropic 风格 tool_use / tool_result 内容块转回 OpenAI 语义
    2. 扁平工具定义补成标准 function tool
    3. 对象式 tool_choice 转为字符串写法
    """
    payload['model'] = upstream_model

    if isinstance(payload.get('messages'), list):
        converted: list[Any] = []
        for message in payload['messages']:
            converted.extend(_normalize_cc_message(message))
        payload['messages'] = converted

    if isinstance(payload.get('tools'), list):
        payload['tools'] = [_normalize_tool_definition(t) for t in payload['tools']]

    tool_choice = payload.get('tool_choice')
    if isinstance(tool_choice, dict):
        if tool_choice.get('type') == 'auto':
            payload['tool_choice'] = 'auto'
        elif tool_choice.get('type') == 'any':
            payload['tool_choice'] = 'required'

    return payload


def _normalize_cc_message(message: Any) -> list[Any]:
    """将单条消息中的 Anthropic 风格内容块转换为 OpenAI 风格消息。"""
    if not isinstance(message, dict):
        return [message]
    content = message.get('content')
    if not isinstance(content, list):
        return [message]

    has_tool_use = any(isinstance(b, dict) and b.get('type') == 'tool_use' for b in content)
    has_tool_result = any(isinstance(b, dict) and b.get('type') == 'tool_result' for b in content)
    if not has_tool_use and not has_tool_result:
        return [message]

    role = message.get('role', '')
    if role == 'assistant' and has_tool_use:
        text_parts: list[str] = []
        tool_calls: list[dict[str, Any]] = []
        for block in content:
            if not isinstance(block, dict):
                continue
            if block.get('type') == 'text':
                text_parts.append(block.get('text', ''))
            elif block.get('type') == 'tool_use':
                tool_calls.append({
                    'id': block.get('id', gen_id('call_')),
                    'type': 'function',
                    'function': {
                        'name': block.get('name', ''),
                        'arguments': dump_arguments(block.get('input', {})),
                    },
                })
        result: dict[str, Any] = {
            'role': 'assistant',
            'content': '\n'.join(text_parts) if text_parts else None,
        }
        if tool_calls:
            result['tool_calls'] = tool_calls
        return [result]

    if has_tool_result:
        converted: list[dict[str, Any]] = []
        other_parts: list[Any] = []
        for block in content:
            if not isinstance(block, dict):
                continue
            if block.get('type') == 'tool_result':
                converted.append({
                    'role': 'tool',
                    'tool_call_id': block.get('tool_use_id', ''),
                    'content': _flatten_tool_content(block.get('content', '')),
                })
            else:
                other_parts.append(block)
        if other_parts:
            converted.append({'role': role, 'content': other_parts})
        return converted

    return [message]


def _normalize_tool_definition(tool: Any) -> Any:
    if not isinstance(tool, dict):
        return tool
    if tool.get('type') == 'function' and 'function' in tool:
        return tool
    if 'name' not in tool:
        return tool
    return {
        'type': 'function',
        'function': {
            'name': tool.get('name', ''),
            'description': tool.get('description', ''),
            'parameters': (
                tool.get('input_schema')
                or tool.get('parameters')
                or {'type': 'object', 'properties': {}}
            ),
        },
    }


# ═══════════════════════════════════════════════════════════
#  内部辅助
# ═══════════════════════════════════════════════════════════


def parse_cc_usage(usage: Any) -> IRUsage:
    """解析 CC usage 结构。"""
    if not isinstance(usage, dict):
        return IRUsage()
    prompt_details = usage.get('prompt_tokens_details') or {}
    completion_details = usage.get('completion_tokens_details') or {}
    return IRUsage(
        input_tokens=usage.get('prompt_tokens', 0) or 0,
        output_tokens=usage.get('completion_tokens', 0) or 0,
        cached_tokens=(prompt_details.get('cached_tokens', 0) or 0) if isinstance(prompt_details, dict) else 0,
        reasoning_tokens=(completion_details.get('reasoning_tokens', 0) or 0) if isinstance(completion_details, dict) else 0,
    )


def _parse_stop(stop: Any) -> list[str]:
    if isinstance(stop, str):
        return [stop] if stop else []
    if isinstance(stop, list):
        return [s for s in stop if isinstance(s, str)]
    return []


def _legacy_function_call_to_tool_call(function_call: dict[str, Any]) -> dict[str, Any]:
    tool_call: dict[str, Any] = {'index': 0, 'function': {}}
    if 'name' in function_call:
        tool_call['id'] = gen_id('call_')
        tool_call['function']['name'] = function_call['name']
    if 'arguments' in function_call:
        tool_call['function']['arguments'] = function_call['arguments']
    return tool_call


def _parse_content_blocks(content: Any) -> list[Block]:
    """将 CC 消息 content 解析为 IR 内容块。"""
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
        if part_type in ('text', 'input_text', 'output_text'):
            text = part.get('text', '')
            if text:
                blocks.append(TextBlock(text=text))
        elif part_type == 'image_url':
            blocks.append(_parse_image_url(part))
        elif part_type == 'image':
            blocks.append(_parse_anthropic_image(part))
        elif part_type == 'tool_use':
            blocks.append(ToolCallBlock(
                id=part.get('id') or gen_id('call_'),
                name=part.get('name', ''),
                arguments=dump_arguments(part.get('input', {})),
            ))
        elif part_type == 'tool_result':
            blocks.append(ToolResultBlock(
                call_id=part.get('tool_use_id', ''),
                content=_flatten_tool_content(part.get('content', '')),
            ))
    return blocks


def _parse_image_url(part: dict[str, Any]) -> ImageBlock:
    url_data = part.get('image_url', {})
    url = url_data.get('url', '') if isinstance(url_data, dict) else str(url_data)
    if url.startswith('data:'):
        media_type, _, base64_data = url.partition(';base64,')
        return ImageBlock(
            media_type=media_type.replace('data:', '') or 'image/png',
            data=base64_data,
        )
    return ImageBlock(url=url)


def _parse_anthropic_image(part: dict[str, Any]) -> ImageBlock:
    source = part.get('source', {})
    if not isinstance(source, dict):
        return ImageBlock()
    if source.get('type') == 'base64':
        return ImageBlock(
            media_type=source.get('media_type', 'image/png'),
            data=source.get('data', ''),
        )
    return ImageBlock(url=source.get('url', ''))


def _build_user_message(blocks: list[Block]) -> dict[str, Any]:
    has_image = any(isinstance(b, ImageBlock) for b in blocks)
    if not has_image:
        text = '\n'.join(b.text for b in blocks if isinstance(b, TextBlock))
        return {'role': 'user', 'content': text}

    parts: list[dict[str, Any]] = []
    for block in blocks:
        if isinstance(block, TextBlock):
            parts.append({'type': 'text', 'text': block.text})
        elif isinstance(block, ImageBlock):
            parts.append({'type': 'image_url', 'image_url': {'url': block.to_data_url()}})
    return {'role': 'user', 'content': parts}


def _build_assistant_message(message: IRMessage) -> dict[str, Any]:
    text = '\n'.join(b.text for b in message.blocks if isinstance(b, TextBlock))
    result: dict[str, Any] = {'role': 'assistant', 'content': text or None}

    thinking = ''.join(b.text for b in message.blocks if isinstance(b, ThinkingBlock))
    if thinking:
        result['reasoning_content'] = thinking

    tool_calls = [
        {
            'id': b.id or gen_id('call_'),
            'type': 'function',
            'function': {'name': b.name, 'arguments': b.arguments},
        }
        for b in message.blocks if isinstance(b, ToolCallBlock)
    ]
    if tool_calls:
        result['tool_calls'] = tool_calls
    return result


def _build_tool_choice(tool_choice: Any) -> Any:
    if tool_choice in ('auto', 'required', 'none'):
        return tool_choice
    if isinstance(tool_choice, dict) and tool_choice.get('name'):
        return {'type': 'function', 'function': {'name': tool_choice['name']}}
    return None


def _flatten_text(content: Any) -> str:
    """将 content 扁平化为纯文本。"""
    if content is None:
        return ''
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict) and part.get('type') in ('text', 'input_text', 'output_text'):
                parts.append(part.get('text', ''))
            elif isinstance(part, dict) and part.get('type') == 'refusal':
                parts.append(part.get('refusal', ''))
        return '\n'.join(p for p in parts if p)
    return str(content)


def _flatten_tool_content(content: Any) -> str:
    """将工具结果内容降维为字符串。"""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [
            block.get('text', '')
            for block in content
            if isinstance(block, dict) and block.get('type') == 'text'
        ]
        if parts:
            return '\n'.join(parts)
        return json.dumps(content, ensure_ascii=False)
    if content is None:
        return ''
    return json.dumps(content, ensure_ascii=False) if not isinstance(content, str) else content
