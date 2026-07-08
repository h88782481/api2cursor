"""Gemini generateContent 协议编解码器（仅上游方向）

把 IR 构建为 Gemini generateContent / streamGenerateContent 请求，
并将 Gemini 响应与 SSE 流解析回 IR。Cursor 不会以 Gemini 格式发请求，
因此不实现客户端方向。

Gemini 3（2026）起强制要求在多轮工具调用中回传 functionCall part 的
thoughtSignature，缺失会直接 400。由于 Cursor 不会回传该字段，这里用
进程内缓存按 tool_call_id 记忆签名，构建请求时重新附加；缓存未命中时
使用官方的 skip_thought_signature_validator 哨兵值兜底。
"""

from __future__ import annotations

import time
from typing import Any

from ..compat.tools import dump_arguments, parse_arguments_dict
from ..core.ir import (
    Block,
    IRMessage,
    IRRequest,
    IRResponse,
    IRUsage,
    ImageBlock,
    StreamEnd,
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
from .base import Codec, StreamDecoder, parse_json, strip_version_suffix

_FINISH_REASON_MAP = {
    'STOP': 'stop',
    'MAX_TOKENS': 'length',
    'SAFETY': 'content_filter',
    'RECITATION': 'content_filter',
}

# Gemini 3 缺少 thoughtSignature 时的官方哨兵值（跳过校验，代价是推理质量下降）
_SKIP_SIGNATURE = 'skip_thought_signature_validator'
_SIGNATURE_TTL = 86400
_SIGNATURE_CACHE_LIMIT = 4096

# tool_call_id → (thoughtSignature, 时间戳)
_signature_cache: dict[str, tuple[str, float]] = {}


def _remember_signature(call_id: str, signature: str) -> None:
    if not call_id or not signature:
        return
    if len(_signature_cache) >= _SIGNATURE_CACHE_LIMIT:
        now = time.time()
        expired = [k for k, (_, ts) in _signature_cache.items() if now - ts >= _SIGNATURE_TTL]
        for key in expired:
            del _signature_cache[key]
        while len(_signature_cache) >= _SIGNATURE_CACHE_LIMIT:
            _signature_cache.pop(next(iter(_signature_cache)))
    _signature_cache[call_id] = (signature, time.time())


def _lookup_signature(call_id: str) -> str:
    entry = _signature_cache.get(call_id)
    if entry and (time.time() - entry[1]) < _SIGNATURE_TTL:
        return entry[0]
    return ''


class GeminiCodec(Codec):
    format = 'gemini'

    def build_request(self, request: IRRequest, upstream_model: str) -> dict[str, Any]:
        contents: list[dict[str, Any]] = []
        tool_names = _collect_tool_names(request.messages)

        for message in request.messages:
            converted = _build_content(message, tool_names)
            if converted:
                contents.append(converted)
        contents = _merge_same_role(contents)

        payload: dict[str, Any] = {
            'contents': contents,
            'generationConfig': _build_generation_config(request),
        }
        if request.system:
            payload['systemInstruction'] = {'parts': [{'text': request.system}]}
        if request.tools:
            payload['tools'] = [{
                'functionDeclarations': [
                    _build_declaration(t.name, t.description, t.parameters)
                    for t in request.tools
                ],
            }]
        return payload

    def parse_response(self, payload: dict[str, Any]) -> IRResponse:
        candidates = payload.get('candidates', [])
        candidate = candidates[0] if candidates else {}
        if not isinstance(candidate, dict):
            candidate = {}

        content = candidate.get('content', {})
        parts = content.get('parts', []) if isinstance(content, dict) else []
        blocks = _parse_parts(parts)

        finish = candidate.get('finishReason', 'STOP')
        has_tools = any(isinstance(b, ToolCallBlock) for b in blocks)
        if has_tools and finish == 'STOP':
            finish_reason = 'tool_calls'
        else:
            finish_reason = _FINISH_REASON_MAP.get(finish, 'stop')

        return IRResponse(
            id=gen_id('chatcmpl-'),
            model=payload.get('modelVersion', 'gemini'),
            blocks=blocks,
            finish_reason=finish_reason,
            usage=_parse_usage(payload.get('usageMetadata')),
        )

    def stream_decoder(self) -> StreamDecoder:
        return GeminiStreamDecoder()

    def upstream_url(self, base_url: str, model: str, stream: bool) -> str:
        # v1beta 是官方 SDK 默认版本；v1 端点不支持函数调用
        base = strip_version_suffix(base_url)
        if stream:
            return f'{base}/v1beta/models/{model}:streamGenerateContent?alt=sse'
        return f'{base}/v1beta/models/{model}:generateContent'

    def build_headers(self, api_key: str) -> dict[str, str]:
        headers = {'Content-Type': 'application/json'}
        if api_key.startswith('AIza'):
            headers['x-goog-api-key'] = api_key
        elif api_key:
            headers['Authorization'] = f'Bearer {api_key}'
        return headers


# ═══════════════════════════════════════════════════════════
#  流式解码器: Gemini SSE → IR 事件
# ═══════════════════════════════════════════════════════════


class GeminiStreamDecoder(StreamDecoder):
    """Gemini 流式每个 SSE data 都是一个完整的 GenerateContentResponse。"""

    def __init__(self):
        self._started = False
        self._ended = False
        self._tool_count = 0
        self._finish: str | None = None
        self._usage: IRUsage | None = None

    def decode(self, event_type: str, data: str) -> list[StreamEvent]:
        payload = parse_json(data)
        if payload is None:
            return []

        events: list[StreamEvent] = []
        if not self._started:
            self._started = True
            events.append(StreamStart(id=gen_id('chatcmpl-'), model=payload.get('modelVersion', '')))

        usage_meta = payload.get('usageMetadata')
        if isinstance(usage_meta, dict) and usage_meta:
            self._usage = _parse_usage(usage_meta)

        candidates = payload.get('candidates', [])
        candidate = candidates[0] if candidates else {}
        if not isinstance(candidate, dict):
            candidate = {}

        content = candidate.get('content', {})
        parts = content.get('parts', []) if isinstance(content, dict) else []
        for part in parts:
            if not isinstance(part, dict):
                continue
            if part.get('thought') and part.get('text'):
                events.append(ThinkingDelta(part['text']))
            elif 'text' in part and not part.get('thought'):
                if part['text']:
                    events.append(TextDelta(part['text']))
            elif 'functionCall' in part:
                fc = part['functionCall']
                index = self._tool_count
                self._tool_count += 1
                call_id = fc.get('id') or gen_id('call_')
                name = fc.get('name', '')
                arguments = dump_arguments(fc.get('args', {}))
                _remember_signature(call_id, part.get('thoughtSignature', ''))
                events.append(ToolCallStart(index=index, id=call_id, name=name))
                events.append(ToolCallDelta(index=index, arguments=arguments))
                events.append(ToolCallEnd(index=index, id=call_id, name=name, arguments=arguments))

        finish = candidate.get('finishReason')
        if finish:
            self._finish = finish
        return events

    def finalize(self) -> list[StreamEvent]:
        if self._ended:
            return []
        self._ended = True
        finish = self._finish or 'STOP'
        if self._tool_count and finish == 'STOP':
            finish_reason = 'tool_calls'
        else:
            finish_reason = _FINISH_REASON_MAP.get(finish, 'stop')
        return [StreamEnd(finish_reason=finish_reason, usage=self._usage)]


# ═══════════════════════════════════════════════════════════
#  请求 / 响应辅助
# ═══════════════════════════════════════════════════════════


def _collect_tool_names(messages: list[IRMessage]) -> dict[str, str]:
    """建立 tool_call_id → 工具名 的映射，供 functionResponse 使用。"""
    names: dict[str, str] = {}
    for message in messages:
        for block in message.blocks:
            if isinstance(block, ToolCallBlock) and block.id:
                names[block.id] = block.name
    return names


def _build_content(message: IRMessage, tool_names: dict[str, str]) -> dict[str, Any] | None:
    role = 'model' if message.role == 'assistant' else 'user'
    parts: list[dict[str, Any]] = []

    for block in message.blocks:
        if isinstance(block, ThinkingBlock):
            if block.text:
                parts.append({'text': block.text, 'thought': True})
        elif isinstance(block, TextBlock):
            if block.text:
                parts.append({'text': block.text})
        elif isinstance(block, ImageBlock):
            if block.data:
                parts.append({'inlineData': {'mimeType': block.media_type, 'data': block.data}})
        elif isinstance(block, ToolCallBlock):
            part: dict[str, Any] = {'functionCall': {
                'name': block.name,
                'args': parse_arguments_dict(block.arguments),
            }}
            # Gemini 3 强制要求回传 thoughtSignature，缓存未命中时用官方哨兵值兜底
            part['thoughtSignature'] = _lookup_signature(block.id) or _SKIP_SIGNATURE
            parts.append(part)
        elif isinstance(block, ToolResultBlock):
            parts.append({'functionResponse': {
                'name': block.name or tool_names.get(block.call_id, block.call_id),
                'response': _parse_response_payload(block.content),
            }})

    if not parts:
        return None
    return {'role': role, 'parts': parts}


def _parse_response_payload(content: str) -> Any:
    parsed = parse_json(content) if isinstance(content, str) else None
    if parsed is not None:
        return parsed
    return {'result': content} if content else {}


def _build_generation_config(request: IRRequest) -> dict[str, Any]:
    config: dict[str, Any] = {}
    if request.max_tokens:
        config['maxOutputTokens'] = request.max_tokens
    if request.temperature is not None:
        config['temperature'] = request.temperature
    if request.top_p is not None:
        config['topP'] = request.top_p
    if request.stop:
        config['stopSequences'] = request.stop
    return config


def _build_declaration(name: str, description: str, parameters: dict[str, Any]) -> dict[str, Any]:
    declaration: dict[str, Any] = {'name': name, 'description': description}
    if parameters:
        declaration['parameters'] = parameters
    return declaration


def _parse_parts(parts: list[Any]) -> list[Block]:
    blocks: list[Block] = []
    for part in parts:
        if not isinstance(part, dict):
            continue
        if part.get('thought') and 'text' in part:
            blocks.append(ThinkingBlock(text=part['text']))
        elif 'text' in part:
            blocks.append(TextBlock(text=part['text']))
        elif 'functionCall' in part:
            fc = part['functionCall']
            call_id = fc.get('id') or gen_id('call_')
            _remember_signature(call_id, part.get('thoughtSignature', ''))
            blocks.append(ToolCallBlock(
                id=call_id,
                name=fc.get('name', ''),
                arguments=dump_arguments(fc.get('args', {})),
            ))
    return blocks


def _parse_usage(meta: Any) -> IRUsage:
    if not isinstance(meta, dict):
        return IRUsage()
    prompt = meta.get('promptTokenCount', 0) or 0
    candidates = meta.get('candidatesTokenCount', 0) or 0
    thoughts = meta.get('thoughtsTokenCount', 0) or 0
    return IRUsage(
        input_tokens=prompt,
        output_tokens=candidates + thoughts,
        cached_tokens=meta.get('cachedContentTokenCount', 0) or 0,
        reasoning_tokens=thoughts,
    )


def _merge_same_role(contents: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """合并相邻同角色的 Gemini contents。"""
    if not contents:
        return contents
    merged = [contents[0]]
    for content in contents[1:]:
        if content['role'] == merged[-1]['role']:
            merged[-1]['parts'].extend(content['parts'])
        else:
            merged.append(content)
    return merged
