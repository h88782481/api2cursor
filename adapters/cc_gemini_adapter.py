"""OpenAI Chat Completions ↔ Gemini Contents 格式转换

将 CC 格式请求转换为 Gemini generateContent 格式，
并将 Gemini 响应转换回 CC 格式。仅支持出站方向（CC → Gemini → CC）。
"""

from __future__ import annotations

import json
import logging
from typing import Any

from utils.http import gen_id

JsonDict = dict[str, Any]

logger = logging.getLogger(__name__)

_FINISH_REASON_MAP = {
    'STOP': 'stop',
    'MAX_TOKENS': 'length',
    'SAFETY': 'content_filter',
    'RECITATION': 'content_filter',
}


# ═══════════════════════════════════════════════════════════
#  请求转换: CC → Gemini generateContent
# ═══════════════════════════════════════════════════════════


def cc_to_gemini_request(payload: JsonDict) -> JsonDict:
    """将 CC 请求转换为 Gemini generateContent 请求。"""
    messages = payload.get('messages', [])
    system_parts: list[str] = []
    contents: list[JsonDict] = []

    for msg in messages:
        role = msg.get('role', '')
        if role in ('system', 'developer'):
            system_parts.append(_flatten_text(msg.get('content', '')))
            continue
        converted = _convert_message(msg)
        if converted:
            contents.append(converted)

    contents = _merge_same_role(contents)

    result: JsonDict = {
        'contents': contents,
        'generationConfig': _build_generation_config(payload),
    }

    if system_parts:
        result['systemInstruction'] = {
            'parts': [{'text': '\n\n'.join(system_parts)}],
        }

    tools = _convert_tools(payload.get('tools'))
    if tools:
        result['tools'] = tools

    return result


# ═══════════════════════════════════════════════════════════
#  非流式响应转换: Gemini → CC
# ═══════════════════════════════════════════════════════════


def gemini_to_cc_response(data: JsonDict, request_id: str | None = None) -> JsonDict:
    """将 Gemini generateContent 响应转换为 CC 响应。"""
    request_id = request_id or gen_id('chatcmpl-')
    candidates = data.get('candidates', [])
    candidate = candidates[0] if candidates else {}

    content_text, reasoning_text, tool_calls = _extract_parts(
        candidate.get('content', {}).get('parts', [])
    )

    finish = candidate.get('finishReason', 'STOP')
    if tool_calls and finish == 'STOP':
        finish_reason = 'tool_calls'
    else:
        finish_reason = _FINISH_REASON_MAP.get(finish, 'stop')

    message: JsonDict = {'role': 'assistant', 'content': content_text or None}
    if reasoning_text:
        message['reasoning_content'] = reasoning_text
    if tool_calls:
        message['tool_calls'] = tool_calls

    usage = _convert_usage(data.get('usageMetadata', {}))

    return {
        'id': request_id,
        'object': 'chat.completion',
        'model': data.get('modelVersion', 'gemini'),
        'choices': [{'index': 0, 'message': message, 'finish_reason': finish_reason}],
        'usage': usage,
    }


# ═══════════════════════════════════════════════════════════
#  流式转换: Gemini SSE → CC chunks
# ═══════════════════════════════════════════════════════════


class GeminiStreamConverter:
    """将 Gemini SSE chunk 逐个转换为 CC chunk。

    Gemini 流式每个 SSE data 是一个完整的 GenerateContentResponse，
    包含 candidates[0].content.parts。
    """

    def __init__(self, request_id: str | None = None):
        self._id = request_id or gen_id('chatcmpl-')
        self._tool_call_index = 0
        self._started = False

    def process_chunk(self, data: JsonDict) -> list[JsonDict]:
        """处理一个 Gemini SSE chunk，返回 CC chunk 列表。"""
        results: list[JsonDict] = []
        candidates = data.get('candidates', [])
        if not candidates:
            return results

        candidate = candidates[0]
        parts = candidate.get('content', {}).get('parts', [])

        if not self._started:
            self._started = True
            results.append(self._make_chunk({'role': 'assistant', 'content': ''}))

        for part in parts:
            if part.get('thought') and part.get('text'):
                results.append(self._make_chunk({'reasoning_content': part['text']}))
            elif 'text' in part and not part.get('thought'):
                results.append(self._make_chunk({'content': part['text']}))
            elif 'functionCall' in part:
                fc = part['functionCall']
                results.append(self._make_chunk({'tool_calls': [{
                    'index': self._tool_call_index,
                    'id': fc.get('id') or gen_id('call_'),
                    'type': 'function',
                    'function': {
                        'name': fc.get('name', ''),
                        'arguments': json.dumps(fc.get('args', {}), ensure_ascii=False),
                    },
                }]}))
                self._tool_call_index += 1

        finish = candidate.get('finishReason')
        if finish:
            has_tools = self._tool_call_index > 0
            if has_tools and finish == 'STOP':
                fr = 'tool_calls'
            else:
                fr = _FINISH_REASON_MAP.get(finish, 'stop')
            chunk = self._make_chunk({}, finish_reason=fr)
            usage_meta = data.get('usageMetadata')
            if usage_meta:
                chunk['usage'] = _convert_usage(usage_meta)
            results.append(chunk)

        return results

    def _make_chunk(self, delta: JsonDict, finish_reason: str | None = None) -> JsonDict:
        choice: JsonDict = {'index': 0, 'delta': delta}
        if finish_reason:
            choice['finish_reason'] = finish_reason
        return {
            'id': self._id,
            'object': 'chat.completion.chunk',
            'model': 'gemini',
            'choices': [choice],
        }


# ═══════════════════════════════════════════════════════════
#  请求转换辅助
# ═══════════════════════════════════════════════════════════


def _convert_message(msg: JsonDict) -> JsonDict | None:
    """将单条 CC 消息转为 Gemini Content。"""
    role = msg.get('role', '')
    gemini_role = 'model' if role == 'assistant' else 'user'
    parts: list[JsonDict] = []

    if role == 'tool':
        return {
            'role': 'user',
            'parts': [{
                'functionResponse': {
                    'name': msg.get('name', msg.get('tool_call_id', '')),
                    'response': _parse_json_safe(msg.get('content', '')),
                },
            }],
        }

    if msg.get('reasoning_content'):
        parts.append({'text': msg['reasoning_content'], 'thought': True})

    content = msg.get('content')
    if isinstance(content, str) and content:
        parts.append({'text': content})
    elif isinstance(content, list):
        for block in content:
            if not isinstance(block, dict):
                continue
            if block.get('type') == 'text':
                parts.append({'text': block.get('text', '')})
            elif block.get('type') == 'image_url':
                img = _convert_image_part(block)
                if img:
                    parts.append(img)

    for tc in msg.get('tool_calls', []):
        func = tc.get('function', {})
        parts.append({
            'functionCall': {
                'name': func.get('name', ''),
                'args': _parse_json_safe(func.get('arguments', '{}')),
            },
        })

    if not parts:
        return None
    return {'role': gemini_role, 'parts': parts}


def _convert_image_part(block: JsonDict) -> JsonDict | None:
    """将 OpenAI image_url 转为 Gemini inlineData。"""
    url_data = block.get('image_url', {})
    url = url_data.get('url', '') if isinstance(url_data, dict) else str(url_data)
    if url.startswith('data:'):
        media_type, _, b64 = url.partition(';base64,')
        return {'inlineData': {
            'mimeType': media_type.replace('data:', '') or 'image/png',
            'data': b64,
        }}
    return None


def _build_generation_config(payload: JsonDict) -> JsonDict:
    """从 CC payload 构建 Gemini generationConfig。"""
    config: JsonDict = {}
    if 'max_tokens' in payload:
        config['maxOutputTokens'] = payload['max_tokens']
    elif 'max_completion_tokens' in payload:
        config['maxOutputTokens'] = payload['max_completion_tokens']
    if 'temperature' in payload:
        config['temperature'] = payload['temperature']
    if 'top_p' in payload:
        config['topP'] = payload['top_p']
    stop = payload.get('stop')
    if stop:
        config['stopSequences'] = stop if isinstance(stop, list) else [stop]
    return config


def _convert_tools(tools: Any) -> list[JsonDict] | None:
    """将 CC tools 转为 Gemini functionDeclarations。"""
    if not isinstance(tools, list) or not tools:
        return None
    declarations: list[JsonDict] = []
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        func = tool.get('function', tool) if tool.get('type') == 'function' else tool
        if 'name' not in func:
            continue
        decl: JsonDict = {
            'name': func.get('name', ''),
            'description': func.get('description', ''),
        }
        params = func.get('parameters')
        if params:
            decl['parameters'] = params
        declarations.append(decl)
    if not declarations:
        return None
    return [{'functionDeclarations': declarations}]


# ═══════════════════════════════════════════════════════════
#  响应转换辅助
# ═══════════════════════════════════════════════════════════


def _extract_parts(parts: list[Any]) -> tuple[str, str, list[JsonDict]]:
    """从 Gemini parts 中提取文本、思考内容和工具调用。"""
    text = ''
    reasoning = ''
    tool_calls: list[JsonDict] = []

    for part in parts:
        if not isinstance(part, dict):
            continue
        if part.get('thought') and 'text' in part:
            reasoning += part['text']
        elif 'text' in part:
            text += part['text']
        elif 'functionCall' in part:
            fc = part['functionCall']
            tool_calls.append({
                'index': len(tool_calls),
                'id': fc.get('id') or gen_id('call_'),
                'type': 'function',
                'function': {
                    'name': fc.get('name', ''),
                    'arguments': json.dumps(fc.get('args', {}), ensure_ascii=False),
                },
            })

    return text, reasoning, tool_calls


def _convert_usage(meta: JsonDict) -> JsonDict:
    """将 Gemini usageMetadata 转为 CC usage。"""
    prompt = meta.get('promptTokenCount', 0)
    candidates = meta.get('candidatesTokenCount', 0)
    thoughts = meta.get('thoughtsTokenCount', 0)
    completion = candidates + thoughts
    return {
        'prompt_tokens': prompt,
        'completion_tokens': completion,
        'total_tokens': prompt + completion,
    }


def _merge_same_role(contents: list[JsonDict]) -> list[JsonDict]:
    """合并相邻同角色的 Gemini contents。"""
    if not contents:
        return contents
    merged = [contents[0]]
    for c in contents[1:]:
        if c['role'] == merged[-1]['role']:
            merged[-1]['parts'].extend(c['parts'])
        else:
            merged.append(c)
    return merged


def _flatten_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return '\n'.join(
            p.get('text', '') if isinstance(p, dict) else str(p)
            for p in content
        )
    return str(content)


def _parse_json_safe(text: Any) -> Any:
    if not isinstance(text, str):
        return text if text is not None else {}
    try:
        return json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return {'result': text} if text else {}
