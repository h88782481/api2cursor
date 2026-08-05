"""Cursor BYOK 思考内容显示兼容层。

这是针对 Cursor 未渲染 ``reasoning_content`` 的临时回程适配：
https://forum.cursor.com/t/165533

Cursor 修复后可删除本模块及 ``builder.py``、``cursor.py``、``gateway.py``、
``streaming.py`` 中的调用点。
"""

from __future__ import annotations

import base64
import copy
import hashlib
import json
import re
from typing import Any

from ..protocol import WireProtocol


_THINKING_START = '<details>\n<summary>Thinking</summary>\n\n> '
_THINKING_END = '\n\n</details>\n\n'
_STATE_SUMMARY = 'encrypted_content'
_MAX_STATE_BYTES = 4 * 1024 * 1024
_STATE_PATTERN = re.compile(
    r'<details>\r?\n<summary>encrypted_content</summary>\r?\n\r?\n'
    r'```json\r?\n(\{.*?\})\r?\n```\r?\n\r?\n</details>(?:\r?\n){0,2}',
    re.DOTALL,
)
_THINKING_PATTERN = re.compile(
    r'<details>\r?\n<summary>Thinking</summary>\r?\n\r?\n'
    r'(.*?)\r?\n\r?\n</details>(?:\r?\n){0,2}',
    re.DOTALL,
)


class CursorReasoningDisplay:
    """将流式思考镜像为折叠文本，并附带 provider 原生状态。"""

    def __init__(self) -> None:
        self._open_choices: set[int] = set()
        self._metadata: dict[str, Any] = {}
        self._pending_states: dict[int, dict[str, Any]] = {}
        self._reasoning_text: dict[int, str] = {}
        self._seen_response_reasoning_ids: set[str] = set()

    def capture_upstream(self, protocol: str, chunk: dict[str, Any]) -> None:
        """捕获不会经过 Chat IR 流事件的 provider 原生推理状态。"""
        if protocol == 'chat':
            self._capture_chat_state(chunk)
            return
        if protocol == 'messages':
            self._capture_messages_state(chunk)
            return
        if protocol != 'responses':
            return
        items: list[Any] = []
        if chunk.get('type') == 'response.output_item.done':
            items.append(chunk.get('item'))
        elif chunk.get('type') == 'response.completed':
            response = chunk.get('response')
            if isinstance(response, dict) and isinstance(response.get('output'), list):
                items.extend(response['output'])
        for item in items:
            if not isinstance(item, dict) or item.get('type') != 'reasoning':
                continue
            encrypted = item.get('encrypted_content')
            item_id = item.get('id')
            if (
                not isinstance(encrypted, str)
                or not encrypted
                or not isinstance(item_id, str)
                or not item_id
                or item_id in self._seen_response_reasoning_ids
            ):
                continue
            self._seen_response_reasoning_ids.add(item_id)
            self._pending_states[0] = {
                'provider': 'responses',
                'item': copy.deepcopy(item),
            }

    def _capture_chat_state(self, chunk: dict[str, Any]) -> None:
        choices = chunk.get('choices')
        if not isinstance(choices, list):
            return
        for choice in choices:
            if not isinstance(choice, dict):
                continue
            index = _choice_index(choice)
            delta = choice.get('delta')
            if not isinstance(delta, dict):
                continue
            value = delta.get('reasoning_content')
            if not isinstance(value, str):
                value = delta.get('reasoning')
            if not isinstance(value, str) or not value:
                continue
            accumulated = self._reasoning_text.get(index, '') + value
            self._reasoning_text[index] = accumulated
            self._pending_states[index] = {
                'provider': 'chat',
                'reasoning_content': accumulated,
            }

    def _capture_messages_state(self, chunk: dict[str, Any]) -> None:
        if chunk.get('type') != 'content_block_delta':
            return
        delta = chunk.get('delta')
        if not isinstance(delta, dict):
            return
        state = dict(self._pending_states.get(0) or {'provider': 'messages'})
        if delta.get('type') == 'thinking_delta':
            value = delta.get('thinking')
            if isinstance(value, str) and value:
                accumulated = self._reasoning_text.get(0, '') + value
                self._reasoning_text[0] = accumulated
                state['reasoning'] = accumulated
        elif delta.get('type') == 'signature_delta':
            signature = delta.get('signature')
            if isinstance(signature, str) and signature:
                state['signature'] = str(state.get('signature', '')) + signature
        if len(state) > 1:
            self._pending_states[0] = state

    def rewrite_chunk(self, chunk: dict[str, Any]) -> list[dict[str, Any]]:
        """保留原始 chunk，并在它之前插入独立的可见文本 chunk。

        可见思考块可在正文/工具调用边界关闭，但 provider 原生状态载体
        只在 finish / flush 提交，避免工具调用过早弹出未完成状态。
        """
        self._remember_metadata(chunk)
        choices = chunk.get('choices')
        if not isinstance(choices, list):
            return [chunk]

        display_choices: list[dict[str, Any]] = []
        for raw_choice in choices:
            if not isinstance(raw_choice, dict):
                continue
            index = _choice_index(raw_choice)
            raw_delta = raw_choice.get('delta')
            delta = raw_delta if isinstance(raw_delta, dict) else {}
            finished = _has_finish_reason(raw_choice)
            existing_content = delta.get('content')
            has_content = isinstance(existing_content, str) and bool(existing_content)
            has_tools = bool(delta.get('tool_calls'))
            close_display = has_content or has_tools or finished

            reasoning = delta.get('reasoning_content')
            if isinstance(reasoning, str) and reasoning:
                first = index not in self._open_choices
                mirrored = _format_reasoning(reasoning, first=first)
                self._open_choices.add(index)
                if close_display:
                    mirrored += _THINKING_END
                    self._open_choices.discard(index)
                if finished:
                    mirrored += self._take_state(index)
                display_choices.append(_display_choice(index, mirrored))
                continue

            if not close_display:
                continue

            prefix = ''
            if index in self._open_choices:
                prefix = _THINKING_END
                self._open_choices.discard(index)
            # 工具调用/正文只关闭可见块；状态延后到 finish，等待晚到的
            # signature / encrypted_content / response.output_item.done。
            state = self._take_state(index) if finished else ''
            if prefix or state:
                display_choices.append(_display_choice(index, prefix + state))

        if not display_choices:
            return [chunk]
        return [self._display_chunk(display_choices), chunk]

    def flush_chunk(self, model: str) -> dict[str, Any] | None:
        """在流末尾关闭未完成思考块，并提交尚未写出的原生状态。"""
        indexes = self._open_choices | set(self._pending_states)
        if not indexes:
            return None
        choices = [
            _display_choice(
                index,
                (_THINKING_END if index in self._open_choices else '')
                + self._take_state(index),
            )
            for index in sorted(indexes)
        ]
        self._open_choices.clear()
        self._metadata.setdefault('model', model)
        return self._display_chunk(choices)

    def _take_state(self, index: int) -> str:
        state = self._pending_states.pop(index, None)
        return encode_reasoning_state(state) if state else ''

    def _remember_metadata(self, chunk: dict[str, Any]) -> None:
        for key in ('id', 'object', 'created', 'model'):
            if key in chunk:
                self._metadata[key] = chunk[key]

    def _display_chunk(self, choices: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            'id': self._metadata.get('id', 'chatcmpl-reasoning-display'),
            'object': self._metadata.get('object', 'chat.completion.chunk'),
            'created': self._metadata.get('created', 0),
            'model': self._metadata.get('model', ''),
            'choices': choices,
        }


def extract_response_reasoning_states(
    response: dict[str, Any],
    upstream_protocol: WireProtocol,
) -> dict[int, dict[str, Any]]:
    """从非流式 IR 响应中提取 provider 原生推理状态。"""
    states: dict[int, dict[str, Any]] = {}
    for choice in response.get('choices', []):
        if not isinstance(choice, dict):
            continue
        choice_index = _choice_index(choice)
        content = choice.get('message', {}).get('content', [])
        for part in content if isinstance(content, list) else []:
            if not isinstance(part, dict) or part.get('type') != 'reasoning':
                continue
            encrypted = part.get('signature')
            metadata = part.get('provider_metadata', {})
            reasoning = part.get('reasoning')
            if upstream_protocol == 'chat' and isinstance(reasoning, str) and reasoning:
                states[choice_index] = {
                    'provider': 'chat',
                    'reasoning_content': reasoning,
                }
            elif upstream_protocol == 'messages' and (
                isinstance(reasoning, str) and reasoning
                or isinstance(encrypted, str) and encrypted
            ):
                states[choice_index] = {
                    'provider': 'messages',
                    'reasoning': reasoning or '',
                    'signature': encrypted or '',
                }
            elif upstream_protocol == 'responses':
                item_id = (
                    metadata.get('responses_reasoning_id')
                    if isinstance(metadata, dict)
                    else None
                )
                if not isinstance(encrypted, str) or not encrypted or not item_id:
                    continue
                item: dict[str, Any] = {
                    'id': item_id,
                    'type': 'reasoning',
                    'summary': metadata.get('responses_reasoning_summary', []),
                    'encrypted_content': encrypted,
                }
                if 'responses_reasoning_content' in metadata:
                    item['content'] = metadata['responses_reasoning_content']
                states[choice_index] = {'provider': 'responses', 'item': item}
            break
    return states


def mirror_reasoning_response(
    response: dict[str, Any],
    states: dict[int, dict[str, Any]] | None = None,
) -> None:
    """为非流式 Chat 响应添加思考显示和无数据库状态载体。"""
    states = states or {}
    choices = response.get('choices')
    if not isinstance(choices, list):
        return
    for choice in choices:
        if not isinstance(choice, dict):
            continue
        message = choice.get('message')
        if not isinstance(message, dict):
            continue
        reasoning = message.get('reasoning_content')
        content = message.get('content')
        suffix = content if isinstance(content, str) else ''
        visible = ''
        if isinstance(reasoning, str) and reasoning:
            visible = f'{_format_reasoning(reasoning, first=True)}{_THINKING_END}'
        state = states.get(_choice_index(choice))
        carrier = encode_reasoning_state(state) if state else ''
        if visible or carrier:
            message['content'] = f'{visible}{carrier}{suffix}'


def restore_reasoning_carriers(
    payload: dict[str, Any],
) -> tuple[dict[str, Any], dict[int, dict[str, Any]]]:
    """剥离 Cursor 回传的折叠块，并返回每个 assistant 的原生状态。"""
    messages = payload.get('messages')
    if not isinstance(messages, list):
        return payload, {}

    updated_messages: list[Any] | None = None
    carriers: dict[int, dict[str, Any]] = {}
    assistant_index = -1
    for index, message in enumerate(messages):
        if not isinstance(message, dict) or message.get('role') != 'assistant':
            continue
        assistant_index += 1
        cleaned, restored, state, changed = _restore_content(message.get('content'))
        if not changed:
            continue
        original_reasoning = message.get('reasoning_content')
        if isinstance(original_reasoning, str) and original_reasoning:
            restored = original_reasoning
        if updated_messages is None:
            updated_messages = list(messages)
        updated_message = dict(message)
        updated_message['content'] = cleaned
        if restored:
            updated_message['reasoning_content'] = restored
        updated_messages[index] = updated_message
        if state:
            carriers[assistant_index] = state

    if updated_messages is None:
        return payload, carriers
    updated_payload = dict(payload)
    updated_payload['messages'] = updated_messages
    return updated_payload, carriers


def restore_ir_reasoning_states(
    request: dict[str, Any],
    carriers: dict[int, dict[str, Any]],
    upstream_protocol: WireProtocol | None = None,
) -> None:
    """把客户端携带的状态恢复到 IR reasoning part，供 Responses 回编。"""
    assistant_index = -1
    for message in request.get('messages', []):
        if not isinstance(message, dict) or message.get('role') != 'assistant':
            continue
        assistant_index += 1
        state = carriers.get(assistant_index)
        if not state:
            continue
        provider = state.get('provider')
        if upstream_protocol is not None and provider != upstream_protocol:
            continue
        content = message.setdefault('content', [])
        reasoning = next(
            (
                part for part in content
                if isinstance(part, dict) and part.get('type') == 'reasoning'
            ),
            None,
        )
        if reasoning is None:
            reasoning = {'type': 'reasoning'}
            content.insert(0, reasoning)

        if provider == 'chat':
            value = state.get('reasoning_content')
            if isinstance(value, str) and value:
                reasoning['reasoning'] = value
        elif provider == 'messages':
            value = state.get('reasoning')
            signature = state.get('signature')
            if isinstance(value, str) and value:
                reasoning['reasoning'] = value
            if isinstance(signature, str) and signature:
                reasoning['signature'] = signature
        elif provider == 'responses':
            _restore_responses_state(reasoning, state)


def _restore_responses_state(
    reasoning: dict[str, Any],
    state: dict[str, Any],
) -> None:
    item = state.get('item')
    if not isinstance(item, dict):
        return
    encrypted = item.get('encrypted_content')
    item_id = item.get('id')
    if not isinstance(encrypted, str) or not encrypted or not item_id:
        return
    reasoning['signature'] = encrypted
    metadata = dict(reasoning.get('provider_metadata') or {})
    metadata['responses_reasoning_id'] = item_id
    metadata['responses_reasoning_summary'] = item.get('summary', [])
    if 'content' in item:
        metadata['responses_reasoning_content'] = item['content']
    reasoning['provider_metadata'] = metadata


def encode_reasoning_state(state: dict[str, Any]) -> str:
    """把原生状态编码为可由 Cursor 原样携带的折叠 JSON 块。"""
    payload = json.dumps(
        state,
        ensure_ascii=False,
        separators=(',', ':'),
        sort_keys=True,
    ).encode('utf-8')
    if len(payload) > _MAX_STATE_BYTES:
        return ''
    encoded = base64.urlsafe_b64encode(payload).decode('ascii').rstrip('=')
    envelope = {
        'v': 1,
        'encoding': 'base64url',
        'sha256': hashlib.sha256(payload).hexdigest(),
        'payload': encoded,
    }
    body = json.dumps(envelope, ensure_ascii=False, indent=2)
    return (
        f'<details>\n<summary>{_STATE_SUMMARY}</summary>\n\n'
        f'```json\n{body}\n```\n\n</details>\n\n'
    )


def _format_reasoning(text: str, *, first: bool) -> str:
    prefix = _THINKING_START if first else ''
    return prefix + text.replace('\n', '\n> ')


def _restore_content(
    content: Any,
) -> tuple[Any, str | None, dict[str, Any] | None, bool]:
    if isinstance(content, str):
        return _restore_text(content)
    if not isinstance(content, list):
        return content, None, None, False

    updated: list[Any] = []
    restored: str | None = None
    state: dict[str, Any] | None = None
    changed = False
    for part in content:
        if not isinstance(part, dict) or not isinstance(part.get('text'), str):
            updated.append(part)
            continue
        cleaned, part_reasoning, part_state, part_changed = _restore_text(part['text'])
        changed = changed or part_changed
        restored = restored or part_reasoning
        state = state or part_state
        if not part_changed:
            updated.append(part)
            continue
        if cleaned:
            updated_part = dict(part)
            updated_part['text'] = cleaned
            updated.append(updated_part)
    return updated, restored, state, changed


def _restore_text(
    text: str,
) -> tuple[str, str | None, dict[str, Any] | None, bool]:
    reasoning_parts: list[str] = []
    state: dict[str, Any] | None = None
    changed = False

    def _strip_thinking(match: re.Match[str]) -> str:
        nonlocal changed
        changed = True
        reasoning_parts.append(_unquote_reasoning(match.group(1)))
        return ''

    text = _THINKING_PATTERN.sub(_strip_thinking, text)

    def _strip_state(match: re.Match[str]) -> str:
        nonlocal changed, state
        changed = True
        decoded = _decode_reasoning_state(match.group(1))
        if decoded is not None:
            state = decoded
        return ''

    text = _STATE_PATTERN.sub(_strip_state, text)
    reasoning = '\n'.join(part for part in reasoning_parts if part) or None
    return text, reasoning, state, changed


def _decode_reasoning_state(raw: str) -> dict[str, Any] | None:
    try:
        envelope = json.loads(raw)
        if envelope.get('v') != 1 or envelope.get('encoding') != 'base64url':
            return None
        encoded = envelope['payload']
        if not isinstance(encoded, str) or len(encoded) > _MAX_STATE_BYTES * 2:
            return None
        padding = '=' * (-len(encoded) % 4)
        payload = base64.b64decode(
            encoded + padding,
            altchars=b'-_',
            validate=True,
        )
        if len(payload) > _MAX_STATE_BYTES:
            return None
        if hashlib.sha256(payload).hexdigest() != envelope.get('sha256'):
            return None
        state = json.loads(payload)
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None
    if not isinstance(state, dict):
        return None
    provider = state.get('provider')
    if provider == 'chat':
        if not isinstance(state.get('reasoning_content'), str):
            return None
    elif provider == 'messages':
        if not any(
            isinstance(state.get(key), str) and state.get(key)
            for key in ('reasoning', 'signature')
        ):
            return None
    elif provider == 'responses':
        item = state.get('item')
        if (
            not isinstance(item, dict)
            or item.get('type') != 'reasoning'
            or not isinstance(item.get('id'), str)
            or not isinstance(item.get('encrypted_content'), str)
            or not item.get('encrypted_content')
        ):
            return None
    else:
        return None
    return state


def _unquote_reasoning(value: str) -> str:
    if value.startswith('> '):
        value = value[2:]
    elif value.startswith('>'):
        value = value[1:]
    return value.replace('\n> ', '\n').replace('\n>', '\n')


def _choice_index(choice: dict[str, Any]) -> int:
    value = choice.get('index', 0)
    return value if isinstance(value, int) else 0


def _has_finish_reason(choice: dict[str, Any]) -> bool:
    value = choice.get('finish_reason')
    return value not in (None, '', {'reason': None})


def _display_choice(index: int, content: str) -> dict[str, Any]:
    return {
        'index': index,
        'delta': {'content': content},
        'finish_reason': None,
    }
