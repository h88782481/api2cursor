"""Cursor BYOK 思考内容显示兼容层。

这是针对 Cursor 未渲染 ``reasoning_content`` 的临时回程适配：
https://forum.cursor.com/t/165533

Cursor 修复后可删除本模块及 ``cursor.py``、``streaming.py`` 中的调用点。
"""

from __future__ import annotations

from typing import Any


_MIRROR_START = '> 💭 [](api2cursor-thinking) '
_MIRROR_END = '\n\n'


class CursorReasoningDisplay:
    """将 Chat 流中的 reasoning_content 同步镜像为 Markdown 引用文本。"""

    def __init__(self) -> None:
        self._open_choices: set[int] = set()
        self._metadata: dict[str, Any] = {}

    def rewrite_chunk(self, chunk: dict[str, Any]) -> list[dict[str, Any]]:
        """保留原始 chunk，并在它之前插入独立的可见文本 chunk。"""
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

            reasoning = delta.get('reasoning_content')
            if isinstance(reasoning, str) and reasoning:
                first = index not in self._open_choices
                mirrored = _format_reasoning(reasoning, first=first)
                self._open_choices.add(index)

                existing_content = delta.get('content')
                boundary = (
                    isinstance(existing_content, str) and bool(existing_content)
                ) or bool(delta.get('tool_calls')) or _has_finish_reason(raw_choice)
                if boundary:
                    mirrored += _MIRROR_END
                    self._open_choices.discard(index)
                display_choices.append(_display_choice(index, mirrored))
                continue

            existing_content = delta.get('content')
            boundary = (
                isinstance(existing_content, str) and bool(existing_content)
            ) or bool(delta.get('tool_calls')) or _has_finish_reason(raw_choice)
            if index in self._open_choices and boundary:
                display_choices.append(_closing_choice(index))
                self._open_choices.discard(index)

        if not display_choices:
            return [chunk]
        return [self._display_chunk(display_choices), chunk]

    def flush_chunk(self, model: str) -> dict[str, Any] | None:
        """在异常结束或仅有 reasoning 的流末尾关闭未完成的引用块。"""
        if not self._open_choices:
            return None
        choices = [_closing_choice(index) for index in sorted(self._open_choices)]
        self._open_choices.clear()
        self._metadata.setdefault('model', model)
        return self._display_chunk(choices)

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


def mirror_reasoning_response(response: dict[str, Any]) -> None:
    """为非流式 Chat 响应添加相同的可见文本镜像。"""
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
        if not isinstance(reasoning, str) or not reasoning:
            continue
        content = message.get('content')
        suffix = content if isinstance(content, str) else ''
        message['content'] = (
            f'{_format_reasoning(reasoning, first=True)}{_MIRROR_END}{suffix}'
        )


def restore_reasoning_mirrors(payload: dict[str, Any]) -> dict[str, Any]:
    """将 Cursor 回传的文本镜像还原为原生 reasoning_content。"""
    messages = payload.get('messages')
    if not isinstance(messages, list):
        return payload

    updated_messages: list[Any] | None = None
    for index, message in enumerate(messages):
        if not isinstance(message, dict) or message.get('role') != 'assistant':
            continue
        cleaned, restored = _restore_content(message.get('content'))
        if restored is None:
            continue
        original_reasoning = message.get('reasoning_content')
        if isinstance(original_reasoning, str) and original_reasoning:
            restored = original_reasoning
        if updated_messages is None:
            updated_messages = list(messages)
        updated_message = dict(message)
        updated_message['content'] = cleaned
        updated_message['reasoning_content'] = restored
        updated_messages[index] = updated_message

    if updated_messages is None:
        return payload
    updated_payload = dict(payload)
    updated_payload['messages'] = updated_messages
    return updated_payload


def _format_reasoning(text: str, *, first: bool) -> str:
    prefix = _MIRROR_START if first else ''
    return prefix + text.replace('\n', '\n> ')


def _restore_content(content: Any) -> tuple[Any, str | None]:
    if isinstance(content, str):
        return _restore_text(content)
    if not isinstance(content, list):
        return content, None

    for index, part in enumerate(content):
        if not isinstance(part, dict) or not isinstance(part.get('text'), str):
            continue
        cleaned, restored = _restore_text(part['text'])
        if restored is None:
            continue
        updated = list(content)
        if cleaned:
            updated_part = dict(part)
            updated_part['text'] = cleaned
            updated[index] = updated_part
        else:
            updated.pop(index)
        return updated, restored
    return content, None


def _restore_text(text: str) -> tuple[str, str | None]:
    if not text.startswith(_MIRROR_START):
        return text, None
    end = text.find(_MIRROR_END, len(_MIRROR_START))
    if end < 0:
        return text, None
    mirrored = text[len(_MIRROR_START):end]
    reasoning = mirrored.replace('\n> ', '\n')
    return text[end + len(_MIRROR_END):], reasoning


def _choice_index(choice: dict[str, Any]) -> int:
    value = choice.get('index', 0)
    return value if isinstance(value, int) else 0


def _has_finish_reason(choice: dict[str, Any]) -> bool:
    value = choice.get('finish_reason')
    return value not in (None, '', {'reason': None})


def _closing_choice(index: int) -> dict[str, Any]:
    return _display_choice(index, _MIRROR_END)


def _display_choice(index: int, content: str) -> dict[str, Any]:
    return {
        'index': index,
        'delta': {'content': content},
        'finish_reason': None,
    }
