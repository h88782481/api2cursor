"""轻量 Thinking 缓存

纯内存缓存，在多轮对话中保存和恢复 thinking/reasoning 内容。
解决 Cursor 不会把 thinking 内容回传给 API 的问题，
某些模型（如推理模型）在缺少历史 thinking 时表现会下降。
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import time
from typing import Any

logger = logging.getLogger(__name__)

_THINK_RE = re.compile(r'<think>.*?</think>', re.DOTALL)
_UNCLOSED_THINK_RE = re.compile(r'<think>.*$', re.DOTALL)
_TOOL_ID_RE = re.compile(r'[^a-zA-Z0-9_-]')
_TTL = 86400  # 24 hours


class ThinkingCache:
    """纯内存 thinking 缓存，TTL 2 小时。"""

    def __init__(self):
        self._store: dict[str, tuple[str, float]] = {}

    def inject(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """遍历 assistant 消息，缺少 reasoning_content 时从缓存注入。"""
        sid = self._session_id(messages)
        if not sid:
            return messages

        now = time.time()
        for msg in messages:
            if msg.get('role') != 'assistant':
                continue
            if msg.get('reasoning_content'):
                continue
            key = sid + ':' + self._message_hash(msg)
            entry = self._store.get(key)
            if entry and (now - entry[1]) < _TTL:
                msg['reasoning_content'] = entry[0]
                logger.debug('已从缓存注入 thinking (%d 字符)', len(entry[0]))

        return messages

    def store_from_response(
        self,
        messages: list[dict[str, Any]],
        reasoning_content: str,
    ) -> None:
        """将响应中的 thinking 内容存入缓存。"""
        if not reasoning_content:
            return
        sid = self._session_id(messages)
        if not sid:
            return

        fake_msg: dict[str, Any] = {'role': 'assistant', 'content': '', 'tool_calls': []}
        key = sid + ':' + self._message_hash(fake_msg)
        self._store[key] = (reasoning_content, time.time())
        self._cleanup()

    def store_assistant_thinking(
        self,
        messages: list[dict[str, Any]],
        assistant_msg: dict[str, Any],
    ) -> None:
        """从完整的 assistant 消息中提取并缓存 thinking。"""
        rc = assistant_msg.get('reasoning_content', '')
        if not rc:
            return
        sid = self._session_id(messages)
        if not sid:
            return
        key = sid + ':' + self._message_hash(assistant_msg)
        self._store[key] = (rc, time.time())
        self._cleanup()

    def _session_id(self, messages: list[dict[str, Any]]) -> str:
        first_user = ''
        first_assistant = ''
        for msg in messages:
            role = msg.get('role', '')
            if role in ('system', 'developer'):
                continue
            if role == 'user' and not first_user:
                first_user = self._normalize_content(
                    msg.get('content', '')
                )
            elif role == 'assistant' and not first_assistant:
                first_assistant = self._normalize_content(
                    msg.get('content', '')
                )
            if first_user and first_assistant:
                break

        if not first_user or not first_assistant:
            return ''

        raw = first_user + '|' + first_assistant
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    def _message_hash(self, msg: dict[str, Any]) -> str:
        content = self._normalize_content(msg.get('content', ''))
        tool_ids = sorted(
            self._normalize_tool_id(tc.get('id', ''))
            for tc in msg.get('tool_calls', [])
            if isinstance(tc, dict)
        )
        raw = json.dumps({'c': content, 't': tool_ids}, ensure_ascii=False)
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    @staticmethod
    def _normalize_content(content: Any) -> str:
        if isinstance(content, list):
            parts = []
            for p in content:
                if isinstance(p, dict) and p.get('type') == 'text':
                    parts.append(p.get('text', ''))
                elif isinstance(p, str):
                    parts.append(p)
            text = '\n'.join(parts)
        elif isinstance(content, str):
            text = content
        else:
            text = str(content) if content else ''
        text = _THINK_RE.sub('', text)
        text = _UNCLOSED_THINK_RE.sub('', text)
        return text.strip()

    @staticmethod
    def _normalize_tool_id(tid: str) -> str:
        return _TOOL_ID_RE.sub('', tid)

    def _cleanup(self) -> None:
        """惰性清理过期条目（每 100 次写入触发一次全量扫描）。"""
        if len(self._store) < 100:
            return
        now = time.time()
        expired = [k for k, (_, ts) in self._store.items() if (now - ts) >= _TTL]
        for k in expired:
            del self._store[k]


thinking_cache = ThinkingCache()
