"""思考内容兼容层

三块职责：
1. `<think>` 标签提取 —— 部分上游（如 DeepSeek 系）把思考内容直接混在正文里，
   这里在非流式（IRResponse）和流式（IR 事件）两个层面统一提取为 Thinking 语义。
2. ThinkingCache —— Cursor 不会把 assistant 的思考内容回传给 API，推理模型缺少
   历史 thinking 时表现下降；这里用纯内存缓存做多轮注入。
3. /v1/messages 透传链路的 reasoning_content → thinking block 注入。
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import time
from typing import Any

from ..core.ir import (
    IRMessage,
    IRRequest,
    IRResponse,
    StreamEvent,
    TextBlock,
    TextDelta,
    ThinkingBlock,
    ThinkingDelta,
    ToolCallStart,
)

logger = logging.getLogger(__name__)

_THINK_RE = re.compile(r'<think>(.*?)</think>', re.DOTALL)
_STRIP_THINK_RE = re.compile(r'<think>.*?</think>', re.DOTALL)
_STRIP_UNCLOSED_THINK_RE = re.compile(r'<think>.*$', re.DOTALL)
_TOOL_ID_RE = re.compile(r'[^a-zA-Z0-9_-]')

_OPEN_TAG = '<think>'
_CLOSE_TAG = '</think>'


# ═══════════════════════════════════════════════════════════
#  <think> 标签提取
# ═══════════════════════════════════════════════════════════


def extract_think_from_text(content: str) -> tuple[str | None, str | None]:
    """从文本中提取 <think> 标签（非流式）。

    返回: (cleaned_content, reasoning_content)；未命中时 reasoning 为 None。
    """
    if not isinstance(content, str) or _OPEN_TAG not in content:
        return content, None
    m = _THINK_RE.search(content)
    if not m:
        return content, None
    reasoning = m.group(1).strip()
    cleaned = (content[:m.start()] + content[m.end():]).strip() or None
    return cleaned, reasoning


def extract_think_from_response(response: IRResponse) -> None:
    """就地处理非流式 IRResponse：正文中的 <think> 提取为 ThinkingBlock。"""
    if response.thinking():
        return
    text = response.text()
    if _OPEN_TAG not in text:
        return

    cleaned, reasoning = extract_think_from_text(text)
    if not reasoning:
        return

    rebuilt: list[Any] = [ThinkingBlock(text=reasoning)]
    if cleaned:
        rebuilt.append(TextBlock(text=cleaned))
    rebuilt.extend(b for b in response.blocks if not isinstance(b, (TextBlock, ThinkingBlock)))
    response.blocks = rebuilt
    logger.info('已提取 <think> 标签内容并映射为思考块，长度=%s', len(reasoning))


class ThinkTagFilter:
    """流式 <think> 标签提取器（IR 事件级过滤器）。

    处理跨事件的 <think>...</think> 标签，把标签内的 TextDelta 改写为 ThinkingDelta。
    额外处理：首个工具调用出现前，如已输出过正文则补一个换行，
    保证 Cursor 侧文本与工具调用的分隔显示。
    """

    def __init__(self):
        self._in_thinking = False
        self._text_emitted = False
        self._tool_seen = False

    def process(self, event: StreamEvent) -> list[StreamEvent]:
        if isinstance(event, TextDelta):
            return self._split(event.text)
        if isinstance(event, ToolCallStart):
            return self._on_tool_start(event)
        return [event]

    def finalize(self) -> list[StreamEvent]:
        self._in_thinking = False
        return []

    def _on_tool_start(self, event: ToolCallStart) -> list[StreamEvent]:
        results: list[StreamEvent] = []
        if self._in_thinking:
            self._in_thinking = False
        if not self._tool_seen:
            self._tool_seen = True
            if self._text_emitted:
                results.append(TextDelta('\n'))
        results.append(event)
        return results

    def _split(self, text: str) -> list[StreamEvent]:
        """根据 <think> 标签拆分文本增量。"""
        if not text:
            return []
        results: list[StreamEvent] = []

        if self._in_thinking:
            end = text.find(_CLOSE_TAG)
            if end >= 0:
                self._in_thinking = False
                if text[:end]:
                    results.append(ThinkingDelta(text[:end]))
                rest = text[end + len(_CLOSE_TAG):].lstrip('\n')
                if rest:
                    results.append(self._text(rest))
            else:
                results.append(ThinkingDelta(text))
            return results

        start = text.find(_OPEN_TAG)
        if start < 0:
            return [self._text(text)]

        before = text[:start]
        after = text[start + len(_OPEN_TAG):]
        if before:
            results.append(self._text(before))

        end = after.find(_CLOSE_TAG)
        if end >= 0:
            if after[:end]:
                results.append(ThinkingDelta(after[:end]))
            rest = after[end + len(_CLOSE_TAG):].lstrip('\n')
            if rest:
                results.append(self._text(rest))
        else:
            self._in_thinking = True
            if after:
                results.append(ThinkingDelta(after))
        return results

    def _text(self, text: str) -> TextDelta:
        self._text_emitted = True
        return TextDelta(text)


# ═══════════════════════════════════════════════════════════
#  Thinking 缓存
# ═══════════════════════════════════════════════════════════

_TTL = 86400  # 24 小时


class ThinkingCache:
    """纯内存 thinking 缓存，在多轮对话中保存和恢复思考内容。"""

    def __init__(self):
        self._store: dict[str, tuple[str, float]] = {}

    # ─── IR 转换路径 ─────────────────────────────

    def inject_ir(self, request: IRRequest) -> None:
        """遍历 IR assistant 消息，缺少 ThinkingBlock 时从缓存注入。"""
        simple = [self._simplify_ir(m) for m in request.messages]
        for index, reasoning in self._lookup(simple).items():
            request.messages[index].blocks.insert(0, ThinkingBlock(text=reasoning))

    def store_ir(self, request: IRRequest, reasoning: str) -> None:
        """将本次响应的思考内容存入缓存（IR 路径）。"""
        simple = [self._simplify_ir(m) for m in request.messages]
        self._store_reasoning(simple, reasoning)

    # ─── Chat 透传路径（dict 消息） ──────────────

    def inject_cc(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """遍历 CC 格式 assistant 消息，缺少 reasoning_content 时从缓存注入。"""
        simple = [self._simplify_cc(m) for m in messages if isinstance(m, dict)]
        targets = [m for m in messages if isinstance(m, dict)]
        for index, reasoning in self._lookup(simple).items():
            targets[index]['reasoning_content'] = reasoning
        return messages

    def store_cc(self, messages: list[dict[str, Any]], reasoning: str) -> None:
        """将本次响应的思考内容存入缓存（CC 透传路径）。"""
        simple = [self._simplify_cc(m) for m in messages if isinstance(m, dict)]
        self._store_reasoning(simple, reasoning)

    # ─── 内部核心 ────────────────────────────────

    def _lookup(self, simple: list[dict[str, Any]]) -> dict[int, str]:
        """返回 {消息下标: 待注入 reasoning}。"""
        sid = self._session_id(simple)
        if not sid:
            return {}

        now = time.time()
        result: dict[int, str] = {}
        for index, msg in enumerate(simple):
            if msg['role'] != 'assistant' or msg['has_reasoning']:
                continue
            key = sid + ':' + self._message_hash(msg['text'], msg['tool_ids'])
            entry = self._store.get(key)
            if entry and (now - entry[1]) < _TTL:
                result[index] = entry[0]
                logger.debug('已从缓存注入 thinking (%d 字符)', len(entry[0]))
        return result

    def _store_reasoning(self, simple: list[dict[str, Any]], reasoning: str) -> None:
        if not reasoning:
            return
        sid = self._session_id(simple)
        if not sid:
            return
        # 以"空 assistant 消息"的哈希作为 key，与注入侧的哈希规则保持一致
        key = sid + ':' + self._message_hash('', [])
        self._store[key] = (reasoning, time.time())
        self._cleanup()

    def _session_id(self, simple: list[dict[str, Any]]) -> str:
        first_user = ''
        first_assistant = ''
        for msg in simple:
            role = msg['role']
            if role == 'user' and not first_user:
                first_user = msg['text']
            elif role == 'assistant' and not first_assistant:
                first_assistant = msg['text']
            if first_user and first_assistant:
                break
        if not first_user or not first_assistant:
            return ''
        raw = first_user + '|' + first_assistant
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    def _message_hash(self, text: str, tool_ids: list[str]) -> str:
        raw = json.dumps({'c': text, 't': sorted(tool_ids)}, ensure_ascii=False)
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    @staticmethod
    def _simplify_ir(message: IRMessage) -> dict[str, Any]:
        return {
            'role': message.role,
            'text': _strip_think(message.text()),
            'tool_ids': [_TOOL_ID_RE.sub('', b.id) for b in message.tool_calls()],
            'has_reasoning': message.has_thinking(),
        }

    @staticmethod
    def _simplify_cc(message: dict[str, Any]) -> dict[str, Any]:
        role = message.get('role', '')
        return {
            'role': 'skip' if role in ('system', 'developer') else role,
            'text': _strip_think(_flatten_content(message.get('content'))),
            'tool_ids': [
                _TOOL_ID_RE.sub('', tc.get('id', ''))
                for tc in message.get('tool_calls', [])
                if isinstance(tc, dict)
            ],
            'has_reasoning': bool(message.get('reasoning_content')),
        }

    def _cleanup(self) -> None:
        """惰性清理过期条目（缓存达到 100 条后触发全量扫描）。"""
        if len(self._store) < 100:
            return
        now = time.time()
        expired = [k for k, (_, ts) in self._store.items() if (now - ts) >= _TTL]
        for k in expired:
            del self._store[k]


thinking_cache = ThinkingCache()


def _flatten_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for p in content:
            if isinstance(p, dict) and p.get('type') == 'text':
                parts.append(p.get('text', ''))
            elif isinstance(p, str):
                parts.append(p)
        return '\n'.join(parts)
    return str(content) if content else ''


def _strip_think(text: str) -> str:
    text = _STRIP_THINK_RE.sub('', text)
    text = _STRIP_UNCLOSED_THINK_RE.sub('', text)
    return text.strip()


# ═══════════════════════════════════════════════════════════
#  /v1/messages 透传链路的 thinking 注入
# ═══════════════════════════════════════════════════════════


def inject_thinking_into_messages_response(data: dict[str, Any]) -> None:
    """非流式：将非标准 reasoning_content 字段注入为 Anthropic thinking block。"""
    rc = data.pop('reasoning_content', None) or data.pop('reasoningContent', None)
    if not rc:
        return

    content = data.get('content')
    if not isinstance(content, list):
        content = []

    # 避免重复注入
    if any(isinstance(b, dict) and b.get('type') == 'thinking' for b in content):
        return

    content.insert(0, {'type': 'thinking', 'thinking': rc})
    data['content'] = content
    logger.info('已注入 thinking block (%d 字符)', len(rc))


class MessagesThinkingStreamFilter:
    """流式：/v1/messages 透传时的 reasoning → thinking 事件注入。

    追踪上游 content block 的 index，在注入 thinking blocks 时使用独立的 index，
    并将后续上游 block 的 index 偏移，避免冲突。
    """

    def __init__(self):
        self._reasoning_buf = ''
        self._injected = False
        self._index_offset = 0

    def process(self, event_type: str, data: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
        """处理一条上游事件，返回 (event_type, data) 列表。"""
        out: list[tuple[str, dict[str, Any]]] = []

        for container_key in ('message', 'delta'):
            container = data.get(container_key)
            if isinstance(container, dict):
                rc = container.pop('reasoning_content', None) or container.pop('reasoningContent', None)
                if rc:
                    self._reasoning_buf += rc

        if self._reasoning_buf and not self._injected:
            delta = data.get('delta')
            if isinstance(delta, dict) and delta.get('type') == 'text_delta':
                self._injected = True
                out.extend(_thinking_block_events(self._reasoning_buf))
                self._index_offset = 1
                self._reasoning_buf = ''

        if self._index_offset and isinstance(data.get('index'), int):
            data['index'] += self._index_offset

        out.append((event_type or data.get('type', ''), data))
        return out


def _thinking_block_events(text: str) -> list[tuple[str, dict[str, Any]]]:
    """生成一组等价的 Anthropic thinking block 事件。"""
    return [
        ('content_block_start', {
            'type': 'content_block_start',
            'index': 0,
            'content_block': {'type': 'thinking', 'thinking': ''},
        }),
        ('content_block_delta', {
            'type': 'content_block_delta',
            'index': 0,
            'delta': {'type': 'thinking_delta', 'thinking': text},
        }),
        ('content_block_stop', {
            'type': 'content_block_stop',
            'index': 0,
        }),
    ]
