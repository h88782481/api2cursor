"""详细模式下记录对话请求与流事件。"""

from __future__ import annotations

import copy
import hashlib
import json
import logging
import queue
import re
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..settings import DATA_DIR, SettingsRepository, effective_debug_mode

logger = logging.getLogger(__name__)
_USER_QUERY = re.compile(r'<user_query>\s*(.*?)\s*</user_query>', re.DOTALL)
_TIMESTAMP = re.compile(r'<timestamp>\s*(.*?)\s*</timestamp>', re.DOTALL)
_SENSITIVE_HEADERS = {'authorization', 'x-api-key', 'api-key', 'x-goog-api-key'}
_STOP = object()


class RequestLogger:
    def __init__(self, repository: SettingsRepository):
        self.repository = repository
        self.directory = Path(DATA_DIR) / 'conversations'
        self._locks: dict[str, threading.Lock] = {}
        self._guard = threading.Lock()
        self._queue: queue.Queue[dict[str, Any] | object] = queue.Queue(maxsize=256)
        self._worker = threading.Thread(target=self._run, daemon=True)
        self._worker.start()
        self._closed = False

    def debug(self, tag: str, message: str) -> None:
        if effective_debug_mode(self.repository) in ('simple', 'verbose'):
            logger.info('[%s] %s', tag, message)

    def start(
        self,
        *,
        payload: dict[str, Any],
        request_headers: dict[str, Any],
        upstream_protocol: str,
        upstream_model: str,
        target_url: str,
        dialect: str,
    ) -> dict[str, Any] | None:
        if effective_debug_mode(self.repository) != 'verbose':
            return None
        now = _now()
        return {
            'conversation_id': _conversation_id(payload),
            'turn_id': f'turn_{uuid.uuid4().hex}',
            'route': 'chat',
            'client_model': payload.get('model', ''),
            'client_format': 'chat',
            'upstream_format': upstream_protocol,
            'upstream_model': upstream_model,
            'target_url': target_url,
            'stream': bool(payload.get('stream')),
            'started_at': now,
            'updated_at': now,
            'request_headers': sanitize_headers(request_headers),
            'client_request': copy.deepcopy(payload),
            'metadata': {'dialect': dialect, 'warnings': []},
            'upstream_request': None,
            'upstream_response': None,
            'client_response': None,
            'stream_trace': {
                'upstream_events': [],
                'client_events': [],
                'upstream_total': 0,
                'client_total': 0,
                'upstream_dropped': 0,
                'client_dropped': 0,
                'summary': {},
            },
            'error': None,
        }

    def upstream_request(
        self,
        turn: dict[str, Any] | None,
        body: dict[str, Any],
        headers: dict[str, Any],
    ) -> None:
        if turn is not None:
            turn['upstream_request'] = {
                'headers': sanitize_headers(headers),
                'body': copy.deepcopy(body),
            }

    def upstream_response(self, turn: dict[str, Any] | None, body: Any) -> None:
        if turn is not None:
            turn['upstream_response'] = copy.deepcopy(body)

    def client_response(self, turn: dict[str, Any] | None, body: Any) -> None:
        if turn is not None:
            turn['client_response'] = copy.deepcopy(body)

    def upstream_event(self, turn: dict[str, Any] | None, event: Any) -> None:
        self._append(turn, 'upstream', event)

    def client_event(self, turn: dict[str, Any] | None, event: Any) -> None:
        self._append(turn, 'client', {'raw': event})

    def warnings(self, turn: dict[str, Any] | None, warnings: list[str]) -> None:
        if turn is not None and warnings:
            turn['metadata']['warnings'].extend(warnings)

    def error(self, turn: dict[str, Any] | None, error: Any) -> None:
        if turn is not None:
            turn['error'] = copy.deepcopy(error)

    def finish(
        self,
        turn: dict[str, Any] | None,
        usage: dict[str, Any] | None = None,
    ) -> None:
        if turn is None or self._closed or effective_debug_mode(self.repository) != 'verbose':
            return
        turn['updated_at'] = _now()
        if usage:
            turn['usage'] = copy.deepcopy(usage)
        trace = turn['stream_trace']
        trace['summary'].update({
            'upstream_total': trace['upstream_total'],
            'client_total': trace['client_total'],
            'upstream_dropped': trace['upstream_dropped'],
            'client_dropped': trace['client_dropped'],
        })
        try:
            self._queue.put_nowait(copy.deepcopy(turn))
        except queue.Full:
            logger.warning('对话日志队列已满，丢弃日志: %s', turn['conversation_id'])

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self._queue.put(_STOP, timeout=1)
        except queue.Full:
            logger.warning('对话日志队列未能正常关闭')
            return
        self._worker.join()

    def _run(self) -> None:
        while True:
            turn = self._queue.get()
            try:
                if turn is _STOP:
                    return
                assert isinstance(turn, dict)
                self._write(turn)
            except Exception:
                logger.exception('写入对话日志失败')
            finally:
                self._queue.task_done()

    def _append(self, turn: dict[str, Any] | None, kind: str, event: Any) -> None:
        if turn is None:
            return
        trace = turn['stream_trace']
        events = trace[f'{kind}_events']
        trace[f'{kind}_total'] += 1
        if len(events) < 24:
            events.append(copy.deepcopy(event))
            return
        events.pop(12)
        events.append(copy.deepcopy(event))
        trace[f'{kind}_dropped'] += 1

    def _write(self, turn: dict[str, Any]) -> None:
        conversation_id = turn['conversation_id']
        with self._guard:
            lock = self._locks.setdefault(conversation_id, threading.Lock())
        with lock:
            directory = self.directory / turn['started_at'][:10]
            directory.mkdir(parents=True, exist_ok=True)
            path = directory / f'{conversation_id}.json'
            try:
                document = json.loads(path.read_text(encoding='utf-8')) if path.exists() else {
                    'conversation_id': conversation_id,
                    'route': 'chat',
                    'created_at': turn['started_at'],
                    'turns': [],
                }
            except (OSError, json.JSONDecodeError):
                document = {
                    'conversation_id': conversation_id,
                    'route': 'chat',
                    'created_at': turn['started_at'],
                    'turns': [],
                }
            document['turns'].append(turn)
            document['updated_at'] = turn['updated_at']
            document['last_client_model'] = turn['client_model']
            document['last_upstream_format'] = turn['upstream_format']
            document['turn_count'] = len(document['turns'])
            try:
                path.write_text(
                    json.dumps(document, ensure_ascii=False, indent=2),
                    encoding='utf-8',
                )
            except OSError as exc:
                logger.warning('写入对话日志失败: %s', exc)


def sanitize_headers(headers: dict[str, Any]) -> dict[str, Any]:
    return {
        key: _mask(value) if key.lower() in _SENSITIVE_HEADERS else value
        for key, value in headers.items()
    }


def _conversation_id(payload: dict[str, Any]) -> str:
    for key in ('conversation_id', 'conversationId', 'session_id', 'sessionId'):
        if value := payload.get(key):
            return _safe_id(str(value))
    seed = _root_user_text(payload.get('messages', []))
    return f'conv_{hashlib.sha256(seed.encode()).hexdigest()[:24]}'


def _root_user_text(messages: Any) -> str:
    first = ''
    for message in messages if isinstance(messages, list) else []:
        if not isinstance(message, dict) or message.get('role') != 'user':
            continue
        text = _flatten(message.get('content'))
        if not first:
            first = text
        match = _USER_QUERY.search(text)
        if match:
            timestamp = _TIMESTAMP.search(text)
            return json.dumps({
                'user_query': match.group(1).strip(),
                'timestamp': timestamp.group(1).strip() if timestamp else '',
            }, ensure_ascii=False, separators=(',', ':'))
    return first


def _flatten(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return '\n'.join(
            str(item.get('text', ''))
            for item in content
            if isinstance(item, dict) and item.get('text')
        )
    return str(content or '')


def _safe_id(value: str) -> str:
    return ''.join(char if char.isalnum() or char in '-_.' else '_' for char in value)[:120]


def _mask(value: Any) -> str:
    text = str(value or '')
    return '***' if len(text) <= 8 else f'{text[:4]}***{text[-4:]}'


def _now() -> str:
    return datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3] + 'Z'
