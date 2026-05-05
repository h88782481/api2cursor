"""请求历史记录。

为管理后台提供最近请求查询能力，默认仅保留最近 500 条，
重启后会从磁盘恢复最近一次快照。
"""

from __future__ import annotations

import json
import os
import threading
from collections import deque
from datetime import datetime, timezone
from typing import Any

from settings import DATA_DIR

_MAX_RECORDS = 500
_FILE_PATH = os.path.join(DATA_DIR, 'request_logs.json')


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')


def _safe_int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _normalize_usage(usage: dict[str, Any] | None) -> dict[str, int]:
    usage = usage or {}
    input_tokens = _safe_int(
        usage.get('prompt_tokens', usage.get('input_tokens', 0))
    )
    output_tokens = _safe_int(
        usage.get('completion_tokens', usage.get('output_tokens', 0))
    )
    total_tokens = _safe_int(usage.get('total_tokens', input_tokens + output_tokens))
    return {
        'input_tokens': input_tokens,
        'output_tokens': output_tokens,
        'total_tokens': total_tokens,
    }


class RequestHistory:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._records: deque[dict[str, Any]] = deque(maxlen=_MAX_RECORDS)
        self._load()

    def record(
        self,
        *,
        route: str,
        client_model: str,
        actual_model: str,
        backend: str,
        upstream_url: str,
        usage: dict[str, Any] | None,
        duration_ms: int,
        started_at: str | None = None,
        status: str = 'ok',
        error_message: str = '',
    ) -> None:
        record = {
            'requested_at': started_at or _now_iso(),
            'route': route,
            'requested_model': client_model or '',
            'actual_model': actual_model or '',
            'backend': backend or '',
            'upstream_url': upstream_url or '',
            'duration_ms': max(_safe_int(duration_ms), 0),
            'status': status or 'ok',
            'error_message': error_message or '',
            'usage': _normalize_usage(usage),
            'recorded_at': _now_iso(),
        }
        with self._lock:
            self._records.appendleft(record)
            self._persist_locked()

    def get_recent(self, limit: int = _MAX_RECORDS) -> list[dict[str, Any]]:
        size = max(1, min(_safe_int(limit), _MAX_RECORDS))
        with self._lock:
            return list(self._records)[:size]

    def _load(self) -> None:
        if not os.path.exists(_FILE_PATH):
            return
        try:
            with open(_FILE_PATH, 'r', encoding='utf-8') as f:
                data = json.load(f)
            if not isinstance(data, list):
                return
            for item in data[:_MAX_RECORDS]:
                if isinstance(item, dict):
                    self._records.append(item)
        except (OSError, json.JSONDecodeError):
            self._records.clear()

    def _persist_locked(self) -> None:
        os.makedirs(DATA_DIR, exist_ok=True)
        with open(_FILE_PATH, 'w', encoding='utf-8') as f:
            json.dump(list(self._records), f, ensure_ascii=False, indent=2)


request_history = RequestHistory()
