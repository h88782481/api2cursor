"""用量统计 — 内存聚合

按模型名聚合请求数、token 用量等统计数据。
重启后重置，适合轻量监控场景。
"""

from __future__ import annotations

import threading
import time
from typing import Any


class _ModelStats:
    __slots__ = ('request_count', 'input_tokens', 'output_tokens', 'first_seen', 'last_seen')

    def __init__(self):
        self.request_count = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self.first_seen = time.time()
        self.last_seen = time.time()


class UsageTracker:
    def __init__(self):
        self._lock = threading.Lock()
        self._stats: dict[str, _ModelStats] = {}
        self._start_time = time.time()

    def record(
        self,
        model: str,
        usage: dict[str, Any] | None = None,
        *,
        input_key: str = 'prompt_tokens',
        output_key: str = 'completion_tokens',
    ) -> None:
        """记录一次请求的用量。"""
        with self._lock:
            if model not in self._stats:
                self._stats[model] = _ModelStats()
            s = self._stats[model]
            s.request_count += 1
            s.last_seen = time.time()
            if usage:
                s.input_tokens += usage.get(input_key, 0) or 0
                s.output_tokens += usage.get(output_key, 0) or 0

    def get_stats(self) -> dict[str, Any]:
        """返回所有模型的聚合统计。"""
        with self._lock:
            result = {}
            for model, s in self._stats.items():
                result[model] = {
                    'request_count': s.request_count,
                    'input_tokens': s.input_tokens,
                    'output_tokens': s.output_tokens,
                    'total_tokens': s.input_tokens + s.output_tokens,
                }
            return {
                'uptime_seconds': int(time.time() - self._start_time),
                'models': result,
            }

    def reset(self) -> None:
        with self._lock:
            self._stats.clear()
            self._start_time = time.time()


usage_tracker = UsageTracker()
