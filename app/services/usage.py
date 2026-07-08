"""用量统计 — 内存聚合

按 Cursor 模型名聚合请求数、token 用量。重启后重置，适合轻量监控场景。
"""

from __future__ import annotations

import threading
import time
from typing import Any

from ..core.ir import IRUsage


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

    def record(self, model: str, usage: IRUsage | None = None) -> None:
        """记录一次请求的用量。"""
        with self._lock:
            if model not in self._stats:
                self._stats[model] = _ModelStats()
            stats = self._stats[model]
            stats.request_count += 1
            stats.last_seen = time.time()
            if usage is not None:
                stats.input_tokens += usage.input_tokens or 0
                stats.output_tokens += usage.output_tokens or 0

    def get_stats(self) -> dict[str, Any]:
        """返回所有模型的聚合统计。"""
        with self._lock:
            models = {
                model: {
                    'request_count': s.request_count,
                    'input_tokens': s.input_tokens,
                    'output_tokens': s.output_tokens,
                    'total_tokens': s.input_tokens + s.output_tokens,
                }
                for model, s in self._stats.items()
            }
            return {
                'uptime_seconds': int(time.time() - self._start_time),
                'models': models,
            }

    def reset(self) -> None:
        with self._lock:
            self._stats.clear()
            self._start_time = time.time()


usage_tracker = UsageTracker()
