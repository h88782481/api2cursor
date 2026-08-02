"""按 Cursor 模型聚合 token 用量。"""

from __future__ import annotations

import threading
import time
from typing import Any, Mapping


class UsageTracker:
    def __init__(self):
        self._lock = threading.Lock()
        self._models: dict[str, dict[str, int]] = {}
        self._started_at = time.time()

    def record(self, model: str, usage: Mapping[str, Any] | None = None) -> None:
        with self._lock:
            stats = self._models.setdefault(
                model,
                {'request_count': 0, 'input_tokens': 0, 'output_tokens': 0},
            )
            stats['request_count'] += 1
            if usage:
                stats['input_tokens'] += int(usage.get('prompt_tokens') or 0)
                stats['output_tokens'] += int(usage.get('completion_tokens') or 0)

    def get_stats(self) -> dict[str, Any]:
        with self._lock:
            models = {
                name: {
                    **stats,
                    'total_tokens': stats['input_tokens'] + stats['output_tokens'],
                }
                for name, stats in self._models.items()
            }
        return {
            'uptime_seconds': int(time.time() - self._started_at),
            'models': models,
        }
