"""配置文件读写与缓存。"""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Any

from .schema import Settings

ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / 'data'
SETTINGS_FILE = DATA_DIR / 'settings.json'


class SettingsRepository:
    def __init__(self, path: Path = SETTINGS_FILE):
        self.path = path
        self._lock = threading.RLock()
        self._settings: Settings | None = None

    def load(self) -> Settings:
        with self._lock:
            if self.path.exists():
                try:
                    raw = json.loads(self.path.read_text(encoding='utf-8'))
                except json.JSONDecodeError as exc:
                    raise ValueError(f'配置文件不是有效 JSON: {exc}') from exc
                settings = Settings.model_validate(raw)
            else:
                settings = Settings()
            self._settings = settings
            return self._settings

    def read(self) -> Settings:
        with self._lock:
            if self._settings is None:
                return self.load()
            return self._settings

    def edit(self) -> Settings:
        return self.read().model_copy(deep=True)

    def save(self, value: Settings | dict[str, Any]) -> Settings:
        settings = value if isinstance(value, Settings) else Settings.model_validate(value)
        payload = settings.model_dump(by_alias=True, mode='json')
        serialized = json.dumps(payload, ensure_ascii=False, indent=2)

        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            temp = self.path.with_suffix('.tmp')
            temp.write_text(serialized, encoding='utf-8')
            os.replace(temp, self.path)
            self._settings = Settings.model_validate_json(serialized)
            return self._settings
