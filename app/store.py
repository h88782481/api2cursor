"""持久化配置管理

使用 data/settings.json 存储可通过管理面板修改的设置：
  - proxy_target_url / proxy_api_key / debug_mode: 覆盖环境变量的全局配置
  - model_mappings: Cursor 模型名 → 路由配置

模型映射结构（新版）：
  {
    "upstream_model": "发送到中转站的模型名",
    "client_format": "auto | chat | responses | messages",     # Cursor 发送什么格式
    "upstream_format": "auto | chat | responses | messages | gemini",  # 中转站接收什么格式
    "target_url": "", "api_key": "",
    "custom_instructions": "", "instructions_position": "prepend",
    "body_modifications": {}, "header_modifications": {}
  }

旧版的单一 backend 字段（openai/anthropic/responses/gemini/auto）在加载时自动迁移。
"""

from __future__ import annotations

import copy
import json
import logging
import os
import threading
from typing import Any

from .config import env

logger = logging.getLogger(__name__)

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT_DIR, 'data')
SETTINGS_FILE = os.path.join(DATA_DIR, 'settings.json')

CLIENT_FORMATS = ('auto', 'chat', 'responses', 'messages')
UPSTREAM_FORMATS = ('auto', 'chat', 'responses', 'messages', 'gemini')

# 旧版 backend → 新版 upstream_format
_LEGACY_BACKEND_MAP = {
    'openai': 'chat',
    'anthropic': 'messages',
    'responses': 'responses',
    'gemini': 'gemini',
    'auto': 'auto',
    '': 'auto',
}

_DEFAULTS: dict[str, Any] = {
    'proxy_target_url': '',
    'proxy_api_key': '',
    'debug_mode': '',
    'model_mappings': {},
}

_lock = threading.Lock()
_cache: dict[str, Any] | None = None


def load() -> dict[str, Any]:
    """从持久化文件读取配置、执行旧格式迁移并刷新内存缓存。"""
    global _cache
    with _lock:
        data = copy.deepcopy(_DEFAULTS)
        if os.path.exists(SETTINGS_FILE):
            try:
                with open(SETTINGS_FILE, 'r', encoding='utf-8') as f:
                    data = {**_DEFAULTS, **json.load(f)}
            except (json.JSONDecodeError, OSError) as e:
                logger.warning('读取配置失败，使用默认值: %s', e)

        if _migrate_mappings(data.get('model_mappings', {})):
            logger.info('已将旧版模型映射迁移到新格式')
            _write(data)

        _cache = data
    return copy.deepcopy(data)


def save(data: dict[str, Any]) -> None:
    """将配置写回持久化文件并同步缓存。"""
    global _cache
    with _lock:
        _cache = {**_DEFAULTS, **data}
        _write(_cache)


def get() -> dict[str, Any]:
    """获取当前配置的深拷贝快照。"""
    with _lock:
        if _cache is not None:
            return copy.deepcopy(_cache)
    return load()


def get_url() -> str:
    """当前生效的全局上游地址，管理面板配置优先。"""
    return get().get('proxy_target_url') or env.proxy_target_url


def get_key() -> str:
    """当前生效的全局 API 密钥，管理面板配置优先。"""
    return get().get('proxy_api_key') or env.proxy_api_key


def get_debug_mode() -> str:
    """当前生效的调试模式，管理面板配置优先。"""
    mode = (get().get('debug_mode') or '').strip().lower()
    return mode if mode in ('off', 'simple', 'verbose') else env.resolved_debug_mode()


def resolve_model(model_name: str) -> dict[str, Any]:
    """解析模型映射并返回完整的路由信息。"""
    data = get()
    mapping = data.get('model_mappings', {}).get(model_name) or {}

    upstream_model = mapping.get('upstream_model') or model_name
    upstream_format = mapping.get('upstream_format') or 'auto'
    if upstream_format not in UPSTREAM_FORMATS:
        upstream_format = 'auto'
    if upstream_format == 'auto':
        upstream_format = detect_upstream_format(upstream_model)

    client_format = mapping.get('client_format') or 'auto'
    if client_format not in CLIENT_FORMATS:
        client_format = 'auto'

    return {
        'upstream_model': upstream_model,
        'client_format': client_format,
        'upstream_format': upstream_format,
        'target_url': mapping.get('target_url') or get_url(),
        'api_key': mapping.get('api_key') or get_key(),
        'custom_instructions': mapping.get('custom_instructions') or '',
        'instructions_position': mapping.get('instructions_position') or 'prepend',
        'body_modifications': mapping.get('body_modifications') or {},
        'header_modifications': mapping.get('header_modifications') or {},
    }


def detect_upstream_format(model_name: str) -> str:
    """根据模型名关键字推断上游协议格式。"""
    lower = (model_name or '').lower()
    if 'claude' in lower or 'anthropic' in lower:
        return 'messages'
    if 'gemini' in lower:
        return 'gemini'
    return 'chat'


def normalize_mapping_entry(data: dict[str, Any], default_name: str = '') -> dict[str, Any]:
    """将管理面板提交的映射数据规范化为标准结构。"""
    client_format = data.get('client_format') or 'auto'
    upstream_format = data.get('upstream_format') or 'auto'
    return {
        'upstream_model': data.get('upstream_model') or default_name,
        'client_format': client_format if client_format in CLIENT_FORMATS else 'auto',
        'upstream_format': upstream_format if upstream_format in UPSTREAM_FORMATS else 'auto',
        'target_url': data.get('target_url', ''),
        'api_key': data.get('api_key', ''),
        'custom_instructions': data.get('custom_instructions', ''),
        'instructions_position': data.get('instructions_position', 'prepend'),
        'body_modifications': data.get('body_modifications') or {},
        'header_modifications': data.get('header_modifications') or {},
    }


def _migrate_mappings(mappings: dict[str, Any]) -> bool:
    """就地迁移旧版 backend 字段，返回是否发生修改。"""
    changed = False
    for entry in mappings.values():
        if not isinstance(entry, dict):
            continue
        if 'upstream_format' not in entry:
            legacy = str(entry.pop('backend', 'auto') or 'auto').lower()
            entry['upstream_format'] = _LEGACY_BACKEND_MAP.get(legacy, 'auto')
            changed = True
        elif 'backend' in entry:
            entry.pop('backend')
            changed = True
        if 'client_format' not in entry:
            entry['client_format'] = 'auto'
            changed = True
    return changed


def _write(data: dict[str, Any]) -> None:
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(SETTINGS_FILE, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
