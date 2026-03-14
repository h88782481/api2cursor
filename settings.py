"""持久化配置管理

使用 data/settings.json 存储可通过管理面板修改的设置：
  - proxy_target_url / proxy_api_key: 可覆盖环境变量的全局配置
  - model_mappings: Cursor 模型名 → {upstream_model, backend, target_url, api_key, custom_instructions}
"""

import copy
import json
import os
import threading

from config import Config

_ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(_ROOT_DIR, 'data')
SETTINGS_FILE = os.path.join(DATA_DIR, 'settings.json')

_lock = threading.Lock()
_cache = None

_DEFAULTS = {
    'proxy_target_url': '',
    'proxy_api_key': '',
    'model_mappings': {},
}


def load():
    """从持久化文件读取配置并刷新内存缓存。

    如果配置文件不存在或内容损坏，会回退到默认值，保证服务仍然可以正常启动。
    """
    global _cache
    with _lock:
        if os.path.exists(SETTINGS_FILE):
            try:
                with open(SETTINGS_FILE, 'r', encoding='utf-8') as f:
                    _cache = {**_DEFAULTS, **json.load(f)}
            except (json.JSONDecodeError, OSError):
                _cache = copy.deepcopy(_DEFAULTS)
        else:
            _cache = copy.deepcopy(_DEFAULTS)
    return copy.deepcopy(_cache)


def save(data):
    """将当前配置写回到持久化文件并同步缓存。

    保存前会确保数据目录存在，并始终以默认配置为基底合并缺失字段。
    """
    global _cache
    with _lock:
        os.makedirs(DATA_DIR, exist_ok=True)
        _cache = {**_DEFAULTS, **data}
        with open(SETTINGS_FILE, 'w', encoding='utf-8') as f:
            json.dump(_cache, f, ensure_ascii=False, indent=2)


def get():
    """获取当前配置的深拷贝快照，保证调用方修改不影响缓存。"""
    with _lock:
        if _cache is None:
            pass
        else:
            return copy.deepcopy(_cache)
    return load()


def get_url():
    """获取当前生效的上游 URL，优先使用持久化配置。"""
    return get().get('proxy_target_url') or Config.PROXY_TARGET_URL


def get_key():
    """获取当前生效的 API 密钥，优先使用持久化配置。"""
    return get().get('proxy_api_key') or Config.PROXY_API_KEY


def resolve_model(model_name):
    """解析模型映射并返回完整的上游路由信息。"""
    settings = get()
    mappings = settings.get('model_mappings', {})
    base_url, base_key = get_url(), get_key()

    if model_name in mappings:
        m = mappings[model_name]
        backend = m.get('backend')
        if backend in ('', None, 'auto'):
            backend = _auto_detect(model_name)
        return {
            'upstream_model': m.get('upstream_model') or model_name,
            'backend': backend,
            'target_url': m.get('target_url') or base_url,
            'api_key': m.get('api_key') or base_key,
            'custom_instructions': m.get('custom_instructions') or '',
            'instructions_position': m.get('instructions_position') or 'prepend',
            'body_modifications': m.get('body_modifications') or {},
            'header_modifications': m.get('header_modifications') or {},
        }

    return {
        'upstream_model': model_name,
        'backend': _auto_detect(model_name),
        'target_url': base_url,
        'api_key': base_key,
        'custom_instructions': '',
        'instructions_position': 'prepend',
        'body_modifications': {},
        'header_modifications': {},
    }


def _auto_detect(name):
    """根据模型名关键字推断默认后端协议类型。

    当前规则较为保守：命中 `claude` 或 `anthropic` 走 Anthropic，
    其余模型默认视为 OpenAI 兼容后端。
    """
    lower = (name or '').lower()
    if 'claude' in lower or 'anthropic' in lower:
        return 'anthropic'
    if 'gemini' in lower:
        return 'gemini'
    return 'openai'
