"""管理面板路由

  - /admin          管理面板页面
  - /api/admin/*    登录验证、全局设置、模型映射 CRUD、用量统计
"""

from __future__ import annotations

import logging
import os
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import FileResponse, JSONResponse

from .. import store
from ..config import env
from ..services.usage import usage_tracker

logger = logging.getLogger(__name__)

router = APIRouter()

_STATIC_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'static')


# ─── 静态页面 ─────────────────────────────────────


@router.get('/admin')
@router.get('/admin/')
async def admin_page():
    """返回管理面板首页。"""
    return FileResponse(os.path.join(_STATIC_DIR, 'admin.html'))


# ─── 登录验证 ─────────────────────────────────────


@router.post('/api/admin/login')
async def admin_login(request: Request):
    """校验管理面板登录密钥。"""
    data = await _json_body(request)
    if not env.access_api_key:
        return {'ok': True, 'message': '未配置鉴权'}
    if data.get('key', '') == env.access_api_key:
        return {'ok': True}
    return JSONResponse({'ok': False, 'message': '密钥错误'}, status_code=401)


# ─── 全局设置 ─────────────────────────────────────


@router.get('/api/admin/settings')
async def get_settings(request: Request):
    err = _check_auth(request)
    if err:
        return err
    data = store.get()
    return {
        'proxy_target_url': data.get('proxy_target_url', ''),
        'proxy_api_key': data.get('proxy_api_key', ''),
        'debug_mode': data.get('debug_mode', '') or env.resolved_debug_mode(),
        'env_target_url': env.proxy_target_url,
        'env_api_key': '***' if env.proxy_api_key else '',
    }


@router.put('/api/admin/settings')
async def update_settings(request: Request):
    err = _check_auth(request)
    if err:
        return err
    data = await _json_body(request)
    current = store.get()
    for key in ('proxy_target_url', 'proxy_api_key', 'debug_mode'):
        if key in data:
            current[key] = data[key]
    return _save_and_respond(current, '全局设置已更新')


# ─── 模型映射 CRUD ────────────────────────────────


@router.get('/api/admin/mappings')
async def list_mappings(request: Request):
    err = _check_auth(request)
    if err:
        return err
    return store.get().get('model_mappings', {})


@router.post('/api/admin/mappings')
async def add_mapping(request: Request):
    err = _check_auth(request)
    if err:
        return err
    data = await _json_body(request)
    name = str(data.get('name', '')).strip()
    if not name:
        return JSONResponse({'error': '名称不能为空'}, status_code=400)

    current = store.get()
    mappings = current.setdefault('model_mappings', {})
    mappings[name] = store.normalize_mapping_entry(data, default_name=name)
    return _save_and_respond(current, f'映射已添加: {name}')


@router.put('/api/admin/mappings/{name:path}')
async def update_mapping(name: str, request: Request):
    err = _check_auth(request)
    if err:
        return err
    data = await _json_body(request)
    current = store.get()
    mappings = current.get('model_mappings', {})
    if name not in mappings:
        return JSONResponse({'error': '映射不存在'}, status_code=404)

    new_name = str(data.get('name', name)).strip() or name
    entry = store.normalize_mapping_entry(data, default_name=new_name)
    if new_name != name:
        del mappings[name]
    mappings[new_name] = entry
    current['model_mappings'] = mappings
    return _save_and_respond(current, f'映射已更新: {name} → {new_name}')


@router.delete('/api/admin/mappings/{name:path}')
async def delete_mapping(name: str, request: Request):
    err = _check_auth(request)
    if err:
        return err
    current = store.get()
    mappings = current.get('model_mappings', {})
    if name in mappings:
        del mappings[name]
        current['model_mappings'] = mappings
        return _save_and_respond(current, f'映射已删除: {name}')
    return {'ok': True}


# ─── 用量统计 ─────────────────────────────────────


@router.get('/api/admin/stats')
async def get_stats(request: Request):
    err = _check_auth(request)
    if err:
        return err
    return usage_tracker.get_stats()


# ─── 内部辅助 ─────────────────────────────────────


def _check_auth(request: Request) -> JSONResponse | None:
    """Admin API 鉴权，返回 None 表示通过。"""
    if not env.access_api_key:
        return None
    auth = request.headers.get('authorization', '')
    token = auth[7:] if auth.startswith('Bearer ') else request.headers.get('x-api-key', '')
    if token != env.access_api_key:
        return JSONResponse({'error': '未授权'}, status_code=401)
    return None


async def _json_body(request: Request) -> dict[str, Any]:
    try:
        data = await request.json()
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def _save_and_respond(data: dict[str, Any], log_message: str) -> Any:
    try:
        store.save(data)
    except OSError as e:
        logger.error('保存失败: %s', e)
        return JSONResponse(
            {'error': {'message': f'保存失败: {e}', 'type': 'save_error'}},
            status_code=500,
        )
    logger.info(log_message)
    return {'ok': True}
