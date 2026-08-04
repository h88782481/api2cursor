"""管理面板与配置 API。"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, TypeVar

from fastapi import APIRouter, Depends, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel, ValidationError

from ..chat.instructions import blocks_as_dicts, valid_targets
from ..errors import ApiError
from ..settings import Settings, env
from ..settings.schema import (
    AddressTemplate,
    InstructionSettings,
    TextReplacementTemplate,
)
from .common import read_json_object, require_access
from .dto import AdminMapping, AdminSettingsUpdate

logger = logging.getLogger(__name__)
T = TypeVar('T', bound=BaseModel)

router = APIRouter()
admin_api = APIRouter(
    prefix='/api/admin',
    dependencies=[Depends(require_access)],
)

_STATIC_DIR = Path(__file__).resolve().parents[1] / 'static'


# ─── 静态页面 ─────────────────────────────────────


@router.get('/admin')
@router.get('/admin/')
async def admin_page():
    """返回管理面板首页。"""
    return FileResponse(_STATIC_DIR / 'admin.html')


# ─── 登录验证 ─────────────────────────────────────


@router.post('/api/admin/login')
async def admin_login(request: Request):
    """校验管理面板登录密钥。"""
    data = await read_json_object(request)
    if not env.access_api_key:
        return {'ok': True, 'message': '未配置鉴权'}
    key = data.get('key')
    if not isinstance(key, str):
        raise ApiError('key 必须是字符串', 'invalid_request_error', 400)
    if key == env.access_api_key:
        return {'ok': True}
    raise ApiError('密钥错误', 'authentication_error', 401)


# ─── 全局设置 ─────────────────────────────────────


@admin_api.get('/settings')
async def get_settings(request: Request):
    data = request.app.state.settings_repository.read()
    upstream = data.global_.upstream
    return {
        'proxy_target_url': upstream.base_url,
        'proxy_api_key': upstream.api_key,
        'debug_mode': data.global_.debug_mode or env.debug_mode,
        'env_target_url': env.proxy_target_url,
        'env_api_key': '***' if env.proxy_api_key else '',
    }


@admin_api.put('/settings')
async def update_settings(request: Request):
    data = _validate(AdminSettingsUpdate, await read_json_object(request))
    repository = request.app.state.settings_repository
    current = repository.edit()
    changes = data.model_fields_set
    if 'proxy_target_url' in changes:
        current.global_.upstream.base_url = data.proxy_target_url or ''
    if 'proxy_api_key' in changes:
        current.global_.upstream.api_key = data.proxy_api_key or ''
    if 'debug_mode' in changes:
        current.global_.debug_mode = data.debug_mode
    return _save_and_respond(repository, current, '全局设置已更新')


# ─── 指令注入块元数据 ─────────────────────────────


@admin_api.get('/instruction-blocks')
async def instruction_blocks():
    return blocks_as_dicts()


@admin_api.get('/templates')
async def get_templates(request: Request):
    settings = request.app.state.settings_repository.read()
    return settings.templates.model_dump(mode='json')


@admin_api.put('/templates/{kind}/{name:path}')
async def save_template(kind: str, name: str, request: Request):
    if kind not in ('address', 'instruction', 'body', 'header', 'replacement'):
        raise ApiError('模板类型无效', 'invalid_request_error', 400)
    name = name.strip()
    if not name:
        raise ApiError('模板名称不能为空', 'invalid_request_error', 400)

    repository = request.app.state.settings_repository
    current = repository.edit()
    templates = getattr(current.templates, kind)
    templates[name] = _parse_template(kind, await read_json_object(request))
    if kind == 'instruction':
        for model, mapping in current.models.items():
            if mapping.templates.instruction == name:
                request.app.state.instruction_status.clear(model)
    return _save_and_respond(repository, current, f'{kind} 模板已保存: {name}')


@admin_api.delete('/templates/{kind}/{name:path}')
async def delete_template(kind: str, name: str, request: Request):
    if kind not in ('address', 'instruction', 'body', 'header', 'replacement'):
        raise ApiError('模板类型无效', 'invalid_request_error', 400)
    repository = request.app.state.settings_repository
    current = repository.edit()
    if any(
        name in _mapping_template_names(mapping, kind)
        for mapping in current.models.values()
    ):
        raise ApiError('模板仍被模型映射使用，不能删除', 'conflict', 409)
    templates = getattr(current.templates, kind)
    templates.pop(name, None)
    return _save_and_respond(repository, current, f'{kind} 模板已删除: {name}')


@admin_api.get('/instruction-status')
async def instruction_status(request: Request):
    settings = request.app.state.settings_repository.read()
    tracker = request.app.state.instruction_status
    return {
        model: {
            dialect: status
            for dialect in ('function', 'custom_grammar')
            if (
                status := tracker.reconcile(
                    model,
                    dialect,
                    settings.templates.instruction.get(
                        mapping.templates.instruction,
                        InstructionSettings(),
                    ).for_dialect(dialect),
                )
            )
        }
        for model, mapping in settings.models.items()
    }


# ─── 模型映射 CRUD ────────────────────────────────


@admin_api.get('/mappings')
async def list_mappings(request: Request):
    settings = request.app.state.settings_repository.read()
    return {
        name: AdminMapping.from_mapping(mapping).model_dump(exclude={'name'})
        for name, mapping in settings.models.items()
    }


@admin_api.post('/mappings')
async def add_mapping(request: Request):
    data = _validate(AdminMapping, await read_json_object(request))
    name = data.name.strip()
    if not name:
        raise ApiError('名称不能为空', 'invalid_request_error', 400)

    repository = request.app.state.settings_repository
    current = repository.edit()
    _validate_template_selection(current, data)
    current.models[name] = data.to_mapping(name)
    request.app.state.instruction_status.clear(name)
    return _save_and_respond(repository, current, f'映射已添加: {name}')


@admin_api.put('/mappings/{name:path}')
async def update_mapping(name: str, request: Request):
    data = _validate(AdminMapping, await read_json_object(request))
    repository = request.app.state.settings_repository
    current = repository.edit()
    if name not in current.models:
        raise ApiError('映射不存在', 'not_found', 404)

    _validate_template_selection(current, data)
    new_name = data.name.strip() or name
    entry = data.to_mapping(new_name)
    if new_name != name:
        del current.models[name]
    current.models[new_name] = entry
    request.app.state.instruction_status.clear(name, new_name)
    return _save_and_respond(repository, current, f'映射已更新: {name} → {new_name}')


@admin_api.delete('/mappings/{name:path}')
async def delete_mapping(name: str, request: Request):
    repository = request.app.state.settings_repository
    current = repository.edit()
    if name in current.models:
        del current.models[name]
        request.app.state.instruction_status.clear(name)
        return _save_and_respond(repository, current, f'映射已删除: {name}')
    return {'ok': True}


# ─── 用量统计 ─────────────────────────────────────


@admin_api.get('/stats')
async def get_stats(request: Request):
    return request.app.state.usage_tracker.get_stats()


# ─── 内部辅助 ─────────────────────────────────────


def _validate(model: type[T], data: dict[str, Any]) -> T:
    try:
        return model.model_validate(data)
    except ValidationError as exc:
        issue = exc.errors(include_url=False)[0]
        location = '.'.join(str(part) for part in issue['loc'])
        message = f'{location}: {issue["msg"]}' if location else issue['msg']
        raise ApiError(message, 'invalid_request_error', 400) from exc


def _parse_template(kind: str, payload: dict[str, Any]) -> Any:
    if kind == 'address':
        return _validate(AddressTemplate, payload)
    if kind == 'instruction':
        value = _validate(InstructionSettings, payload)
        for dialect in ('function', 'custom_grammar'):
            rule = value.for_dialect(dialect)
            if rule.text and rule.target not in valid_targets(dialect):
                raise ApiError(
                    f'instructions.{dialect}.target 无效: {rule.target}',
                    'invalid_request_error',
                    400,
                )
            if rule.text and rule.target != 'all' and f'</{rule.target}>' in rule.text:
                raise ApiError(
                    f'instructions.{dialect}.text 不能包含 </{rule.target}>',
                    'invalid_request_error',
                    400,
                )
        return value
    if kind == 'replacement':
        return _validate(TextReplacementTemplate, payload)
    if not isinstance(payload, dict):
        raise ApiError('模板内容必须是 JSON 对象', 'invalid_request_error', 400)
    return payload


def _validate_template_selection(settings: Settings, mapping: AdminMapping) -> None:
    for kind in ('address', 'instruction', 'body', 'header', 'replacement'):
        for name in _mapping_template_names(mapping, kind):
            if name and name not in getattr(settings.templates, kind):
                raise ApiError(
                    f'{kind} 模板不存在: {name}',
                    'invalid_request_error',
                    400,
                )


def _mapping_template_names(mapping: AdminMapping, kind: str) -> list[str]:
    value = (
        mapping.templates.replacements
        if kind == 'replacement'
        else getattr(mapping.templates, kind)
    )
    return value if isinstance(value, list) else [value]


def _save_and_respond(repository, data: Settings, log_message: str) -> Any:
    try:
        repository.save(data)
    except (OSError, ValueError, ValidationError) as e:
        logger.error('保存失败: %s', e)
        raise ApiError(f'保存失败: {e}', 'save_error', 500) from e
    logger.info(log_message)
    return {'ok': True}


router.include_router(admin_api)
