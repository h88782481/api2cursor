"""数据面入口路由

三个入口全部是薄壳，实际处理都在 core/pipeline.py：
  - POST /v1/chat/completions  Cursor 的 Chat Completions 请求（含 BYOK bug 的 Responses 风格请求体）
  - POST /v1/responses         Cursor 的 Responses 请求
  - POST /v1/messages          Cursor 的 Anthropic Messages 请求
  - GET  /v1/models            模型列表（供 Cursor 拉取）
"""

from __future__ import annotations

from fastapi import APIRouter, Request

from .. import store
from ..core.pipeline import handle_entry

router = APIRouter()


@router.post('/v1/chat/completions')
async def chat_completions(request: Request):
    return await handle_entry(request, 'chat')


@router.post('/v1/responses')
async def responses(request: Request):
    return await handle_entry(request, 'responses')


@router.post('/v1/messages')
async def messages(request: Request):
    return await handle_entry(request, 'messages')


@router.get('/v1/models')
async def list_models():
    """返回当前配置的模型列表，供 Cursor 拉取可用模型。"""
    mappings = store.get().get('model_mappings', {})
    models = [
        {
            'id': name,
            'object': 'model',
            'owned_by': info.get('upstream_format', 'custom') if isinstance(info, dict) else 'custom',
        }
        for name, info in mappings.items()
    ]
    if not models:
        models.append({
            'id': 'claude-sonnet-4-5-20250929',
            'object': 'model',
            'owned_by': 'anthropic',
        })
    return {'object': 'list', 'data': models}
