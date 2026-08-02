"""Cursor Chat 数据面。"""

from __future__ import annotations

from fastapi import APIRouter, Depends, Request

from .common import read_json_object, require_access

router = APIRouter(dependencies=[Depends(require_access)])


@router.post('/v1/chat/completions')
async def chat_completions(request: Request):
    payload = await read_json_object(request)
    return await request.app.state.chat_gateway.handle(payload, dict(request.headers))


@router.get('/v1/models')
async def list_models(request: Request):
    model_ids = request.app.state.route_resolver.model_ids()
    if not model_ids:
        model_ids = ['claude-sonnet-4-5-20250929']
    return {
        'object': 'list',
        'data': [
            {'id': model, 'object': 'model', 'owned_by': 'custom'}
            for model in model_ids
        ],
    }
