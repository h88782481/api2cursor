"""API 共用的错误、鉴权与 JSON 解析。"""

from __future__ import annotations

from typing import Any

from fastapi import Request

from ..errors import ApiError
from ..settings import env


def extract_access_token(request: Request) -> str:
    authorization = request.headers.get('authorization', '')
    if authorization.lower().startswith('bearer '):
        return authorization[7:].strip()
    return request.headers.get('x-api-key', '').strip()


async def read_json_object(request: Request) -> dict[str, Any]:
    try:
        payload = await request.json()
    except Exception as exc:
        raise ApiError('无效的 JSON 请求体', 'invalid_request_error', 400) from exc
    if not isinstance(payload, dict):
        raise ApiError('请求体必须是 JSON 对象', 'invalid_request_error', 400)
    return payload


def require_access(request: Request) -> None:
    if env.access_api_key and extract_access_token(request) != env.access_api_key:
        raise ApiError('API 密钥无效', 'authentication_error', 401)
