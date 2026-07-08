"""FastAPI 应用工厂

统一完成跨路由共享的初始化逻辑：
  - lifespan 管理共享的 httpx.AsyncClient 连接池
  - 全局访问鉴权中间件
  - JSON 错误处理器
  - 健康检查、数据面路由、管理面板路由与静态资源
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from . import store
from .config import env

logger = logging.getLogger(__name__)

STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static')

# 无需鉴权的路径前缀
_AUTH_SKIP = ('/health', '/admin', '/static', '/api/admin', '/favicon.ico')


@asynccontextmanager
async def _lifespan(app: FastAPI):
    timeout = httpx.Timeout(
        connect=30.0,
        read=float(env.api_timeout),
        write=60.0,
        pool=30.0,
    )
    async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
        app.state.http_client = client
        yield


def create_app() -> FastAPI:
    """创建并配置 FastAPI 应用实例。"""
    store.load()

    app = FastAPI(
        title='API 2 Cursor',
        lifespan=_lifespan,
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=['*'],
        allow_methods=['*'],
        allow_headers=['*'],
    )

    # ─── 全局鉴权中间件 ──────────────────────────

    @app.middleware('http')
    async def check_access(request: Request, call_next):
        """在进入业务路由前校验访问密钥。

        当配置了 `ACCESS_API_KEY` 时，除健康检查和管理面板相关路径外，
        所有请求都必须携带正确的 Bearer Token 或 `x-api-key`。
        CORS 预检请求（OPTIONS）不校验，交给 CORS 中间件处理。
        """
        if env.access_api_key and request.method != 'OPTIONS':
            path = request.url.path
            if not any(path == p or path.startswith(p + '/') or path.startswith(p) for p in _AUTH_SKIP):
                auth = request.headers.get('authorization', '')
                token = auth[7:] if auth.startswith('Bearer ') else request.headers.get('x-api-key', '')
                if token != env.access_api_key:
                    logger.warning('鉴权拒绝: %s', path)
                    return JSONResponse(
                        {'error': {'message': 'API 密钥无效', 'type': 'authentication_error'}},
                        status_code=401,
                    )
        return await call_next(request)

    # ─── JSON 错误处理器 ──────────────────────────

    @app.exception_handler(404)
    async def not_found(request: Request, exc):
        return JSONResponse({'error': {'message': '未找到', 'type': 'not_found'}}, status_code=404)

    @app.exception_handler(405)
    async def method_not_allowed(request: Request, exc):
        return JSONResponse({'error': {'message': '方法不允许', 'type': 'method_not_allowed'}}, status_code=405)

    @app.exception_handler(500)
    async def internal_error(request: Request, exc):
        return JSONResponse({'error': {'message': '服务器内部错误', 'type': 'server_error'}}, status_code=500)

    # ─── 健康检查 ────────────────────────────────

    @app.get('/health')
    async def health():
        return {'status': 'ok', 'target': store.get_url()}

    # ─── 路由注册 ────────────────────────────────

    from .api.admin import router as admin_router
    from .api.entry import router as entry_router

    app.include_router(entry_router)
    app.include_router(admin_router)
    app.mount('/static', StaticFiles(directory=STATIC_DIR), name='static')

    return app
