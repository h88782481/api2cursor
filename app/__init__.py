"""FastAPI 应用工厂。"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from .chat.gateway import ChatGateway
from .chat.instructions import InstructionStatusTracker
from .chat.rosetta import Rosetta
from .errors import ApiError
from .observability import RequestLogger, UsageTracker
from .settings import RouteResolver, SettingsRepository, env
from .upstream import UpstreamClient

logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).resolve().parent / 'static'

@asynccontextmanager
async def _lifespan(app: FastAPI):
    timeout = httpx.Timeout(
        connect=30.0,
        read=float(env.api_timeout),
        write=60.0,
        pool=30.0,
    )
    async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
        app.state.chat_gateway = ChatGateway(
            app.state.route_resolver,
            UpstreamClient(client),
            Rosetta(),
            app.state.request_log,
            app.state.usage_tracker,
            app.state.instruction_status,
        )
        try:
            yield
        finally:
            app.state.request_log.close()


def create_app() -> FastAPI:
    """创建并配置 FastAPI 应用实例。"""
    repository = SettingsRepository()
    repository.load()

    app = FastAPI(
        title='API 2 Cursor',
        lifespan=_lifespan,
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )
    app.state.settings_repository = repository
    app.state.route_resolver = RouteResolver(repository)
    app.state.usage_tracker = UsageTracker()
    app.state.request_log = RequestLogger(repository)
    app.state.instruction_status = InstructionStatusTracker()
    app.add_middleware(
        CORSMiddleware,
        allow_origins=['*'],
        allow_methods=['*'],
        allow_headers=['*'],
    )

    # ─── JSON 错误处理器 ──────────────────────────

    @app.exception_handler(ApiError)
    async def api_error(request: Request, exc: ApiError):
        return JSONResponse(exc.body(), status_code=exc.status_code)

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
        route = app.state.settings_repository.read().global_.upstream
        return {'status': 'ok', 'target': route.base_url or env.proxy_target_url}

    # ─── 路由注册 ────────────────────────────────

    from .api.admin import router as admin_router
    from .api.chat import router as chat_router

    app.include_router(chat_router)
    app.include_router(admin_router)
    app.mount('/static', StaticFiles(directory=str(STATIC_DIR)), name='static')

    return app
