"""Flask 应用工厂

创建并配置 Flask 应用：
  - 注册所有路由蓝图
  - 设置 JSON 错误处理器（避免返回 HTML）
  - 配置全局鉴权中间件
"""

import logging

from flask import Flask, jsonify, request
from flask_cors import CORS

import settings
from config import Config
from routes import register_routes

logger = logging.getLogger(__name__)


def create_app():
    """创建并配置 Flask 应用实例。

    这里统一完成跨路由共享的初始化逻辑，包括配置加载、跨域、错误处理、
    访问鉴权、健康检查以及蓝图注册。
    """
    app = Flask(__name__)
    CORS(app)
    settings.load()

    # ─── JSON 错误处理器 ──────────────────────────

    @app.errorhandler(404)
    def not_found(e):
        """将未匹配到的路径统一转换为 JSON 404 响应。"""
        return jsonify({'error': {'message': '未找到', 'type': 'not_found'}}), 404

    @app.errorhandler(405)
    def method_not_allowed(e):
        """将不支持的请求方法统一转换为 JSON 405 响应。"""
        return jsonify({'error': {'message': '方法不允许', 'type': 'method_not_allowed'}}), 405

    @app.errorhandler(500)
    def internal_error(e):
        """将未捕获的服务端异常统一包装为 JSON 500 响应。"""
        return jsonify({'error': {'message': '服务器内部错误', 'type': 'server_error'}}), 500

    # ─── 全局鉴权中间件 ──────────────────────────

    @app.before_request
    def check_access():
        """在进入业务路由前校验访问密钥。

        当配置了 `ACCESS_API_KEY` 时，除健康检查和管理面板相关路径外，
        所有请求都必须携带正确的 Bearer Token 或 `x-api-key`。
        """
        if not Config.ACCESS_API_KEY:
            return

        # 无需鉴权的路径
        skip = ('/health', '/admin', '/static/', '/api/admin')
        if any(request.path == p or request.path.startswith(p) for p in skip):
            return

        auth = request.headers.get('Authorization', '')
        token = auth[7:] if auth.startswith('Bearer ') else request.headers.get('x-api-key', '')
        if token != Config.ACCESS_API_KEY:
            logger.warning(f'鉴权拒绝: {request.path}')
            return jsonify({
                'error': {'message': 'API 密钥无效', 'type': 'authentication_error'}
            }), 401

    # ─── 健康检查 ────────────────────────────────

    @app.route('/health', methods=['GET'])
    def health():
        """返回服务健康状态和当前生效的上游地址。"""
        return jsonify({'status': 'ok', 'target': settings.get_url()})

    # ─── 注册路由蓝图 ────────────────────────────

    register_routes(app)

    return app
