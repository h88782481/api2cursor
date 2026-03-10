"""路由层公共辅助

收敛多个数据面路由都会用到的上下文解析、上游目标构造、日志输出和
SSE 消息拼装逻辑，避免 `chat.py` 和 `responses.py` 各自维护重复实现。
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import logging
from typing import Any

import settings
from utils.http import build_anthropic_headers, build_openai_headers

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RouteContext:
    """数据面路由使用的标准请求上下文。

    路由层会先根据客户端模型名解析出统一上下文，后续处理函数只需要关心
    上游模型、后端类型、目标地址、鉴权信息和流式标记，而不必重复访问配置层。
    """

    client_model: str
    upstream_model: str
    backend: str
    target_url: str
    api_key: str
    is_stream: bool


def build_route_context(client_model: str, is_stream: bool) -> RouteContext:
    """解析模型映射，得到当前请求的统一路由上下文。"""
    mapping = settings.resolve_model(client_model)
    return RouteContext(
        client_model=client_model,
        upstream_model=mapping['upstream_model'],
        backend=mapping['backend'],
        target_url=mapping['target_url'],
        api_key=mapping['api_key'],
        is_stream=is_stream,
    )


def build_openai_target(ctx: RouteContext) -> tuple[str, dict[str, str]]:
    """根据路由上下文生成 OpenAI 兼容后端的地址和请求头。"""
    url = f'{ctx.target_url.rstrip("/")}/v1/chat/completions'
    headers = build_openai_headers(ctx.api_key)
    return url, headers


def build_responses_target(ctx: RouteContext) -> tuple[str, dict[str, str]]:
    """根据路由上下文生成 OpenAI Responses 后端的地址和请求头。"""
    url = f'{ctx.target_url.rstrip("/")}/v1/responses'
    headers = build_openai_headers(ctx.api_key)
    return url, headers


def build_anthropic_target(ctx: RouteContext) -> tuple[str, dict[str, str]]:
    """根据路由上下文生成 Anthropic 后端的地址和请求头。"""
    url = f'{ctx.target_url.rstrip("/")}/v1/messages'
    headers = build_anthropic_headers(ctx.api_key)
    return url, headers


def log_route_context(route_name: str, ctx: RouteContext, *, extra: str = '') -> None:
    """统一输出路由级日志，避免不同入口的日志格式逐渐漂移。"""
    parts = [
        f'[{route_name}]',
        f'模型={ctx.client_model}',
        f'上游模型={ctx.upstream_model}',
        f'后端={ctx.backend}',
        f'流式={ctx.is_stream}',
    ]
    if extra:
        parts.append(extra)
    logger.info(' '.join(parts))


def log_usage(
    route_name: str,
    usage: dict[str, Any],
    *,
    input_key: str,
    output_key: str,
) -> None:
    """统一输出令牌统计日志。

    不同协议对 usage 字段命名不一致，这里只接收字段名，不在调用方重复拼接日志文案。
    """
    logger.info(
        '[%s] 请求完成 输入令牌=%s 输出令牌=%s',
        route_name,
        usage.get(input_key, 0),
        usage.get(output_key, 0),
    )


def sse_data_message(data: Any) -> str:
    """构造仅包含 data 的 SSE 消息。"""
    payload = data if isinstance(data, str) else json.dumps(data, ensure_ascii=False)
    return f'data: {payload}\n\n'


def sse_event_message(event_type: str, data: Any) -> str:
    """构造带 event 名称的 SSE 消息。"""
    payload = data if isinstance(data, str) else json.dumps(data, ensure_ascii=False)
    return f'event: {event_type}\ndata: {payload}\n\n'


def chat_error_chunk(message: str, error_type: str = 'upstream_error') -> str:
    """构造聊天补全流式接口使用的错误消息。"""
    return sse_data_message({'error': {'message': message, 'type': error_type}})


def responses_error_event(message: str) -> str:
    """构造 Responses 流式接口使用的错误事件。"""
    return sse_event_message('error', {'error': message})
