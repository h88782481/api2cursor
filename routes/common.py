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

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RouteContext:
    """数据面路由使用的标准请求上下文。

    路由层会先根据客户端模型名解析出统一上下文，后续处理函数只需要关心
    上游模型、后端类型、目标地址、鉴权信息、流式标记和自定义指令，
    而不必重复访问配置层。
    """

    client_model: str
    upstream_model: str
    backend: str
    target_url: str
    api_key: str
    is_stream: bool
    custom_instructions: str
    instructions_position: str
    body_modifications: dict
    header_modifications: dict


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
        custom_instructions=mapping.get('custom_instructions', ''),
        instructions_position=mapping.get('instructions_position', 'prepend'),
        body_modifications=mapping.get('body_modifications', {}),
        header_modifications=mapping.get('header_modifications', {}),
    )



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


def responses_error_event(message: str) -> str:
    """构造 Responses 流式接口使用的错误事件。"""
    return sse_event_message('error', {'error': message})


# ─── 自定义指令注入 ──────────────────────────────


def _merge_text(custom: str, existing: str, position: str) -> str:
    """根据 position 决定自定义指令与原有内容的拼接顺序。"""
    if not existing:
        return custom
    if position == 'append':
        return existing + '\n\n' + custom
    return custom + '\n\n' + existing


def inject_instructions_cc(payload: dict[str, Any], instructions: str, position: str = 'prepend') -> dict[str, Any]:
    """向 Chat Completions 请求注入自定义指令。

    position='prepend' 时放在 system 消息开头，'append' 时放在末尾。
    """
    if not instructions:
        return payload

    messages = payload.get('messages', [])
    if messages and messages[0].get('role') == 'system':
        first = messages[0]
        original = first.get('content') or ''
        first['content'] = _merge_text(instructions, original, position)
    else:
        messages.insert(0, {'role': 'system', 'content': instructions})
        payload['messages'] = messages

    logger.info('已注入自定义指令到 CC system 消息 (%d 字符, %s)', len(instructions), position)
    return payload


def inject_instructions_responses(payload: dict[str, Any], instructions: str, position: str = 'prepend') -> dict[str, Any]:
    """向 Responses 请求注入自定义指令（写入 instructions 字段）。

    position='prepend' 时放在 instructions 开头，'append' 时放在末尾。
    """
    if not instructions:
        return payload

    existing = payload.get('instructions') or ''
    payload['instructions'] = _merge_text(instructions, existing, position)

    logger.info('已注入自定义指令到 Responses instructions (%d 字符, %s)', len(instructions), position)
    return payload


def inject_instructions_anthropic(payload: dict[str, Any], instructions: str, position: str = 'prepend') -> dict[str, Any]:
    """向 Anthropic Messages 请求注入自定义指令（写入 system 字段）。

    position='prepend' 时放在 system 开头，'append' 时放在末尾。
    """
    if not instructions:
        return payload

    existing = payload.get('system') or ''
    if isinstance(existing, list):
        existing = '\n'.join(
            block.get('text', '') for block in existing
            if isinstance(block, dict) and block.get('type') == 'text'
        )
    payload['system'] = _merge_text(instructions, existing, position)

    logger.info('已注入自定义指令到 Anthropic system (%d 字符, %s)', len(instructions), position)
    return payload


def should_inject_thinking(backend: str) -> bool:
    """判断当前后端是否需要注入历史 thinking。

    仅对明确能消费历史 reasoning/thinking 的后端启用：
    - anthropic
    - gemini
    - responses

    OpenAI Chat 兼容后端通常不接受 `reasoning_content` 历史字段，
    若注入会导致上游报错，因此显式排除。
    """
    return backend in ('anthropic', 'gemini', 'responses')


# ─── Body / Header 修改 ──────────────────────────


def apply_body_modifications(payload: dict[str, Any], modifications: dict[str, Any]) -> dict[str, Any]:
    """对转发请求体应用字段级修改。

    规则与 CursorProxy 一致：值为 null 的字段会被删除，其余字段设置/覆盖。
    """
    if not modifications:
        return payload
    for key, value in modifications.items():
        if value is None:
            payload.pop(key, None)
        else:
            payload[key] = value
    logger.info('已应用 body_modifications: %s', list(modifications.keys()))
    return payload


def apply_header_modifications(headers: dict[str, str], modifications: dict[str, Any]) -> dict[str, str]:
    """对转发请求头应用字段级修改。

    规则同 body：值为 null 删除，其余设置/覆盖。
    """
    if not modifications:
        return headers
    for key, value in modifications.items():
        if value is None:
            headers.pop(key, None)
        else:
            headers[key] = str(value)
    logger.info('已应用 header_modifications: %s', list(modifications.keys()))
    return headers


# ═══════════════════════════════════════════════════════════
#  后端注册表 + ClientFormatter 实现
# ═══════════════════════════════════════════════════════════


def get_outbound(backend: str):
    """根据后端类型获取对应的 OutboundTransformer 实例。"""
    from adapters.cc_anthropic_adapter import AnthropicOutbound
    from adapters.cc_gemini_adapter import GeminiOutbound
    from adapters.openai_compat_fixer import OpenAIChatOutbound
    from adapters.responses_cc_adapter import ResponsesOutbound

    registry = {
        'openai': OpenAIChatOutbound,
        'anthropic': AnthropicOutbound,
        'gemini': GeminiOutbound,
        'responses': ResponsesOutbound,
    }
    cls = registry.get(backend, OpenAIChatOutbound)
    return cls()


class CCClientFormatter:
    """Chat Completions 客户端格式化器。

    将通用处理结果格式化为 OpenAI Chat Completions 格式，
    供 /v1/chat/completions 端点使用。
    """

    def format_response(self, cc_response: dict[str, Any], model: str) -> dict[str, Any]:
        cc_response['model'] = model
        return cc_response

    def wrap_stream_item(self, item: Any) -> str:
        payload = item if isinstance(item, str) else json.dumps(item, ensure_ascii=False)
        return f'data: {payload}\n\n'

    def format_error(self, message: str) -> str:
        return sse_data_message({'error': {'message': message, 'type': 'upstream_error'}})

    def format_done(self) -> str | None:
        return sse_data_message('[DONE]')

    def start_events(self) -> list[str]:
        return []

    @property
    def usage_input_key(self) -> str:
        return 'prompt_tokens'

    @property
    def usage_output_key(self) -> str:
        return 'completion_tokens'


class ResponsesClientFormatter:
    """Responses API 客户端格式化器。

    将通用处理结果格式化为 OpenAI Responses 格式，
    供 /v1/responses 端点使用。

    流式场景使用 ResponsesStreamConverter 做 CC chunk → Responses SSE 转换。
    """

    def __init__(self, model: str = ''):
        from adapters.responses_cc_adapter import ResponsesStreamConverter, cc_to_responses
        self._model = model
        self._converter = ResponsesStreamConverter(model=model)
        self._cc_to_responses = cc_to_responses

    def format_response(self, cc_response: dict[str, Any], model: str) -> dict[str, Any]:
        return self._cc_to_responses(cc_response, model)

    def wrap_stream_item(self, item: Any) -> str:
        if isinstance(item, str):
            return item
        events = self._converter.process_cc_chunk(item)
        return ''.join(events)

    def format_error(self, message: str) -> str:
        return responses_error_event(message)

    def format_done(self) -> str | None:
        events = self._converter.finalize()
        return ''.join(events) if events else None

    def start_events(self) -> list[str]:
        return self._converter.start_events()

    @property
    def usage_input_key(self) -> str:
        return 'input_tokens'

    @property
    def usage_output_key(self) -> str:
        return 'output_tokens'


class ResponsesPassthroughFormatter:
    """Responses 透传格式化器。

    当后端本身就是 Responses 格式时使用，做轻量模型名改写。
    """

    def __init__(self, model: str = ''):
        self._model = model

    def format_response(self, response_data: dict[str, Any], model: str) -> dict[str, Any]:
        response_data['model'] = model
        return response_data

    def wrap_stream_item(self, item: Any) -> str:
        if isinstance(item, str):
            return item
        event_type = item.pop('_sse_event_type', None)
        if event_type:
            return f'event: {event_type}\ndata: {json.dumps(item, ensure_ascii=False)}\n\n'
        return f'data: {json.dumps(item, ensure_ascii=False)}\n\n'

    def format_error(self, message: str) -> str:
        return responses_error_event(message)

    def format_done(self) -> str | None:
        return None

    def start_events(self) -> list[str]:
        return []

    @property
    def usage_input_key(self) -> str:
        return 'input_tokens'

    @property
    def usage_output_key(self) -> str:
        return 'output_tokens'
