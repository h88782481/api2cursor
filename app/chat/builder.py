"""上游请求构造。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..errors import ApiError
from ..settings import Route, RouteResolver
from ..upstream import spec
from .cursor import CursorAdapter
from .exchange import CursorDialect
from .instructions import (
    InjectionResult,
    InstructionStatusTracker,
    apply_system_injection,
)
from .rosetta import Rosetta


@dataclass(slots=True)
class PreparedRequest:
    route: Route
    dialect: CursorDialect
    stream: bool
    custom_tools: set[str]
    injection: InjectionResult
    warnings: list[str]
    body: dict[str, Any]
    headers: dict[str, str]
    url: str


class UpstreamRequestBuilder:
    def __init__(
        self,
        resolver: RouteResolver,
        cursor: CursorAdapter,
        rosetta: Rosetta,
        instruction_status: InstructionStatusTracker,
    ):
        self.resolver = resolver
        self.cursor = cursor
        self.rosetta = rosetta
        self.instruction_status = instruction_status

    def build(self, payload: dict[str, Any]) -> PreparedRequest:
        client_model = payload.get('model')
        if not isinstance(client_model, str) or not client_model.strip():
            raise ApiError('model 必须是非空字符串')
        route = self.resolver.resolve(client_model)
        dialect, request, custom_tools = self.cursor.parse(payload, route.protocol)
        apply_text_replacements(request, route.replacements)
        injection = apply_system_injection(
            request,
            dialect,
            route.instructions.for_dialect(dialect),
        )
        self.instruction_status.record(route.client_model, injection)
        stream = bool(payload.get('stream'))
        request['model'] = route.upstream_model
        request['stream'] = {'enabled': stream, 'include_usage': True}
        _apply_reasoning(request, route)
        if route.protocol not in ('chat', 'responses'):
            request.pop('provider_extensions', None)

        body, warnings = self.rosetta.request_to(route.protocol, request)
        warnings = [*injection.warnings, *warnings]
        _apply_reasoning_wire(body, route)
        _request_encrypted_reasoning(body, route)
        _apply_service_tier(body, route, payload.get('service_tier'), warnings)
        wire = spec(route.protocol)
        wire.prepare_body(body)
        _apply_overrides(body, route.body_overrides)

        headers = wire.headers(route.api_key)
        _apply_header_overrides(headers, route.header_overrides)
        return PreparedRequest(
            route=route,
            dialect=dialect,
            stream=stream,
            custom_tools=custom_tools,
            injection=injection,
            warnings=warnings,
            body=body,
            headers=headers,
            url=wire.url(route.base_url, route.upstream_model, stream),
        )


def _apply_reasoning_wire(body: dict[str, Any], route: Route) -> None:
    if route.thinking_level == 'default':
        return
    if route.protocol == 'chat':
        body['reasoning_effort'] = route.thinking_level
    elif route.protocol == 'responses':
        body.setdefault('reasoning', {})['effort'] = route.thinking_level
    elif route.protocol == 'messages':
        body.setdefault('output_config', {})['effort'] = route.thinking_level
    else:
        body['thinkingConfig'] = {'thinkingLevel': route.thinking_level}


def _request_encrypted_reasoning(body: dict[str, Any], route: Route) -> None:
    if route.protocol != 'responses':
        return
    include = body.setdefault('include', [])
    if isinstance(include, list) and 'reasoning.encrypted_content' not in include:
        include.append('reasoning.encrypted_content')


def apply_text_replacements(request: dict[str, Any], template: Any) -> None:
    for template in template:
        for role in template.roles:
            if role == 'system':
                _replace_system(request, template.find, template.replace)
            elif role == 'user':
                _replace_user(request, template.find, template.replace)


def _replace_system(request: dict[str, Any], find: str, replace: str) -> None:
    _replace_parts(request.get('system_instruction'), find, replace)


def _replace_user(request: dict[str, Any], find: str, replace: str) -> None:
    for message in request.get('messages', []):
        if not isinstance(message, dict) or message.get('role') != 'user':
            continue
        _replace_parts(message.get('content'), find, replace)


def _replace_parts(parts: Any, find: str, replace: str) -> None:
    if isinstance(parts, str):
        return
    if not isinstance(parts, list):
        return
    for part in parts:
        if isinstance(part, dict) and part.get('type') == 'text':
            part['text'] = part.get('text', '').replace(find, replace)


def _apply_service_tier(
    body: dict[str, Any],
    route: Route,
    client_tier: Any,
    warnings: list[str],
) -> None:
    tier = 'priority' if route.fast_mode else client_tier
    if not isinstance(tier, str) or not tier:
        return
    if route.protocol in ('chat', 'responses'):
        body['service_tier'] = tier
    elif tier in ('fast', 'priority'):
        warnings.append(
            f'Cursor 请求了 Fast 模式，但 {route.protocol} 上游不支持，已忽略',
        )


def _apply_reasoning(request: dict[str, Any], route: Route) -> None:
    if route.thinking_level != 'default' and route.protocol != 'gemini':
        request['reasoning'] = {
            'mode': 'auto',
            'effort': route.thinking_level,
        }


def _apply_overrides(payload: dict[str, Any], overrides: dict[str, Any]) -> None:
    for key, value in overrides.items():
        if value is None:
            payload.pop(key, None)
        else:
            payload[key] = value


def _apply_header_overrides(headers: dict[str, str], overrides: dict[str, Any]) -> None:
    for key, value in overrides.items():
        existing = next((name for name in headers if name.lower() == key.lower()), key)
        if value is None:
            headers.pop(existing, None)
        else:
            headers[existing] = str(value)
