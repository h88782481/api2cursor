"""上游请求构造。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

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
        dialect, request, custom_tools = self.cursor.parse(payload)
        route = self.resolver.resolve(payload['model'])
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
        _apply_gemini_reasoning(body, route)
        if route.fast_mode and route.protocol in ('chat', 'responses'):
            body['service_tier'] = 'fast'
        if route.fast_mode and route.protocol not in ('chat', 'responses'):
            warnings.append(
                f'Fast 模式不适用于 {route.protocol} 上游，已忽略',
            )
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


def _apply_gemini_reasoning(body: dict[str, Any], route: Route) -> None:
    if route.protocol != 'gemini' or route.thinking_level == 'default':
        return
    body['thinkingConfig'] = {
        'thinkingLevel': route.thinking_level,
    }


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
