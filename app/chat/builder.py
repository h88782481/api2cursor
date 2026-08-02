"""上游请求构造。"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

from llm_rosetta.types.ir import IRRequest

from ..settings import Route, RouteResolver
from ..upstream import spec
from .cursor import CursorAdapter
from .exchange import CursorDialect
from .rosetta import Rosetta


@dataclass(slots=True)
class PreparedRequest:
    route: Route
    dialect: CursorDialect
    stream: bool
    custom_tools: set[str]
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
    ):
        self.resolver = resolver
        self.cursor = cursor
        self.rosetta = rosetta

    def build(self, payload: dict[str, Any]) -> PreparedRequest:
        dialect, parsed_request, custom_tools = self.cursor.parse(payload)
        route = self.resolver.resolve(payload['model'])
        stream = bool(payload.get('stream'))

        request = copy.deepcopy(parsed_request)
        request['model'] = route.upstream_model
        request['stream'] = {'enabled': stream, 'include_usage': True}
        _inject_instructions(request, route.instructions, route.instructions_position)
        if route.protocol not in ('chat', 'responses'):
            request.pop('provider_extensions', None)

        body, warnings = self.rosetta.request_to(route.protocol, request)
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
            warnings=warnings,
            body=body,
            headers=headers,
            url=wire.url(route.base_url, route.upstream_model, stream),
        )


def _inject_instructions(
    request: IRRequest,
    custom: str,
    position: str,
) -> None:
    if not custom:
        return
    existing = request.get('system_instruction', [])
    custom_part = {'type': 'text', 'text': custom}
    request['system_instruction'] = (
        [*existing, custom_part] if position == 'append' else [custom_part, *existing]
    )


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
