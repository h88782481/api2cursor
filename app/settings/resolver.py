"""模型映射解析。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from ..protocol import ConfiguredProtocol, WireProtocol
from .repository import SettingsRepository
from .schema import env


@dataclass(frozen=True, slots=True)
class Route:
    client_model: str
    upstream_model: str
    protocol: WireProtocol
    base_url: str
    api_key: str
    instructions: str = ''
    instructions_position: Literal['prepend', 'append'] = 'prepend'
    body_overrides: dict[str, Any] = field(default_factory=dict)
    header_overrides: dict[str, Any] = field(default_factory=dict)


class RouteResolver:
    def __init__(self, repository: SettingsRepository):
        self.repository = repository

    def resolve(self, client_model: str) -> Route:
        settings = self.repository.read()
        global_upstream = settings.global_.upstream
        mapping = settings.models.get(client_model)
        upstream = mapping.upstream if mapping else None

        model = (upstream.model if upstream else '') or global_upstream.model or client_model
        protocol: ConfiguredProtocol = upstream.protocol if upstream else 'auto'
        if protocol == 'auto':
            protocol = global_upstream.protocol
        if protocol == 'auto':
            protocol = detect_protocol(model)

        if protocol == 'auto':
            raise RuntimeError('上游协议解析失败')

        return Route(
            client_model=client_model,
            upstream_model=model,
            protocol=protocol,
            base_url=(upstream.base_url if upstream else '')
            or global_upstream.base_url
            or env.proxy_target_url,
            api_key=(upstream.api_key if upstream else '')
            or global_upstream.api_key
            or env.proxy_api_key,
            instructions=mapping.instructions.text if mapping else '',
            instructions_position=mapping.instructions.position if mapping else 'prepend',
            body_overrides=dict(mapping.request.body) if mapping else {},
            header_overrides=dict(mapping.request.headers) if mapping else {},
        )

    def model_ids(self) -> list[str]:
        return list(self.repository.read().models)


def detect_protocol(model: str) -> WireProtocol:
    lower = model.lower()
    if 'claude' in lower or 'anthropic' in lower:
        return 'messages'
    if 'gemini' in lower:
        return 'gemini'
    return 'chat'


def effective_debug_mode(repository: SettingsRepository) -> str:
    return repository.read().global_.debug_mode or env.debug_mode
