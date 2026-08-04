"""管理 API 的请求与响应模型。"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from ..protocol import ConfiguredProtocol
from ..settings.schema import (
    DebugMode,
    ModelMapping,
    ModelUpstreamSettings,
    RequestSettings,
    TemplateSelection,
    ThinkingLevel,
)


class AdminSettingsUpdate(BaseModel):
    model_config = ConfigDict(extra='forbid')

    proxy_target_url: str | None = None
    proxy_api_key: str | None = None
    debug_mode: DebugMode | None = None


class AdminMapping(BaseModel):
    model_config = ConfigDict(extra='forbid')

    name: str = ''
    upstream_model: str = ''
    upstream_protocol: ConfiguredProtocol = 'auto'
    templates: TemplateSelection = Field(default_factory=TemplateSelection)
    thinking_level: ThinkingLevel = 'default'
    fast_mode: bool = False

    def to_mapping(self, default_name: str) -> ModelMapping:
        return ModelMapping(
            upstream=ModelUpstreamSettings(
                model=self.upstream_model or default_name,
                protocol=self.upstream_protocol,
            ),
            templates=self.templates,
            request=RequestSettings(
                thinking_level=self.thinking_level,
                fast_mode=self.fast_mode,
            ),
        )

    @classmethod
    def from_mapping(cls, mapping: ModelMapping) -> AdminMapping:
        return cls.model_construct(
            upstream_model=mapping.upstream.model,
            upstream_protocol=mapping.upstream.protocol,
            templates=mapping.templates,
            thinking_level=mapping.request.thinking_level,
            fast_mode=mapping.request.fast_mode,
        )
