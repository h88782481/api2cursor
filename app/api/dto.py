"""管理 API 的请求与响应模型。"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from ..protocol import ConfiguredProtocol
from ..settings.schema import (
    DebugMode,
    InstructionSettings,
    ModelMapping,
    RequestSettings,
    UpstreamSettings,
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
    target_url: str = ''
    api_key: str = ''
    custom_instructions: str = ''
    instructions_position: Literal['prepend', 'append'] = 'prepend'
    body_modifications: dict[str, Any] = Field(default_factory=dict)
    header_modifications: dict[str, Any] = Field(default_factory=dict)

    def to_mapping(self, default_name: str) -> ModelMapping:
        return ModelMapping(
            upstream=UpstreamSettings(
                model=self.upstream_model or default_name,
                protocol=self.upstream_protocol,
                base_url=self.target_url,
                api_key=self.api_key,
            ),
            instructions=InstructionSettings(
                text=self.custom_instructions,
                position=self.instructions_position,
            ),
            request=RequestSettings(
                body=self.body_modifications,
                headers=self.header_modifications,
            ),
        )

    @classmethod
    def from_mapping(cls, mapping: ModelMapping) -> AdminMapping:
        return cls(
            upstream_model=mapping.upstream.model,
            upstream_protocol=mapping.upstream.protocol,
            target_url=mapping.upstream.base_url,
            api_key=mapping.upstream.api_key,
            custom_instructions=mapping.instructions.text,
            instructions_position=mapping.instructions.position,
            body_modifications=mapping.request.body,
            header_modifications=mapping.request.headers,
        )
