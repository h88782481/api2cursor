"""管理 API 的请求与响应模型。"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from ..chat.instructions import valid_targets
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
    instructions: InstructionSettings = Field(default_factory=InstructionSettings)
    body_modifications: dict[str, Any] = Field(default_factory=dict)
    header_modifications: dict[str, Any] = Field(default_factory=dict)

    def validate_targets(self) -> None:
        for dialect in ('function', 'custom_grammar'):
            rule = self.instructions.for_dialect(dialect)
            if rule.text and rule.target not in valid_targets(dialect):
                raise ValueError(f'instructions.{dialect}.target 无效: {rule.target}')
            if rule.text and rule.target != 'all' and f'</{rule.target}>' in rule.text:
                raise ValueError(
                    f'instructions.{dialect}.text 不能包含 </{rule.target}>',
                )

    def to_mapping(self, default_name: str) -> ModelMapping:
        return ModelMapping(
            upstream=UpstreamSettings(
                model=self.upstream_model or default_name,
                protocol=self.upstream_protocol,
                base_url=self.target_url,
                api_key=self.api_key,
            ),
            instructions=self.instructions,
            request=RequestSettings(
                body=self.body_modifications,
                headers=self.header_modifications,
            ),
        )

    @classmethod
    def from_mapping(cls, mapping: ModelMapping) -> AdminMapping:
        return cls.model_construct(
            upstream_model=mapping.upstream.model,
            upstream_protocol=mapping.upstream.protocol,
            target_url=mapping.upstream.base_url,
            api_key=mapping.upstream.api_key,
            instructions=mapping.instructions,
            body_modifications=mapping.request.body,
            header_modifications=mapping.request.headers,
        )
