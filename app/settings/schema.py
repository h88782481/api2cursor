"""应用配置模型。"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from ..protocol import ConfiguredProtocol

DebugMode = Literal['off', 'simple', 'verbose']
InjectionMode = Literal['prepend', 'append', 'replace']
InstructionTarget = str


class Environment(BaseSettings):
    model_config = SettingsConfigDict(
        env_file='.env',
        env_file_encoding='utf-8',
        extra='ignore',
    )

    proxy_target_url: str = 'https://api.anthropic.com'
    proxy_api_key: str = ''
    proxy_port: int = 3029
    api_timeout: int = 300
    access_api_key: str = ''
    debug_mode: DebugMode = 'off'


class UpstreamSettings(BaseModel):
    model_config = ConfigDict(extra='forbid')

    model: str = ''
    protocol: ConfiguredProtocol = 'auto'
    base_url: str = ''
    api_key: str = ''


class DialectInstruction(BaseModel):
    model_config = ConfigDict(extra='forbid')

    text: str = ''
    target: InstructionTarget = 'all'
    mode: InjectionMode = 'prepend'

    @field_validator('target')
    @classmethod
    def non_empty_target(cls, value: str) -> str:
        value = (value or '').strip()
        return value or 'all'

class InstructionSettings(BaseModel):
    model_config = ConfigDict(extra='forbid')

    function: DialectInstruction = Field(default_factory=DialectInstruction)
    custom_grammar: DialectInstruction = Field(default_factory=DialectInstruction)

    def for_dialect(self, dialect: Literal['function', 'custom_grammar']) -> DialectInstruction:
        return self.function if dialect == 'function' else self.custom_grammar


class RequestSettings(BaseModel):
    model_config = ConfigDict(extra='forbid')

    body: dict[str, Any] = Field(default_factory=dict)
    headers: dict[str, Any] = Field(default_factory=dict)


class ModelMapping(BaseModel):
    model_config = ConfigDict(extra='forbid')

    upstream: UpstreamSettings = Field(default_factory=UpstreamSettings)
    instructions: InstructionSettings = Field(default_factory=InstructionSettings)
    request: RequestSettings = Field(default_factory=RequestSettings)


class GlobalSettings(BaseModel):
    model_config = ConfigDict(extra='forbid')

    upstream: UpstreamSettings = Field(default_factory=UpstreamSettings)
    debug_mode: DebugMode | None = None


class Settings(BaseModel):
    model_config = ConfigDict(extra='forbid')

    global_: GlobalSettings = Field(
        default_factory=GlobalSettings,
        alias='global',
        serialization_alias='global',
    )
    models: dict[str, ModelMapping] = Field(default_factory=dict)


env = Environment()
