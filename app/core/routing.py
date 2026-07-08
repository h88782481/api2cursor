"""路由决策

根据入口路由和模型映射，解析出本次请求的完整路由信息：
Cursor 侧格式（响应必须匹配请求到达的入口）、中转站格式、上游地址与密钥、
自定义指令和 Body/Header 修改项。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from .. import store

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RouteDecision:
    client_model: str
    upstream_model: str
    client_format: str    # 有效的 Cursor 侧格式（由实际入口路由决定）
    upstream_format: str  # 中转站接收格式
    target_url: str
    api_key: str
    stream: bool
    custom_instructions: str = ''
    instructions_position: str = 'prepend'
    body_modifications: dict[str, Any] = field(default_factory=dict)
    header_modifications: dict[str, Any] = field(default_factory=dict)


def resolve_route(entry_format: str, client_model: str, stream: bool) -> RouteDecision:
    """解析模型映射，生成本次请求的路由决策。

    映射中声明的 client_format 主要用于管理面板展示；运行时以请求实际到达的
    入口为准（响应格式必须匹配入口才能被 Cursor 解析），不一致时仅记录警告。
    """
    mapping = store.resolve_model(client_model)

    declared = mapping['client_format']
    if declared not in ('auto', entry_format):
        logger.warning(
            '模型 %s 声明的 Cursor 发送格式为 %s，实际请求到达 %s 入口，以实际入口为准',
            client_model, declared, entry_format,
        )

    return RouteDecision(
        client_model=client_model,
        upstream_model=mapping['upstream_model'],
        client_format=entry_format,
        upstream_format=mapping['upstream_format'],
        target_url=mapping['target_url'],
        api_key=mapping['api_key'],
        stream=stream,
        custom_instructions=mapping['custom_instructions'],
        instructions_position=mapping['instructions_position'],
        body_modifications=mapping['body_modifications'],
        header_modifications=mapping['header_modifications'],
    )
