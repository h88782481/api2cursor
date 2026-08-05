"""Cursor 当前两种 Chat 方言适配。"""

from __future__ import annotations

import copy
import json
from typing import Any

from llm_rosetta.types.ir import IRRequest, IRResponse

from ..errors import ApiError
from ..protocol import WireProtocol
from .exchange import CursorDialect
from .reasoning_display import (
    extract_response_reasoning_states,
    mirror_reasoning_response,
    restore_ir_reasoning_states,
    restore_reasoning_carriers,
)
from .rosetta import Rosetta


class CursorAdapter:
    def __init__(self, rosetta: Rosetta):
        self.rosetta = rosetta

    def parse(
        self,
        payload: dict[str, Any],
        upstream_protocol: WireProtocol | None = None,
    ) -> tuple[CursorDialect, IRRequest, set[str]]:
        self._validate(payload)
        payload, carriers = restore_reasoning_carriers(payload)
        custom_tools = {
            tool['name']
            for tool in payload.get('tools', [])
            if isinstance(tool, dict) and tool.get('type') == 'custom'
        }
        dialect: CursorDialect = 'custom_grammar' if custom_tools else 'function'

        if not custom_tools:
            ir_request = self.rosetta.request_from('chat', payload)
            restore_ir_reasoning_states(ir_request, carriers, upstream_protocol)
            self._preserve_extensions(ir_request, payload)
            return dialect, ir_request, set()

        ordinary = copy.deepcopy(payload)
        custom = [
            tool for tool in ordinary.get('tools', [])
            if isinstance(tool, dict) and tool.get('type') == 'custom'
        ]
        ordinary['tools'] = [
            tool for tool in ordinary.get('tools', [])
            if not isinstance(tool, dict) or tool.get('type') != 'custom'
        ]

        ir_request = self.rosetta.request_from('chat', ordinary)
        custom_ir = self.rosetta.request_from(
            'responses',
            {'model': payload['model'], 'input': '', 'tools': custom},
        )
        ir_request.setdefault('tools', []).extend(custom_ir.get('tools', []))
        self._restore_custom_history(ir_request, custom_tools)
        restore_ir_reasoning_states(ir_request, carriers, upstream_protocol)
        self._preserve_extensions(ir_request, payload)
        return dialect, ir_request, custom_tools

    def response(
        self,
        response: IRResponse,
        client_model: str,
        custom_tools: set[str],
        upstream_protocol: WireProtocol,
    ) -> dict[str, Any]:
        response = copy.deepcopy(response)
        response['model'] = client_model
        reasoning_states = extract_response_reasoning_states(response, upstream_protocol)
        for choice in response.get('choices', []):
            content = choice.get('message', {}).get('content', [])
            if any(part.get('type') == 'tool_call' for part in content):
                choice['finish_reason'] = {'reason': 'tool_calls'}
        result = self.rosetta.response_to_chat(response)
        self._unwrap_custom_response(result, custom_tools)
        mirror_reasoning_response(result, reasoning_states)
        return result

    @staticmethod
    def stream_chunk(chunk: dict[str, Any], client_model: str) -> dict[str, Any]:
        chunk['model'] = client_model
        return chunk

    @staticmethod
    def unwrap_custom_arguments(raw: Any) -> str:
        text = raw if isinstance(raw, str) else str(raw or '')
        try:
            parsed = json.loads(text)
        except (json.JSONDecodeError, TypeError):
            return text
        if isinstance(parsed, dict) and list(parsed) == ['input']:
            return str(parsed['input'])
        return text

    @staticmethod
    def _validate(payload: dict[str, Any]) -> None:
        if not isinstance(payload.get('model'), str) or not payload['model'].strip():
            raise ApiError('model 必须是非空字符串')
        if not isinstance(payload.get('messages'), list):
            raise ApiError('messages 必须是数组')

        for tool in payload.get('tools', []):
            if not isinstance(tool, dict):
                raise ApiError('tools 中的每一项都必须是对象')
            tool_type = tool.get('type')
            if tool_type == 'function':
                if not isinstance(tool.get('function'), dict):
                    raise ApiError('function 工具缺少 function 定义')
            elif tool_type == 'custom':
                if not tool.get('name') or not isinstance(tool.get('format'), dict):
                    raise ApiError('custom 工具必须包含 name 和 format')
            else:
                raise ApiError(f'不支持的 Cursor 工具类型: {tool_type}')

    @staticmethod
    def _restore_custom_history(request: IRRequest, custom_tools: set[str]) -> None:
        for message in request.get('messages', []):
            if not isinstance(message, dict):
                continue
            for part in message.get('content', []):
                if part.get('type') != 'tool_call' or part.get('tool_name') not in custom_tools:
                    continue
                part['tool_type'] = 'custom'
                tool_input = part.get('tool_input', {})
                if 'raw_arguments' in tool_input:
                    part['tool_input'] = {'input': tool_input['raw_arguments']}

    @staticmethod
    def _preserve_extensions(request: IRRequest, payload: dict[str, Any]) -> None:
        extensions = {
            key: payload[key]
            for key in ('user', 'metadata')
            if key in payload
        }
        if extensions:
            request['provider_extensions'] = extensions

    @staticmethod
    def _unwrap_custom_response(result: dict[str, Any], custom_tools: set[str]) -> None:
        if not custom_tools:
            return
        for choice in result.get('choices', []):
            message = choice.get('message', {})
            for tool_call in message.get('tool_calls', []):
                function = tool_call.get('function', {})
                if function.get('name') not in custom_tools:
                    continue
                function['arguments'] = CursorAdapter.unwrap_custom_arguments(
                    function.get('arguments', ''),
                )
