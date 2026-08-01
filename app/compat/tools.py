"""工具定义规范化与参数修复

集中处理旧版散落在各 adapter 中的工具兼容逻辑：
  - 工具定义统一解析为 IRTool（兼容 OpenAI 嵌套、Responses function/custom、Anthropic input_schema、Cursor 扁平写法）
  - tool_choice 各协议写法互转
  - LLM 生成参数的常见问题修复：智能引号、file_path→path、StrReplace 精确匹配
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Any

from ..core.ir import (
    IRRequest,
    IRTool,
    StreamEvent,
    ToolCallBlock,
    ToolCallDelta,
    ToolCallEnd,
    ToolCallStart,
    ToolResultBlock,
)

# 智能引号字符集
_SMART_DOUBLE = frozenset('«»\u201c\u201d\u275e\u201f\u201e\u275d')
_SMART_SINGLE = frozenset('\u2018\u2019\u201a\u201b')

_EMPTY_SCHEMA = {'type': 'object', 'properties': {}}


# ═══════════════════════════════════════════════════════════
#  工具定义解析
# ═══════════════════════════════════════════════════════════


def parse_tool_definitions(tools: Any) -> list[IRTool]:
    """将任意风格的工具定义列表解析为 IRTool 列表。"""
    if not isinstance(tools, list):
        return []
    result: list[IRTool] = []
    for tool in tools:
        parsed = _parse_tool_definition(tool)
        if parsed is not None:
            result.append(parsed)
    return result


def _parse_tool_definition(tool: Any) -> IRTool | None:
    if not isinstance(tool, dict):
        return None

    # 标准 OpenAI 嵌套格式: {type: "function", function: {name, parameters}}
    if tool.get('type') == 'function' and isinstance(tool.get('function'), dict):
        func = tool['function']
        return IRTool(
            name=func.get('name', ''),
            description=func.get('description', ''),
            parameters=func.get('parameters') or _EMPTY_SCHEMA,
        )

    # Responses 自定义工具: {type: "custom", name, format}
    if tool.get('type') == 'custom' and tool.get('name'):
        return IRTool(
            name=tool.get('name', ''),
            description=tool.get('description', ''),
            tool_type='custom',
            format=tool.get('format') or {},
        )

    # Responses 扁平格式: {type: "function", name, parameters} 或
    # Cursor / Anthropic 扁平格式: {name, input_schema | parameters}
    if 'name' in tool:
        return IRTool(
            name=tool.get('name', ''),
            description=tool.get('description', ''),
            parameters=tool.get('input_schema') or tool.get('parameters') or _EMPTY_SCHEMA,
        )

    return None


def parse_tool_choice(tool_choice: Any) -> Any:
    """将各协议的 tool_choice 写法统一为 IR 表示。

    IR 表示: None / 'auto' / 'required' / 'none' / {'name': 工具名}
    """
    if tool_choice is None:
        return None
    if isinstance(tool_choice, str):
        return tool_choice if tool_choice in ('auto', 'required', 'none') else None
    if isinstance(tool_choice, dict):
        choice_type = tool_choice.get('type', '')
        if choice_type == 'auto':
            return 'auto'
        if choice_type in ('any', 'required'):
            return 'required'
        if choice_type == 'none':
            return 'none'
        if choice_type in ('function', 'custom'):
            name = tool_choice.get('name') or (tool_choice.get('function') or {}).get('name')
            if name:
                return {'name': name}
        if choice_type == 'tool' and tool_choice.get('name'):
            return {'name': tool_choice['name']}
    return None


def degrade_custom_tools(request: IRRequest) -> set[str]:
    """将不支持 custom tool 的上游安全降级为单字符串 function tool。

    返回被降级的工具名集合，供响应侧把 JSON 包装还原为原始文本语义。
    """
    names = {tool.name for tool in request.tools if tool.tool_type == 'custom' and tool.name}
    for tool in request.tools:
        if tool.tool_type == 'custom':
            tool.tool_type = 'function'
            tool.parameters = {
                'type': 'object',
                'properties': {
                    'input': {
                        'type': 'string',
                        'description': '自定义工具的完整原始输入文本。',
                    },
                },
                'required': ['input'],
                'additionalProperties': False,
            }

    for message in request.messages:
        for block in message.blocks:
            if isinstance(block, ToolCallBlock) and (
                block.tool_type == 'custom' or block.name in names
            ):
                block.arguments = _wrap_custom_tool_input(block.arguments)
                block.tool_type = 'function'
            elif isinstance(block, ToolResultBlock) and block.tool_type == 'custom':
                block.tool_type = 'function'
    return names


@dataclass
class CustomToolStreamRestoreState:
    """跨 SSE chunk 记忆 custom tool 的名称与 IR index。"""

    names: set[str]
    indexes: set[int] = field(default_factory=set)
    pending_starts: dict[int, ToolCallStart] = field(default_factory=dict)
    pending_deltas: dict[int, list[ToolCallDelta]] = field(default_factory=dict)
    arguments: dict[int, str] = field(default_factory=dict)


def restore_custom_tool_events(
    events: list[StreamEvent],
    state: CustomToolStreamRestoreState,
) -> list[StreamEvent]:
    """恢复「已降级为 function」的 custom tool 流式语义，并处理名称延迟到达。

    state.names 为空时应直接透传（例如 Responses 原生 custom 路径），避免吞掉增量 delta。
    """
    if not state.names:
        return events

    restored: list[StreamEvent] = []
    for event in events:
        if isinstance(event, ToolCallStart):
            if event.name in state.names:
                event.tool_type = 'custom'
                state.indexes.add(event.index)
                state.arguments[event.index] = ''
                restored.append(event)
            elif event.name:
                restored.append(event)
            else:
                state.pending_starts[event.index] = event
            continue

        if isinstance(event, ToolCallDelta):
            state.arguments[event.index] = state.arguments.get(event.index, '') + event.arguments
            if event.index in state.indexes:
                continue
            if event.index in state.pending_starts:
                state.pending_deltas.setdefault(event.index, []).append(event)
                continue

        if isinstance(event, ToolCallEnd):
            pending = state.pending_starts.pop(event.index, None)
            pending_deltas = state.pending_deltas.pop(event.index, [])
            is_custom = event.index in state.indexes or event.name in state.names
            if is_custom:
                event.arguments = unwrap_custom_tool_input(
                    event.arguments or state.arguments.get(event.index, '')
                )
                event.tool_type = 'custom'
                state.indexes.add(event.index)
            state.arguments.pop(event.index, None)
            if pending is not None:
                if event.name:
                    pending.name = event.name
                if is_custom:
                    pending.tool_type = 'custom'
                restored.append(pending)
            if is_custom:
                restored.append(ToolCallDelta(index=event.index, arguments=event.arguments))
            elif pending is not None:
                restored.extend(pending_deltas)
            restored.append(event)
            continue

        restored.append(event)
    return restored


def restore_custom_tool_calls(request: IRRequest, custom_names: set[str]) -> None:
    """恢复已存在于请求历史中的 custom tool 调用与结果语义。"""
    if not custom_names:
        return
    custom_call_ids = {
        block.id
        for message in request.messages
        for block in message.blocks
        if isinstance(block, ToolCallBlock) and block.name in custom_names and block.id
    }
    for message in request.messages:
        for block in message.blocks:
            if isinstance(block, ToolCallBlock) and block.name in custom_names:
                block.tool_type = 'custom'
            elif isinstance(block, ToolResultBlock):
                block.tool_type = 'custom' if block.call_id in custom_call_ids else 'function'


def unwrap_custom_tool_input(arguments: str) -> str:
    """将严格的 {"input": "..."} 参数还原为 custom tool 原始文本。"""
    try:
        parsed = json.loads(arguments)
    except (json.JSONDecodeError, TypeError, ValueError):
        return arguments
    if (
        isinstance(parsed, dict)
        and set(parsed) == {'input'}
        and isinstance(parsed['input'], str)
    ):
        return parsed['input']
    return arguments


def _wrap_custom_tool_input(arguments: str) -> str:
    """把 custom tool 原始文本幂等包装成普通 function tool 的 JSON 参数。"""
    if not isinstance(arguments, str):
        arguments = dump_arguments(arguments)
    # 已是降级包装则原样返回，避免把自由文本或其它 JSON 二次嵌套错乱
    if unwrap_custom_tool_input(arguments) != arguments:
        return arguments
    return json.dumps({'input': arguments}, ensure_ascii=False)


# ═══════════════════════════════════════════════════════════
#  参数修复
# ═══════════════════════════════════════════════════════════


def fix_tool_call_block(block: ToolCallBlock) -> None:
    """就地修复 IR 工具调用块的参数（非流式响应路径）。"""
    if block.tool_type == 'custom':
        return
    try:
        args = json.loads(block.arguments) if isinstance(block.arguments, str) else block.arguments
    except (json.JSONDecodeError, ValueError):
        return
    if not isinstance(args, dict):
        return
    args = normalize_args(args)
    args = repair_str_replace_args(block.name, args)
    block.arguments = json.dumps(args, ensure_ascii=False)


def normalize_args(args: dict[str, Any]) -> dict[str, Any]:
    """规范化工具参数：file_path → path。"""
    if isinstance(args, dict) and 'file_path' in args and 'path' not in args:
        args['path'] = args.pop('file_path')
    return args


def repair_str_replace_args(tool_name: str, args: dict[str, Any]) -> dict[str, Any]:
    """修复 StrReplace/search_replace 工具的精确匹配问题。

    当 old_string 包含智能引号导致无法精确匹配文件内容时，
    用容错正则在文件中查找唯一匹配并替换为实际内容。
    """
    if not isinstance(args, dict):
        return args

    name_lower = (tool_name or '').lower()
    if 'str_replace' not in name_lower and 'search_replace' not in name_lower:
        return args

    old_str = args.get('old_string') or args.get('old_str')
    if not old_str:
        return args

    file_path = args.get('path') or args.get('file_path')
    if not file_path or not os.path.isfile(file_path):
        return args

    try:
        with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
            content = f.read()
    except OSError:
        return args

    # 已精确匹配，无需修复
    if old_str in content:
        return args

    pattern = _build_fuzzy_pattern(old_str)
    try:
        matches = list(re.finditer(pattern, content))
    except re.error:
        return args

    # 仅在唯一匹配时修复，避免歧义
    if len(matches) != 1:
        return args

    matched = matches[0].group()
    if 'old_string' in args:
        args['old_string'] = matched
    elif 'old_str' in args:
        args['old_str'] = matched

    # 同步修复 new_string 中的智能引号
    new_str = args.get('new_string') or args.get('new_str')
    if new_str:
        fixed = replace_smart_quotes(new_str)
        if 'new_string' in args:
            args['new_string'] = fixed
        elif 'new_str' in args:
            args['new_str'] = fixed

    return args


def replace_smart_quotes(text: str) -> str:
    """将智能引号替换为普通 ASCII 引号。"""
    return ''.join(
        '"' if ch in _SMART_DOUBLE else
        "'" if ch in _SMART_SINGLE else
        ch for ch in text
    )


def parse_arguments_dict(arguments: Any) -> dict[str, Any]:
    """将工具参数尽量解析为对象，供需要结构化 input 的协议使用。"""
    if isinstance(arguments, dict):
        return arguments
    if not isinstance(arguments, str):
        return {}
    try:
        parsed = json.loads(arguments)
    except (json.JSONDecodeError, ValueError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def dump_arguments(arguments: Any) -> str:
    """将工具参数统一序列化为字符串，保留 custom tool 的原始文本 input。"""
    if isinstance(arguments, str):
        return arguments
    try:
        return json.dumps(arguments if arguments is not None else {}, ensure_ascii=False)
    except (TypeError, ValueError):
        return '{}'


def _build_fuzzy_pattern(text: str) -> str:
    """构建容错正则：智能引号可互换、空白可伸缩、反斜杠可重复。"""
    parts = []
    for ch in text:
        if ch in _SMART_DOUBLE or ch == '"':
            parts.append('["\u00ab\u201c\u201d\u275e\u201f\u201e\u275d\u00bb]')
        elif ch in _SMART_SINGLE or ch == "'":
            parts.append("['\u2018\u2019\u201a\u201b]")
        elif ch in (' ', '\t'):
            parts.append(r'\s+')
        elif ch == '\\':
            parts.append(r'\\{1,2}')
        else:
            parts.append(re.escape(ch))
    return ''.join(parts)
