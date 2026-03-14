"""OpenAI 格式修复

这个模块专门处理 OpenAI Chat Completions 兼容层里的“脏活”：
- 请求方向：把 Cursor 发来的近似 OpenAI 格式修整成更标准的请求
- 响应方向：把上游返回的近似 OpenAI 格式修整成 Cursor 更容易消费的结果

这里之所以集中做兼容性修复，而不是散落在路由层，是因为这些规则本质上属于
“协议清洗”而不是“请求编排”。路由层只应该关心把请求送到哪里，修复规则则应该
在适配层统一收口，避免两条主链路各自维护一份类似逻辑。
"""

from __future__ import annotations

import json
import logging
from typing import Any

from utils.http import gen_id
from utils.think_tag import extract_from_text
from utils.tool_fixer import normalize_args, repair_str_replace_args

logger = logging.getLogger(__name__)

JsonDict = dict[str, Any]


# ─── 请求预处理 ───────────────────────────────────


def normalize_request(payload: JsonDict, upstream_model: str | None = None) -> JsonDict:
    """预处理 Cursor 发来的 OpenAI 风格请求。

    这个函数只做“让请求更像标准 OpenAI CC”的整理，不负责路由或网络层决策。
    当前处理的重点有两类：
    1. Cursor 偶尔会在 CC 端点混入 Anthropic 风格内容块，需要先转回 OpenAI 语义。
    2. 工具定义和 tool_choice 可能是 Cursor 的便捷写法，需要标准化后再发给上游。
    """
    if upstream_model:
        payload['model'] = upstream_model

    if 'messages' in payload:
        payload['messages'] = _convert_anthropic_messages(payload['messages'])

    if 'tools' not in payload:
        return payload

    payload['tools'] = [_normalize_tool_definition(tool) for tool in payload['tools']]
    _normalize_tool_choice(payload)
    return payload


# ─── 消息兼容转换 ─────────────────────────────────


def _convert_anthropic_messages(messages: Any) -> Any:
    """将消息中的 Anthropic tool_use/tool_result 块转回 OpenAI 风格消息。

    Cursor 在少数场景下会把 Anthropic 风格内容块直接发到
    `/v1/chat/completions`。如果不在这里先转换，后续上游即使是 OpenAI 兼容接口，
    也未必能理解这类内容块。
    """
    if not isinstance(messages, list):
        return messages

    converted: list[JsonDict] = []
    for message in messages:
        converted.extend(_convert_single_message(message))
    return converted


def _convert_single_message(message: Any) -> list[JsonDict]:
    """将单条消息转换为 1 条或多条 OpenAI 风格消息。"""
    if not isinstance(message, dict):
        return [message]

    content = message.get('content')
    if not isinstance(content, list):
        return [message]

    has_tool_use, has_tool_result = _detect_tool_blocks(content)
    if not has_tool_use and not has_tool_result:
        return [message]

    role = message.get('role', '')
    if role == 'assistant' and has_tool_use:
        return [_convert_assistant_tool_use_message(content)]
    if has_tool_result:
        return _convert_tool_result_message(role, content)
    return [message]


def _detect_tool_blocks(content: list[Any]) -> tuple[bool, bool]:
    """识别内容块里是否包含 Anthropic 风格工具调用或工具结果。"""
    has_tool_use = any(
        isinstance(block, dict) and block.get('type') == 'tool_use'
        for block in content
    )
    has_tool_result = any(
        isinstance(block, dict) and block.get('type') == 'tool_result'
        for block in content
    )
    return has_tool_use, has_tool_result


def _convert_assistant_tool_use_message(content: list[Any]) -> JsonDict:
    """将 assistant 的 tool_use 内容块转为 OpenAI tool_calls。"""
    text_parts: list[str] = []
    tool_calls: list[JsonDict] = []

    for block in content:
        if not isinstance(block, dict):
            continue
        if block.get('type') == 'text':
            text_parts.append(block.get('text', ''))
        elif block.get('type') == 'tool_use':
            tool_calls.append({
                'id': block.get('id', gen_id('call_')),
                'type': 'function',
                'function': {
                    'name': block.get('name', ''),
                    'arguments': json.dumps(block.get('input', {}), ensure_ascii=False),
                },
            })

    result: JsonDict = {
        'role': 'assistant',
        'content': '\n'.join(text_parts) if text_parts else None,
    }
    if tool_calls:
        result['tool_calls'] = tool_calls
    return result


def _convert_tool_result_message(role: str, content: list[Any]) -> list[JsonDict]:
    """将 tool_result 块拆成 OpenAI 的 tool 消息，并保留其余内容块。"""
    converted: list[JsonDict] = []
    other_parts: list[Any] = []

    for block in content:
        if not isinstance(block, dict):
            continue
        if block.get('type') == 'tool_result':
            converted.append({
                'role': 'tool',
                'tool_call_id': block.get('tool_use_id', ''),
                'content': _stringify_tool_result_content(block.get('content', '')),
            })
        else:
            other_parts.append(block)

    if other_parts:
        converted.append({'role': role, 'content': other_parts})
    return converted


def _stringify_tool_result_content(content: Any) -> str:
    """将 tool_result 的 content 规范为字符串。

    OpenAI 的 tool 消息内容天然更偏向字符串；而 Anthropic 的 tool_result 允许列表块。
    这里做一次降维，避免后续上游把结构化结果误当成普通消息块。
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return '\n'.join(
            block.get('text', '')
            for block in content
            if isinstance(block, dict) and block.get('type') == 'text'
        )
    return str(content)


def _normalize_tool_definition(tool: Any) -> Any:
    """将 Cursor 可能使用的扁平工具定义补成标准 OpenAI function tool。

    这里不主动过滤未知字段，只做最小标准化，避免在兼容层里过早丢失调用方提供的
    额外上下文。
    """
    if not isinstance(tool, dict):
        return tool
    if tool.get('type') == 'function' and 'function' in tool:
        return tool
    if 'name' not in tool:
        return tool
    return {
        'type': 'function',
        'function': {
            'name': tool.get('name', ''),
            'description': tool.get('description', ''),
            'parameters': (
                tool.get('input_schema')
                or tool.get('parameters')
                or {'type': 'object', 'properties': {}}
            ),
        },
    }


def _normalize_tool_choice(payload: JsonDict) -> None:
    """规范化 tool_choice。

    这里保留当前项目已有的映射约定：
    - `{"type": "auto"}` → `"auto"`
    - `{"type": "any"}`  → `"required"`

    这样做是因为部分上游只接受 OpenAI 常见的字符串写法，而不接受 Cursor/Anthropic
    风格的对象写法。
    """
    tool_choice = payload.get('tool_choice')
    if not isinstance(tool_choice, dict):
        return
    if tool_choice.get('type') == 'auto':
        payload['tool_choice'] = 'auto'
    elif tool_choice.get('type') == 'any':
        payload['tool_choice'] = 'required'


# ─── 非流式响应修复 ───────────────────────────────


def fix_response(data: Any) -> Any:
    """修复上游返回的非流式 OpenAI 响应。"""
    if not isinstance(data, dict):
        return data

    for choice in data.get('choices') or []:
        _fix_response_choice(choice)
    return data


def _fix_response_choice(choice: Any) -> None:
    """修复单个非流式 choice。"""
    if not isinstance(choice, dict):
        return

    message = choice.get('message') or {}
    if not isinstance(message, dict):
        return

    _promote_reasoning_field(message)
    _extract_reasoning_from_content(message)
    _convert_legacy_message_function_call(message, choice)
    _fix_tool_calls(message, choice)


def _promote_reasoning_field(container: JsonDict) -> None:
    """兼容不同上游返回的 reasoning 字段命名差异。"""
    if 'reasoningContent' in container and 'reasoning_content' not in container:
        container['reasoning_content'] = container.pop('reasoningContent')


def _extract_reasoning_from_content(message: JsonDict) -> None:
    """从 `<think>...</think>` 中提取 reasoning_content。

    有些上游把思考内容直接塞进 content 字符串里，而不是单独返回 reasoning 字段。
    这里主动提取，是为了让 Cursor 端更稳定地展示思考过程。
    """
    content = message.get('content') or ''
    if not isinstance(content, str):
        return
    if '<think>' not in content or message.get('reasoning_content'):
        return

    cleaned, reasoning = extract_from_text(content)
    if not reasoning:
        return

    message['reasoning_content'] = reasoning
    message['content'] = cleaned
    logger.info('已提取 <think> 标签内容并映射为 reasoning_content，长度=%s', len(reasoning))


def _convert_legacy_message_function_call(message: JsonDict, choice: JsonDict) -> None:
    """将旧版 function_call 字段升级为新版 tool_calls。"""
    if 'function_call' not in message or 'tool_calls' in message:
        return

    function_call = message.pop('function_call') or {}
    message['tool_calls'] = [{
        'id': gen_id('call_'),
        'type': 'function',
        'function': {
            'name': function_call.get('name', ''),
            'arguments': function_call.get('arguments', '{}'),
        },
    }]
    _rewrite_function_call_finish_reason(choice)


# ─── 流式 chunk 修复 ──────────────────────────────


def fix_stream_chunk(data: Any) -> Any:
    """修复上游返回的流式 OpenAI chunk。"""
    if not isinstance(data, dict):
        return data

    for choice in data.get('choices') or []:
        _fix_stream_choice(choice)
    return data


def _fix_stream_choice(choice: Any) -> None:
    """修复单个流式 choice。"""
    if not isinstance(choice, dict):
        return

    delta = choice.get('delta') or {}
    if not isinstance(delta, dict):
        return

    _promote_reasoning_field(delta)
    _convert_legacy_delta_function_call(delta, choice)
    _sanitize_tool_call_deltas(delta)
    _ensure_stream_tool_calls(delta)
    _rewrite_function_call_finish_reason(choice)


def _convert_legacy_delta_function_call(delta: JsonDict, choice: JsonDict) -> None:
    """将流式旧版 function_call 增量升级为 tool_calls 增量。"""
    if 'function_call' not in delta or 'tool_calls' in delta:
        return

    function_call = delta.pop('function_call') or {}
    tool_call: JsonDict = {'index': 0, 'type': 'function', 'function': {}}
    if 'name' in function_call:
        tool_call['id'] = gen_id('call_')
        tool_call['function']['name'] = function_call['name']
    if 'arguments' in function_call:
        tool_call['function']['arguments'] = function_call['arguments']

    delta['tool_calls'] = [tool_call]
    _rewrite_function_call_finish_reason(choice)


def _sanitize_tool_call_deltas(delta: JsonDict) -> None:
    """清理流式 tool_calls 中的空白字段。

    某些 OpenAI 兼容提供商在后续 tool_calls chunk 中错误地发送空字符串的
    id/type/function.name，导致 Cursor 用空值覆盖真实值。
    不处理 function.arguments，因为空字符串是合法的增量拼接值。
    """
    for tc in delta.get('tool_calls') or []:
        if not isinstance(tc, dict):
            continue
        if 'id' in tc and not str(tc['id']).strip():
            del tc['id']
        if 'type' in tc and not str(tc['type']).strip():
            del tc['type']
        func = tc.get('function')
        if isinstance(func, dict) and 'name' in func and not str(func['name']).strip():
            del func['name']


def _ensure_stream_tool_calls(delta: JsonDict) -> None:
    """补全流式 tool_calls 的最小必需字段。

    流式增量中的 tool_calls 往往是不完整片段，这里只补齐索引、ID、类型等元信息，
    不主动改写 arguments 内容，避免破坏增量拼接语义。
    """
    for tool_call in delta.get('tool_calls') or []:
        if 'index' not in tool_call:
            tool_call['index'] = 0
        function_data = tool_call.get('function') or {}
        if 'id' in tool_call or 'name' in function_data:
            if not tool_call.get('id'):
                tool_call['id'] = gen_id('call_')
            if 'type' not in tool_call:
                tool_call['type'] = 'function'


# ─── tool_calls 修复 ──────────────────────────────


def _fix_tool_calls(message: JsonDict, choice: JsonDict) -> None:
    """修复非流式消息中的 tool_calls 字段。"""
    tool_calls = message.get('tool_calls')
    if not tool_calls:
        return

    for index, tool_call in enumerate(tool_calls):
        _fill_tool_call_metadata(tool_call, index=index)
        _normalize_tool_call_arguments(tool_call)

    if choice.get('finish_reason') not in ('tool_calls', 'function_call'):
        choice['finish_reason'] = 'tool_calls'


def _fill_tool_call_metadata(tool_call: JsonDict, *, index: int) -> None:
    """补齐非流式 tool_call 的通用元数据。"""
    if not tool_call.get('id'):
        tool_call['id'] = gen_id('call_')
    if 'index' not in tool_call:
        tool_call['index'] = index
    if tool_call.get('type') != 'function':
        tool_call['type'] = 'function'


def _normalize_tool_call_arguments(tool_call: JsonDict) -> None:
    """规范化 tool_call 参数。

    这里会顺带调用工具参数修复器，原因是很多兼容性问题不在协议层，而在工具参数本身：
    比如 `file_path`/`path` 命名差异、智能引号、StrReplace 精确匹配失败等。
    """
    function_data = tool_call.get('function') or {}
    raw_arguments = function_data.get('arguments', '{}')

    try:
        arguments = (
            json.loads(raw_arguments)
            if isinstance(raw_arguments, str)
            else (raw_arguments or {})
        )
    except json.JSONDecodeError:
        arguments = {}

    arguments = normalize_args(arguments)
    arguments = repair_str_replace_args(function_data.get('name', ''), arguments)
    function_data['arguments'] = json.dumps(arguments, ensure_ascii=False)


def _rewrite_function_call_finish_reason(choice: JsonDict) -> None:
    """将旧版 finish_reason=function_call 升级为 tool_calls。"""
    if choice.get('finish_reason') == 'function_call':
        choice['finish_reason'] = 'tool_calls'
