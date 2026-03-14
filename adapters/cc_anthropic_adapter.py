"""OpenAI Chat Completions ↔ Anthropic Messages 格式转换

这个模块是项目里最核心的协议桥之一，负责在两套主流对话协议之间做双向适配：
- 请求方向：OpenAI Chat Completions → Anthropic Messages
- 响应方向：Anthropic Messages → OpenAI Chat Completions
- 流式方向：Anthropic SSE 事件 → OpenAI Chat Completions chunk

这里的代码看起来会比普通字段映射更重，是因为它不仅要做字段重命名，还要处理：
- system 消息上提
- tool_calls / tool_use 双向映射
- tool 消息 / tool_result 双向映射
- 图片块转换
- 思考内容与流式工具参数的时序保留
"""

from __future__ import annotations

import json
from typing import Any

from utils.http import gen_id
from utils.tool_fixer import fix_anthropic_tool_use, normalize_args, repair_str_replace_args

JsonDict = dict[str, Any]


# Anthropic stop_reason → OpenAI finish_reason
_STOP_REASON_MAP = {
    'end_turn': 'stop',
    'max_tokens': 'length',
    'tool_use': 'tool_calls',
    'stop_sequence': 'stop',
}


# ═══════════════════════════════════════════════════════════
#  请求转换: CC → Messages
# ═══════════════════════════════════════════════════════════


def cc_to_messages_request(payload: JsonDict) -> JsonDict:
    """将 OpenAI Chat Completions 请求转换为 Anthropic Messages 请求。

    这一步不是简单替换字段名，而是主动把 OpenAI 世界中的几类特殊语义映射到
    Anthropic 世界：
    - `system` 消息提取到顶层 `system`
    - assistant 的 `tool_calls` 变成 `tool_use` 内容块
    - `tool` 角色消息变成 user 侧的 `tool_result` 内容块

    另外，这里会把相邻同角色消息做合并，因为 Anthropic 对消息角色交替的要求更严格。
    """
    messages = payload.get('messages', [])
    anthropic_messages: list[JsonDict] = []
    system_parts: list[str] = []

    for message in messages:
        converted, system_text = _convert_request_message(message)
        if system_text is not None:
            system_parts.append(system_text)
            continue
        if converted is not None:
            anthropic_messages.append(converted)

    anthropic_messages = _merge_same_role(anthropic_messages)
    result = _build_messages_request(payload, anthropic_messages, system_parts)
    optimize_cache_control(result)
    return result


# ═══════════════════════════════════════════════════════════
#  非流式响应转换: Messages → CC
# ═══════════════════════════════════════════════════════════


def messages_to_cc_response(data: JsonDict, request_id: str | None = None) -> JsonDict:
    """将 Anthropic Messages 非流式响应转换为 OpenAI CC 响应。"""
    request_id = request_id or gen_id('chatcmpl-')
    data = fix_anthropic_tool_use(data)

    content_text, reasoning_text, tool_calls = _collect_response_parts(data.get('content', []))
    message = _build_cc_message(content_text, reasoning_text, tool_calls)
    usage = data.get('usage', {})

    return {
        'id': request_id,
        'object': 'chat.completion',
        'model': data.get('model', 'claude'),
        'choices': [{
            'index': 0,
            'message': message,
            'finish_reason': _STOP_REASON_MAP.get(data.get('stop_reason', 'end_turn'), 'stop'),
        }],
        'usage': _build_cc_usage(
            input_tokens=usage.get('input_tokens', 0),
            output_tokens=usage.get('output_tokens', 0),
        ),
    }


# ═══════════════════════════════════════════════════════════
#  流式响应转换: Anthropic SSE → CC chunks
# ═══════════════════════════════════════════════════════════


class AnthropicStreamConverter:
    """将 Anthropic SSE 事件逐个转换为 OpenAI Chat Completions chunk。

    之所以做成有状态转换器，而不是单纯的函数映射，是因为 Anthropic 的流式工具调用
    会把名字、参数、结束信号拆散在多个事件中，而 OpenAI chunk 语义要求我们按顺序
    组装出连续的 `tool_calls` 增量。

    这个类主要维护三类状态：
    1. 当前请求的 chunk ID
    2. 当前工具调用的索引位置
    3. 输入 / 输出令牌统计

    最终目标是把 Anthropic 的事件流稳定映射成 Cursor 能直接消费的 CC chunk 流。
    """

    def __init__(self, request_id: str | None = None):
        """初始化流式转换所需的请求标识、工具索引和 usage 状态。"""
        self._id = request_id or gen_id('chatcmpl-')
        self._tool_index = -1
        self._input_tokens = 0
        self._output_tokens = 0

    def process_event(self, event_type: str, event_data: JsonDict) -> list[str]:
        """处理单个 Anthropic SSE 事件。

        调用方会按事件顺序不断喂入 event/data，这里根据事件类型拆成一个或多个 CC chunk
        字符串，交给上层直接作为 SSE data 发送给 Cursor。
        """
        if event_type == 'message_start':
            return self._handle_message_start(event_data)
        if event_type == 'content_block_start':
            return self._handle_content_block_start(event_data)
        if event_type == 'content_block_delta':
            return self._handle_content_block_delta(event_data)
        if event_type == 'message_delta':
            return self._handle_message_delta(event_data)
        return []

    def _handle_message_start(self, event_data: JsonDict) -> list[str]:
        """处理消息开始事件，产出 assistant 角色起始 chunk。

        这个起始 chunk 很重要，因为 Cursor 侧通常会依赖首个带 role 的 chunk 来初始化
        当前 assistant 消息。
        """
        message = event_data.get('message', {})
        self._input_tokens = message.get('usage', {}).get('input_tokens', 0)

        chunk = self._make_chunk(delta={'role': 'assistant', 'content': ''})
        if message.get('model'):
            chunk['model'] = message['model']
        return [self._dump_chunk(chunk)]

    def _handle_content_block_start(self, event_data: JsonDict) -> list[str]:
        """处理内容块开始事件。

        目前这里只需要显式处理 `tool_use`，因为文本和 thinking 的真正内容都在后续 delta
        事件里；而 tool_use 需要先开一个空 arguments 的 tool_call 槽位。
        """
        block = event_data.get('content_block', {})
        if block.get('type') != 'tool_use':
            return []

        self._tool_index += 1
        return [self._dump_chunk(self._make_chunk(delta={
            'tool_calls': [{
                'index': self._tool_index,
                'id': block.get('id', gen_id('toolu_')),
                'type': 'function',
                'function': {
                    'name': block.get('name', ''),
                    'arguments': '',
                },
            }]
        }))]

    def _handle_content_block_delta(self, event_data: JsonDict) -> list[str]:
        """处理内容块增量事件。

        Anthropic 会把文本、思考内容、工具参数拆成不同 delta 类型，这里要分别映射成
        OpenAI chunk 里的 `content`、`reasoning_content` 和 `tool_calls.function.arguments`。
        """
        delta = event_data.get('delta', {})
        delta_type = delta.get('type', '')

        if delta_type == 'text_delta' and delta.get('text'):
            return [self._dump_chunk(self._make_chunk(delta={'content': delta['text']}))]

        if delta_type == 'thinking_delta' and delta.get('thinking'):
            return [self._dump_chunk(self._make_chunk(delta={'reasoning_content': delta['thinking']}))]

        if delta_type == 'input_json_delta' and delta.get('partial_json'):
            return [self._dump_chunk(self._make_chunk(delta={
                'tool_calls': [{
                    'index': self._tool_index,
                    'function': {'arguments': delta['partial_json']},
                }]
            }))]

        return []

    def _handle_message_delta(self, event_data: JsonDict) -> list[str]:
        """处理消息收尾事件，补出 finish_reason 和 usage。

        当 Anthropic 发出 `message_delta` 时，说明这一轮 assistant 输出已经收束，
        这里会统一生成最后一个带 usage 的收尾 chunk。
        """
        delta = event_data.get('delta', {})
        usage = event_data.get('usage', {})
        self._output_tokens = usage.get('output_tokens', 0)

        chunk = self._make_chunk(
            delta={},
            finish_reason=_STOP_REASON_MAP.get(delta.get('stop_reason', ''), 'stop'),
        )
        chunk['usage'] = _build_cc_usage(
            input_tokens=self._input_tokens,
            output_tokens=self._output_tokens,
        )
        return [self._dump_chunk(chunk)]

    def _make_chunk(self, delta: JsonDict, finish_reason: str | None = None) -> JsonDict:
        """构造标准 OpenAI Chat Completions chunk 对象。"""
        choice: JsonDict = {'index': 0, 'delta': delta}
        if finish_reason:
            choice['finish_reason'] = finish_reason
        return {
            'id': self._id,
            'object': 'chat.completion.chunk',
            'model': 'claude',
            'choices': [choice],
        }

    @staticmethod
    def _dump_chunk(chunk: JsonDict) -> str:
        """统一序列化 chunk，方便上层直接写入 SSE data。"""
        return json.dumps(chunk)


# ═══════════════════════════════════════════════════════════
#  请求转换辅助
# ═══════════════════════════════════════════════════════════


def _convert_request_message(message: Any) -> tuple[JsonDict | None, str | None]:
    """将单条 OpenAI 消息转换为 Anthropic 消息或 system 文本。"""
    if not isinstance(message, dict):
        return None, None

    role = message.get('role', '')
    content = message.get('content', '')

    if role == 'system':
        return None, _flatten_text(content)
    if role == 'tool':
        return _convert_tool_role_message(message), None

    anthropic_role = 'assistant' if role == 'assistant' else 'user'
    anthropic_content = _convert_content(message)

    if role == 'assistant' and 'tool_calls' in message:
        anthropic_content = _append_tool_use_blocks(anthropic_content, message.get('tool_calls', []))

    if not anthropic_content and anthropic_content != 0:
        return None, None
    return {'role': anthropic_role, 'content': anthropic_content}, None


def _convert_tool_role_message(message: JsonDict) -> JsonDict | None:
    """将 OpenAI 的 tool 角色消息转换为 Anthropic 的 tool_result 内容块。"""
    content = message.get('content', '')
    text = content if isinstance(content, str) else json.dumps(content, ensure_ascii=False)
    anthropic_content = [{
        'type': 'tool_result',
        'tool_use_id': message.get('tool_call_id', ''),
        'content': text,
    }]

    if not anthropic_content:
        return None
    return {'role': 'user', 'content': anthropic_content}


def _append_tool_use_blocks(content: Any, tool_calls: list[Any]) -> list[JsonDict]:
    """把 OpenAI assistant.tool_calls 追加成 Anthropic tool_use 内容块。"""
    blocks = _to_blocks(content)
    for tool_call in tool_calls:
        if not isinstance(tool_call, dict):
            continue
        function_data = tool_call.get('function', {})
        blocks.append({
            'type': 'tool_use',
            'id': tool_call.get('id', gen_id('toolu_')),
            'name': function_data.get('name', ''),
            'input': _parse_tool_arguments(function_data.get('arguments', '{}')),
        })
    return blocks


def _build_messages_request(
    payload: JsonDict,
    anthropic_messages: list[JsonDict],
    system_parts: list[str],
) -> JsonDict:
    """组装最终的 Anthropic Messages 请求体。"""
    result: JsonDict = {
        'model': payload.get('model', 'claude-sonnet-4-20250514'),
        'messages': anthropic_messages,
        # 沿用项目当前策略：未设置或设置过小都兜底到 8192，避免上游因默认值过小过早截断。
        'max_tokens': max(payload.get('max_tokens') or 8192, 8192),
    }

    if system_parts:
        result['system'] = '\n\n'.join(system_parts)
    if 'tools' in payload:
        result['tools'] = _convert_tools(payload['tools'])

    for key in ('temperature', 'top_p', 'stream'):
        if key in payload:
            result[key] = payload[key]

    return result


# ═══════════════════════════════════════════════════════════
#  非流式响应转换辅助
# ═══════════════════════════════════════════════════════════


def _collect_response_parts(content_blocks: Any) -> tuple[str, str, list[JsonDict]]:
    """从 Anthropic content 块中提取文本、思考内容和工具调用。"""
    content_text = ''
    reasoning_text = ''
    tool_calls: list[JsonDict] = []

    if not isinstance(content_blocks, list):
        return content_text, reasoning_text, tool_calls

    for block in content_blocks:
        if not isinstance(block, dict):
            continue

        block_type = block.get('type', '')
        if block_type == 'text':
            content_text += block.get('text', '')
        elif block_type == 'thinking':
            reasoning_text += block.get('thinking', '')
        elif block_type == 'tool_use':
            tool_calls.append(_convert_tool_use_block(block, index=len(tool_calls)))

    return content_text, reasoning_text, tool_calls


def _convert_tool_use_block(block: JsonDict, *, index: int) -> JsonDict:
    """将 Anthropic 的 tool_use 块转换为 OpenAI tool_call。"""
    tool_name = block.get('name', '')
    input_data = block.get('input', {})

    if isinstance(input_data, dict):
        input_data = normalize_args(input_data)
        input_data = repair_str_replace_args(tool_name, input_data)
        arguments_text = json.dumps(input_data, ensure_ascii=False)
    else:
        arguments_text = str(input_data)

    return {
        'index': index,
        'id': block.get('id', gen_id('toolu_')),
        'type': 'function',
        'function': {
            'name': tool_name,
            'arguments': arguments_text,
        },
    }


def _build_cc_message(content_text: str, reasoning_text: str, tool_calls: list[JsonDict]) -> JsonDict:
    """构造 OpenAI CC 响应中的 assistant message。"""
    message: JsonDict = {
        'role': 'assistant',
        'content': content_text or None,
    }
    if reasoning_text:
        message['reasoning_content'] = reasoning_text
    if tool_calls:
        message['tool_calls'] = tool_calls
    return message


def _build_cc_usage(*, input_tokens: int, output_tokens: int) -> JsonDict:
    """将 Anthropic usage 字段映射为 OpenAI usage。"""
    return {
        'prompt_tokens': input_tokens,
        'completion_tokens': output_tokens,
        'total_tokens': input_tokens + output_tokens,
    }


# ═══════════════════════════════════════════════════════════
#  通用辅助
# ═══════════════════════════════════════════════════════════


def _parse_tool_arguments(arguments: Any) -> Any:
    """将 tool_call.arguments 尽量解析为对象，供 Anthropic tool_use.input 使用。

    Anthropic 的 `tool_use.input` 天然期望对象结构；如果这里直接保留原始字符串，
    后续上游会把它当普通文本而不是工具参数对象。
    """
    if not isinstance(arguments, str):
        return arguments if arguments is not None else {}
    try:
        return json.loads(arguments)
    except json.JSONDecodeError:
        return {}


def _flatten_text(content: Any) -> str:
    """将 content 扁平化为纯文本，主要用于 system 消息上提。"""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict) and part.get('type') == 'text':
                parts.append(part.get('text', ''))
        return '\n'.join(parts)
    return str(content)


def _convert_content(message: JsonDict) -> Any:
    """将 OpenAI 消息的 content 字段转换为 Anthropic 内容格式。"""
    content = message.get('content', '')
    if content is None:
        return ''
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return str(content)

    blocks: list[JsonDict] = []
    for part in content:
        converted = _convert_content_part(part)
        if converted is not None:
            blocks.append(converted)
    return blocks


def _convert_content_part(part: Any) -> JsonDict | None:
    """将单个 OpenAI content part 转为 Anthropic block。"""
    if isinstance(part, str):
        return {'type': 'text', 'text': part}
    if not isinstance(part, dict):
        return None

    part_type = part.get('type', '')
    if part_type == 'text':
        return {'type': 'text', 'text': part.get('text', '')}
    if part_type == 'image_url':
        return _convert_image(part)
    if part_type in ('tool_use', 'tool_result'):
        return part
    return None


def _convert_image(part: JsonDict) -> JsonDict:
    """将 OpenAI image_url 格式转换为 Anthropic image 格式。"""
    url_data = part.get('image_url', {})
    url = url_data.get('url', '') if isinstance(url_data, dict) else str(url_data)

    if url.startswith('data:'):
        media_type, _, base64_data = url.partition(';base64,')
        return {
            'type': 'image',
            'source': {
                'type': 'base64',
                'media_type': media_type.replace('data:', '') or 'image/png',
                'data': base64_data,
            },
        }

    return {
        'type': 'image',
        'source': {
            'type': 'url',
            'url': url,
        },
    }


def _convert_tools(tools: Any) -> list[JsonDict]:
    """将 OpenAI tools 转为 Anthropic tools 格式。

    这里兼容两种常见输入：
    - 标准 OpenAI `{"type": "function", "function": {...}}`
    - Cursor 常见的扁平工具格式 `{"name": ..., "input_schema": ...}`
    """
    if not isinstance(tools, list):
        return []

    result: list[JsonDict] = []
    for tool in tools:
        converted = _convert_tool_definition(tool)
        if converted is not None:
            result.append(converted)
    return result


def _convert_tool_definition(tool: Any) -> JsonDict | None:
    """将 OpenAI 或 Anthropic 风格的工具定义统一转换为 Anthropic `tools` 项。"""
    if not isinstance(tool, dict):
        return None

    if tool.get('type') == 'function' and 'function' in tool:
        function_data = tool['function']
        return {
            'name': function_data.get('name', ''),
            'description': function_data.get('description', ''),
            'input_schema': function_data.get('parameters', {'type': 'object', 'properties': {}}),
        }

    if 'name' in tool and 'input_schema' in tool:
        return {
            'name': tool.get('name', ''),
            'description': tool.get('description', ''),
            'input_schema': tool.get('input_schema', {'type': 'object', 'properties': {}}),
        }

    return None


def _to_blocks(content: Any) -> list[JsonDict]:
    """将内容统一转换成 block 列表。"""
    if isinstance(content, str):
        return [{'type': 'text', 'text': content}] if content else []
    if isinstance(content, list):
        return list(content)
    return [{'type': 'text', 'text': str(content)}] if content else []


def _merge_same_role(messages: list[JsonDict]) -> list[JsonDict]:
    """合并相邻同角色消息。

    Anthropic 要求消息角色严格交替，而 OpenAI/调用方不一定遵守这一点。
    这里仅合并“相邻同角色”消息，以最小改动满足 Anthropic 约束，同时尽量保留
    原本的消息顺序和内容块排列。
    """
    if not messages:
        return messages

    merged = [messages[0]]
    for message in messages[1:]:
        if message['role'] == merged[-1]['role']:
            previous_blocks = _to_blocks(merged[-1]['content'])
            current_blocks = _to_blocks(message['content'])
            merged[-1]['content'] = previous_blocks + current_blocks
        else:
            merged.append(message)
    return merged


# ═══════════════════════════════════════════════════════════
#  Anthropic cache_control 优化
# ═══════════════════════════════════════════════════════════

_MAX_BREAKPOINTS = 4
_BLOCK_WINDOW = 20
_EPHEMERAL = {'type': 'ephemeral'}


def optimize_cache_control(request: JsonDict) -> None:
    """自动设置最优的 Anthropic cache_control 断点。

    算法移植自 CursorProxy 的 ensure_cache_control.go：
    1. 归一化所有消息 content 为数组格式
    2. 清空所有已有 cache_control
    3. 注入结构锚点（tools 末尾 + system 末尾）
    4. 注入消息锚点（最后一个可缓存块 + 窗口边界）
    5. 总断点数不超过 4 个
    """
    _normalize_message_contents(request)
    _clear_all_cache_controls(request)

    structural = _inject_structural_anchors(request)
    remaining = _MAX_BREAKPOINTS - structural
    if remaining <= 0:
        return

    refs = _collect_cacheable_block_refs(request)
    if not refs:
        return

    desired = 1 if len(refs) < _BLOCK_WINDOW else 2
    anchors = min(desired, remaining)

    if anchors >= 1 and refs:
        refs[-1]['cache_control'] = _EPHEMERAL

    if anchors >= 2 and len(refs) > 1:
        target = len(refs) - _BLOCK_WINDOW
        idx = _pick_window_anchor(refs, target)
        if idx is not None and idx != len(refs) - 1:
            refs[idx]['cache_control'] = _EPHEMERAL


def _normalize_message_contents(request: JsonDict) -> None:
    """将所有消息的 content 统一转为数组格式。"""
    for msg in request.get('messages', []):
        content = msg.get('content')
        if isinstance(content, str):
            msg['content'] = [{'type': 'text', 'text': content}]
        elif content is None:
            msg['content'] = []


def _clear_all_cache_controls(request: JsonDict) -> None:
    """清空所有已有的 cache_control 字段。"""
    for tool in request.get('tools', []):
        tool.pop('cache_control', None)

    system = request.get('system')
    if isinstance(system, list):
        for block in system:
            if isinstance(block, dict):
                block.pop('cache_control', None)

    for msg in request.get('messages', []):
        content = msg.get('content')
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict):
                    block.pop('cache_control', None)


def _inject_structural_anchors(request: JsonDict) -> int:
    """在 tools 末尾和 system 末尾注入结构锚点，返回注入数量。"""
    count = 0

    tools = request.get('tools')
    if tools and isinstance(tools, list):
        tools[-1]['cache_control'] = _EPHEMERAL
        count += 1

    system = request.get('system')
    if isinstance(system, list) and system:
        last = system[-1]
        if isinstance(last, dict):
            last['cache_control'] = _EPHEMERAL
            count += 1
    elif isinstance(system, str) and system:
        request['system'] = [
            {'type': 'text', 'text': system, 'cache_control': _EPHEMERAL}
        ]
        count += 1

    return count


def _is_cacheable_block(block: Any) -> bool:
    """判断一个内容块是否可以设置 cache_control。"""
    if not isinstance(block, dict):
        return False
    block_type = block.get('type', '')
    if block_type in ('thinking', 'redacted_thinking'):
        return False
    if block_type == 'text' and not block.get('text'):
        return False
    return True


def _collect_cacheable_block_refs(request: JsonDict) -> list[JsonDict]:
    """收集所有消息中可缓存块的引用列表。"""
    refs: list[JsonDict] = []
    for msg in request.get('messages', []):
        content = msg.get('content')
        if not isinstance(content, list):
            continue
        for block in content:
            if _is_cacheable_block(block):
                refs.append(block)
    return refs


def _pick_window_anchor(refs: list[JsonDict], target: int) -> int | None:
    """在目标位置附近选择一个窗口锚点，优先左侧。"""
    if target < 0:
        target = 0
    if target >= len(refs):
        return None

    for i in range(target, -1, -1):
        if 'cache_control' not in refs[i]:
            return i
    for i in range(target + 1, len(refs)):
        if 'cache_control' not in refs[i]:
            return i
    return None
