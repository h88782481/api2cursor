"""Responses API 适配

Cursor 对 GPT、Claude-Opus 等模型使用 `/v1/responses` 格式。
本模块负责在 Responses 与 Chat Completions 两种表示之间做双向转换：
- 请求方向：Responses → Chat Completions
- 响应方向：Chat Completions → Responses（非流式）
- 流式方向：CC chunk / Anthropic SSE → Responses SSE

这个模块之所以相对复杂，是因为 Responses 在“输出项”层面比 Chat Completions 更细：
它会把思考、文本、工具调用拆成独立项目，因此流式场景必须靠状态机把不同来源的
增量重新组织成稳定的 Responses 事件序列。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from utils.http import gen_id

JsonDict = dict[str, Any]


# ═══════════════════════════════════════════════════════════
#  请求转换: Responses → CC
# ═══════════════════════════════════════════════════════════


def responses_to_cc(payload: JsonDict) -> JsonDict:
    """将 `/v1/responses` 请求转换为 `/v1/chat/completions` 请求。"""
    messages: list[JsonDict] = []

    if payload.get('instructions'):
        messages.append({'role': 'system', 'content': payload['instructions']})

    input_data = payload.get('input', [])
    if isinstance(input_data, str):
        messages.append({'role': 'user', 'content': input_data})
    elif isinstance(input_data, list):
        _convert_input_items(input_data, messages)

    result: JsonDict = {
        'model': payload.get('model', ''),
        'messages': messages,
        'stream': payload.get('stream', False),
    }
    _copy_request_options(payload, result)
    return result


def cc_to_responses_request(payload: JsonDict) -> JsonDict:
    """将 Chat Completions 请求转换为原生 Responses 请求。"""
    instructions: list[str] = []
    input_items: list[JsonDict] = []

    for message in payload.get('messages', []):
        _append_responses_input_item(message, instructions, input_items)

    result: JsonDict = {
        'model': payload.get('model', ''),
        'input': input_items,
        'stream': payload.get('stream', False),
    }
    if instructions:
        result['instructions'] = '\n\n'.join(instructions)
    _copy_responses_request_options(payload, result)
    return result


# ═══════════════════════════════════════════════════════════
#  非流式响应转换: CC → Responses
# ═══════════════════════════════════════════════════════════


def cc_to_responses(cc_resp: JsonDict, model: str = '') -> JsonDict:
    """将 Chat Completions 响应转换为 Responses 格式。"""
    choice = (cc_resp.get('choices') or [{}])[0]
    message = choice.get('message') or {}
    finish_reason = choice.get('finish_reason', 'stop')

    return {
        'id': cc_resp.get('id', gen_id('resp_')),
        'object': 'response',
        'status': _response_status_from_finish_reason(finish_reason),
        'model': model or cc_resp.get('model', ''),
        'output': _build_responses_output(message),
        'usage': _build_responses_usage(cc_resp.get('usage', {})),
    }


def responses_to_cc_response(response_data: JsonDict, model: str = '') -> JsonDict:
    """将原生 Responses 非流式响应转换为 Chat Completions 响应。"""
    output_items = response_data.get('output', [])
    content_text, reasoning_text, tool_calls = _collect_cc_parts_from_responses_output(output_items)
    finish_reason = _cc_finish_reason_from_responses(response_data, tool_calls)
    message = {
        'role': 'assistant',
        'content': content_text or None,
    }
    if reasoning_text:
        message['reasoning_content'] = reasoning_text
    if tool_calls:
        message['tool_calls'] = tool_calls

    usage = response_data.get('usage', {})
    return {
        'id': response_data.get('id', gen_id('chatcmpl-')),
        'object': 'chat.completion',
        'model': model or response_data.get('model', ''),
        'choices': [{
            'index': 0,
            'message': message,
            'finish_reason': finish_reason,
        }],
        'usage': {
            'prompt_tokens': usage.get('input_tokens', 0),
            'completion_tokens': usage.get('output_tokens', 0),
            'total_tokens': usage.get('total_tokens', 0),
        },
    }


# ═══════════════════════════════════════════════════════════
#  流式转换器: CC chunks / Anthropic SSE → Responses SSE
# ═══════════════════════════════════════════════════════════


@dataclass
class _ToolBuffer:
    """缓存单个工具调用的流式状态。

    Responses 风格的 function_call 会把名称、参数增量和完成时机拆散在多个事件里，
    转换器需要用这个缓冲结构暂存工具标识与累计参数，便于后续按顺序补齐事件。
    """

    name: str
    args: str
    call_id: str
    fc_id: str


class ResponsesStreamConverter:
    """有状态转换器：将 CC 流式 chunk、Anthropic SSE 或原生 Responses SSE 转换为 Responses SSE。

    这个类是 `/v1/responses` 流式链路里的核心状态机，负责把不同来源的增量统一整理成
    Responses 风格的事件序列。之所以必须做成有状态类，而不是简单的事件映射函数，是因为：
    - Responses 协议按“输出项生命周期”发事件，而不是只按文本顺序吐 token
    - 思考内容、普通文本、工具调用三类输出会彼此交错，需要维护关闭顺序
    - 工具调用的名称、参数增量、完成事件分散在多个事件里，必须用缓冲区重组
    - 同一个转换器既要处理 CC chunk，也要处理 Anthropic SSE 和原生 Responses SSE

    这里内部主要维护三组状态：
    1. reasoning 输出项缓冲
    2. assistant 文本输出项缓冲
    3. function_call 输出项缓冲

    最终目标是保证前端或调用方看到的 Responses 事件顺序稳定、字段完整、状态闭合。
    """

    def __init__(self, response_id: str | None = None, model: str = ''):
        """初始化 Responses 流式状态机所需的各类缓冲区与标识。"""
        self.resp_id = response_id or gen_id('resp_')
        self.model = model

        self._rs_buf = ''
        self._rs_started = False
        self._rs_closed = False
        self._rs_id = gen_id('rs_')

        self._text_buf = ''
        self._text_started = False
        self._text_closed = False
        self._msg_id = gen_id('msg_')

        self._tools: dict[int, _ToolBuffer] = {}

        self._output_items: list[JsonDict] = []
        self._finished = False
        self._input_tokens = 0

    def start_events(self) -> list[str]:
        """生成 Responses 流式生命周期的起始事件。

        调用方在真正转发上游流之前应先发出这个事件，确保前端先拿到一个
        `response.created`，后续再逐步追加 output_item / delta / completed。
        """
        return [self._sse('response.created', {
            'id': self.resp_id,
            'object': 'response',
            'status': 'in_progress',
            'model': self.model,
            'output': [],
        })]

    def process_cc_chunk(self, chunk: JsonDict) -> list[str]:
        """处理单个 Chat Completions chunk。

        这个入口用于“先转成 CC，再转成 Responses”的链路。它会把一个 chunk 中的
        reasoning、文本、tool_calls 和 finish_reason 依次拆解成 Responses 事件。
        """
        events: list[str] = []
        usage = chunk.get('usage')
        for choice in chunk.get('choices') or []:
            events.extend(self._process_cc_choice(choice, usage))
        return events

    def process_anthropic_event(self, event_type: str, event_data: JsonDict) -> list[str]:
        """直接处理 Anthropic SSE 事件。

        这里故意跳过 CC 中间态，是为了减少一次协议重组，尽量保留 Anthropic 原始事件
        的时序关系，同时降低流式转换的中间状态复杂度。
        """
        if event_type == 'message_start':
            return self._handle_anthropic_message_start(event_data)
        if event_type == 'content_block_start':
            return self._handle_anthropic_content_block_start(event_data)
        if event_type == 'content_block_delta':
            return self._handle_anthropic_content_block_delta(event_data)
        if event_type == 'message_delta':
            return self._handle_anthropic_message_delta(event_data)
        return []

    def finalize(self) -> list[str]:
        """在上游流自然结束但尚未显式完成时，补发收尾事件。

        有些后端不会补齐所有 completed 事件，这里统一把尚未关闭的 reasoning / text /
        tool_call 项收尾，避免前端看到半开的输出项。
        """
        if self._finished:
            return []
        self._finished = True
        return self._finish_stream('stop', None)

    def process_responses_event(self, event_type: str, event_data: JsonDict) -> list[str]:
        """处理上游原生 Responses SSE 事件。

        当当前链路本身就是 `/v1/responses -> /v1/responses` 时，这里主要做轻量重写：
        保持事件结构不变，只把顶层模型名改成 Cursor 侧看到的展示模型名。
        """
        if event_type == 'response.created':
            return [self._sse(event_type, self._rewrite_top_level_model(event_data))]
        if event_type == 'response.completed':
            self._finished = True
            return [self._sse(event_type, self._rewrite_top_level_model(event_data))]
        return [self._sse(event_type, event_data)]

    def _process_cc_choice(self, choice: JsonDict, usage: Any) -> list[str]:
        """处理单个 CC choice。

        同一个 chunk 里可能同时携带文本增量、思考增量、工具调用增量和结束原因；
        这里按 Responses 协议要求的输出项顺序，把它们拆成一组更细粒度的 SSE 事件。
        """
        events: list[str] = []
        delta = choice.get('delta') or {}
        finish_reason = choice.get('finish_reason')

        if delta.get('reasoning_content'):
            events.extend(self._append_reasoning_delta(delta['reasoning_content']))
        if delta.get('content') is not None and delta['content'] != '':
            events.extend(self._append_text_delta(delta['content']))
        for tool_call in delta.get('tool_calls') or []:
            events.extend(self._on_tool_call(tool_call))

        if finish_reason and not self._finished:
            self._finished = True
            events.extend(self._finish_stream(finish_reason, usage))
        return events

    def _handle_anthropic_message_start(self, event_data: JsonDict) -> list[str]:
        """记录输入令牌统计。

        Anthropic 会在消息开始事件里给出 input_tokens，这里先缓存下来，等消息结束时再与
        output_tokens 合并成 Responses 需要的 usage 结构。
        """
        usage = event_data.get('message', {}).get('usage', {})
        self._input_tokens = usage.get('input_tokens', 0)
        return []

    def _handle_anthropic_content_block_start(self, event_data: JsonDict) -> list[str]:
        """处理 Anthropic 内容块起始事件。

        不同 block 类型会开启不同的 Responses 输出项：
        - thinking → reasoning 项
        - text → message 项
        - tool_use → function_call 项
        """
        block = event_data.get('content_block', {})
        block_type = block.get('type', '')

        if block_type == 'thinking':
            return self._ensure_reasoning_started()
        if block_type == 'text':
            return self._ensure_text_started()
        if block_type == 'tool_use':
            return self._start_tool_from_block(block)
        return []

    def _handle_anthropic_content_block_delta(self, event_data: JsonDict) -> list[str]:
        """处理 Anthropic 内容块增量事件。

        这里负责把 thinking_delta / text_delta / input_json_delta 映射成 Responses 的
        summary_text.delta / output_text.delta / function_call_arguments.delta。
        """
        delta = event_data.get('delta', {})
        delta_type = delta.get('type', '')

        if delta_type == 'thinking_delta' and delta.get('thinking'):
            return self._append_reasoning_delta(delta['thinking'])
        if delta_type == 'text_delta' and delta.get('text'):
            return self._append_text_delta(delta['text'])
        if delta_type == 'input_json_delta' and delta.get('partial_json') and self._tools:
            index = max(self._tools.keys())
            return self._append_tool_arguments(index, delta['partial_json'])
        return []

    def _handle_anthropic_message_delta(self, event_data: JsonDict) -> list[str]:
        """处理 Anthropic 消息结束事件。

        当上游发出 stop_reason 后，这里会把当前缓冲区统一收尾，并补出最终的 usage 和
        `response.completed` 事件。
        """
        if self._finished:
            return []

        delta = event_data.get('delta', {})
        usage = event_data.get('usage', {})
        finish_reason = _map_anthropic_stop_reason(delta.get('stop_reason', 'end_turn'))
        self._finished = True

        usage_payload = {
            'input_tokens': self._input_tokens,
            'output_tokens': usage.get('output_tokens', 0),
            'total_tokens': self._input_tokens + usage.get('output_tokens', 0),
        }
        return self._finish_stream(finish_reason, usage_payload)

    def _ensure_reasoning_started(self) -> list[str]:
        """确保 reasoning 输出项已经创建。

        Responses 协议要求先有 `output_item.added`，后面才能合法发送 reasoning 的 delta。
        所以这里负责幂等地创建 reasoning 项。
        """
        if self._rs_started:
            return []
        self._rs_started = True
        return [self._sse('response.output_item.added', {
            'type': 'reasoning',
            'id': self._rs_id,
            'summary': [],
        })]

    def _append_reasoning_delta(self, text: str) -> list[str]:
        """向 reasoning 输出项追加思考内容增量。"""
        events = self._ensure_reasoning_started()
        self._rs_buf += text
        events.append(self._sse('response.reasoning_summary_text.delta', {
            'type': 'summary_text',
            'delta': text,
        }))
        return events

    def _append_text_delta(self, text: str) -> list[str]:
        """向 assistant 文本输出项追加文本增量。"""
        events = self._ensure_text_started()
        self._text_buf += text
        events.append(self._sse('response.output_text.delta', {
            'type': 'output_text',
            'delta': text,
        }))
        return events

    def _on_tool_call(self, tool_call: JsonDict) -> list[str]:
        """处理来自 CC chunk 的工具调用增量，并映射成 Responses function_call 事件。"""
        events: list[str] = []
        index = tool_call.get('index', 0)
        function_data = tool_call.get('function') or {}

        if index not in self._tools:
            call_id = tool_call.get('id', gen_id('call_'))
            name = function_data.get('name', '')
            events.extend(self._start_tool(index=index, call_id=call_id, name=name))

        if function_data.get('name'):
            self._tools[index].name = function_data['name']
        if function_data.get('arguments', ''):
            events.extend(self._append_tool_arguments(index, function_data['arguments']))
        return events

    def _ensure_text_started(self) -> list[str]:
        """确保 assistant 文本输出项已开启，并在必要时先关闭 reasoning 项。"""
        events: list[str] = []
        if self._rs_started and not self._rs_closed:
            events.extend(self._close_reasoning())
        if not self._text_started:
            self._text_started = True
            events.append(self._sse('response.output_item.added', {
                'type': 'message',
                'id': self._msg_id,
                'status': 'in_progress',
                'role': 'assistant',
                'content': [],
            }))
            events.append(self._sse('response.content_part.added', {
                'type': 'output_text',
                'text': '',
            }))
        return events

    def _start_tool_from_block(self, block: JsonDict) -> list[str]:
        """根据 Anthropic `tool_use` block 创建对应的 function_call 输出项。"""
        return self._start_tool(
            index=len(self._tools),
            call_id=block.get('id', gen_id('toolu_')),
            name=block.get('name', ''),
        )

    def _start_tool(self, *, index: int, call_id: str, name: str) -> list[str]:
        """开启一个新的 Responses function_call 输出项，并关闭前置输出段。"""
        events: list[str] = []
        if self._rs_started and not self._rs_closed:
            events.extend(self._close_reasoning())
        if self._text_started and not self._text_closed:
            events.extend(self._close_text())

        function_call_id = gen_id('fc_')
        self._tools[index] = _ToolBuffer(
            name=name,
            args='',
            call_id=call_id,
            fc_id=function_call_id,
        )
        events.append(self._sse('response.output_item.added', {
            'type': 'function_call',
            'id': function_call_id,
            'status': 'in_progress',
            'call_id': call_id,
            'name': name,
            'arguments': '',
        }))
        return events

    def _append_tool_arguments(self, index: int, arguments_delta: str) -> list[str]:
        """向指定 function_call 缓冲区追加参数增量，并发出对应 SSE 事件。"""
        buffer = self._tools[index]
        buffer.args += arguments_delta
        return [self._sse('response.function_call_arguments.delta', {
            'type': 'function_call',
            'delta': arguments_delta,
        })]

    def _close_reasoning(self) -> list[str]:
        """关闭 reasoning 输出项，并补发 summary 完成事件。"""
        if self._rs_closed:
            return []
        self._rs_closed = True

        reasoning_item = {
            'type': 'reasoning',
            'id': self._rs_id,
            'summary': [{'type': 'summary_text', 'text': self._rs_buf}],
        }
        self._output_items.append(reasoning_item)
        return [
            self._sse('response.reasoning_summary_text.done', {
                'type': 'summary_text',
                'text': self._rs_buf,
            }),
            self._sse('response.output_item.done', reasoning_item),
        ]

    def _close_text(self) -> list[str]:
        """关闭 assistant 文本输出项，并补发文本完成事件。"""
        if self._text_closed:
            return []
        self._text_closed = True

        message_item = {
            'type': 'message',
            'id': self._msg_id,
            'status': 'completed',
            'role': 'assistant',
            'content': [{'type': 'output_text', 'text': self._text_buf}],
        }
        self._output_items.append(message_item)
        return [
            self._sse('response.output_text.done', {
                'type': 'output_text',
                'text': self._text_buf,
            }),
            self._sse('response.output_item.done', message_item),
        ]

    def _finish_stream(self, finish_reason: str, usage: Any) -> list[str]:
        """统一关闭所有未完成输出项，并发出最终 completed 事件。

        这是整个状态机的统一收口点：无论结束来源是 CC、Anthropic 还是手动 finalize，
        最终都通过这里补齐 reasoning / text / tool_call 的 done 事件。
        """
        events: list[str] = []
        if self._rs_started and not self._rs_closed:
            events.extend(self._close_reasoning())
        if self._text_started and not self._text_closed:
            events.extend(self._close_text())
        events.extend(self._finish_tool_calls())

        usage_data = usage if isinstance(usage, dict) else {}
        events.append(self._sse('response.completed', {
            'id': self.resp_id,
            'object': 'response',
            'status': _response_status_from_finish_reason(finish_reason),
            'model': self.model,
            'output': self._output_items,
            'usage': usage_data,
        }))
        return events

    def _finish_tool_calls(self) -> list[str]:
        """关闭所有尚未完成的 function_call 输出项。"""
        events: list[str] = []
        for index in sorted(self._tools.keys()):
            buffer = self._tools[index]
            events.append(self._sse('response.function_call_arguments.done', {
                'type': 'function_call',
                'arguments': buffer.args,
            }))
            function_call_item = {
                'type': 'function_call',
                'id': buffer.fc_id,
                'status': 'completed',
                'call_id': buffer.call_id,
                'name': buffer.name,
                'arguments': buffer.args,
            }
            events.append(self._sse('response.output_item.done', function_call_item))
            self._output_items.append(function_call_item)
        return events

    def _sse(self, event_type: str, data: JsonDict) -> str:
        """将事件类型与负载编码为标准 Responses SSE 字符串。"""
        return f'event: {event_type}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n'

    def _rewrite_top_level_model(self, payload: JsonDict) -> JsonDict:
        """在保持上游事件结构不变的前提下回填展示模型名。

        Responses 原生事件经常会在顶层带一个 `model` 字段；这里不重写内部结构，只把这个
        顶层字段替换为当前映射的 Cursor 模型名，避免界面显示上游原始模型名。
        """
        if not isinstance(payload, dict):
            return payload
        copied = dict(payload)
        if copied.get('model'):
            copied['model'] = self.model or copied['model']
        return copied


class ResponsesToCCStreamConverter:
    """将原生 Responses SSE 事件转换为 Chat Completions chunk。

    当上游只支持 `/v1/responses` 时，聊天补全接口需要把 Responses 世界中的事件重新映射回
    OpenAI Chat Completions 的流式 chunk，保证 Cursor 仍然能按聊天补全协议消费结果。

    这个类主要负责三件事：
    1. 把 Responses 的文本与思考增量映射为 `content` / `reasoning_content` delta
    2. 把 function_call 项与 arguments 增量重组为 `tool_calls` 增量
    3. 在 response.completed 时补出 finish_reason 与 usage
    """

    def __init__(self, request_id: str | None = None, model: str = ''):
        """初始化 Responses → Chat Completions 流式转换所需的状态。"""
        self._id = request_id or gen_id('chatcmpl-')
        self._model = model
        self._tool_index = 0
        self._tool_slots: dict[str, int] = {}
        self._usage: JsonDict = {}

    def process_event(self, event_type: str, event_data: JsonDict) -> list[JsonDict]:
        """处理单个 Responses 事件并产出一个或多个 CC chunk。

        这里按事件类型分发，而不是一次性拼完整响应，是为了让聊天补全调用方仍然能保留
        原生的流式体验。
        """
        if event_type == 'response.created':
            return [self._make_chunk(delta={'role': 'assistant', 'content': ''})]
        if event_type == 'response.output_text.delta':
            return [self._make_chunk(delta={'content': event_data.get('delta', '')})]
        if event_type == 'response.reasoning_summary_text.delta':
            return [self._make_chunk(delta={'reasoning_content': event_data.get('delta', '')})]
        if event_type == 'response.output_item.added':
            return self._handle_output_item_added(event_data)
        if event_type == 'response.function_call_arguments.delta':
            return self._handle_function_arguments_delta(event_data)
        if event_type == 'response.completed':
            return self._handle_completed(event_data)
        return []

    def _handle_output_item_added(self, event_data: JsonDict) -> list[JsonDict]:
        """处理 Responses 的 output_item.added 事件。

        这里主要关心 function_call，因为文本和 reasoning 的真正内容会分别在后续 delta
        事件里补全；function_call 则需要先创建一个空参数的 tool_call 槽位。
        """
        if event_data.get('type') != 'function_call':
            return []
        call_id = event_data.get('call_id') or gen_id('call_')
        index = self._tool_slots.setdefault(call_id, self._tool_index)
        if index == self._tool_index:
            self._tool_index += 1
        return [self._make_chunk(delta={
            'tool_calls': [{
                'index': index,
                'id': call_id,
                'type': 'function',
                'function': {
                    'name': event_data.get('name', ''),
                    'arguments': '',
                },
            }]
        })]

    def _handle_function_arguments_delta(self, event_data: JsonDict) -> list[JsonDict]:
        """处理工具参数增量，并追加到最近一次打开的 tool_call 上。"""
        if not self._tool_slots:
            return []
        index = max(self._tool_slots.values())
        return [self._make_chunk(delta={
            'tool_calls': [{
                'index': index,
                'function': {
                    'arguments': event_data.get('delta', ''),
                },
            }]
        })]

    def _handle_completed(self, event_data: JsonDict) -> list[JsonDict]:
        """处理 response.completed，补出聊天补全流的最终收尾 chunk。"""
        self._usage = event_data.get('usage', {}) or {}
        finish_reason = 'tool_calls' if any(
            isinstance(item, dict) and item.get('type') == 'function_call'
            for item in event_data.get('output', [])
        ) else 'stop'
        chunk = self._make_chunk(delta={}, finish_reason=finish_reason)
        chunk['usage'] = {
            'prompt_tokens': self._usage.get('input_tokens', 0),
            'completion_tokens': self._usage.get('output_tokens', 0),
            'total_tokens': self._usage.get('total_tokens', 0),
        }
        return [chunk]

    def _make_chunk(self, delta: JsonDict, finish_reason: str | None = None) -> JsonDict:
        """构造标准 Chat Completions chunk。"""
        choice: JsonDict = {'index': 0, 'delta': delta}
        if finish_reason:
            choice['finish_reason'] = finish_reason
        return {
            'id': self._id,
            'object': 'chat.completion.chunk',
            'model': self._model,
            'choices': [choice],
        }


# ═══════════════════════════════════════════════════════════
#  请求转换辅助
# ═══════════════════════════════════════════════════════════


def _copy_request_options(payload: JsonDict, result: JsonDict) -> None:
    """将 Responses 请求中的通用选项复制到 CC 请求体。"""
    if 'tools' in payload:
        result['tools'] = _convert_tools(payload['tools'])
    for key in ('temperature', 'top_p'):
        if key in payload:
            result[key] = payload[key]
    if 'max_output_tokens' in payload:
        result['max_tokens'] = payload['max_output_tokens']
    if 'tool_choice' in payload:
        result['tool_choice'] = payload['tool_choice']


def _copy_responses_request_options(payload: JsonDict, result: JsonDict) -> None:
    """将聊天补全请求中的通用选项复制到原生 Responses 请求体。"""
    if 'tools' in payload:
        result['tools'] = _convert_cc_tools_to_responses(payload['tools'])
    for key in ('temperature', 'top_p', 'tool_choice'):
        if key in payload:
            result[key] = payload[key]
    if 'max_tokens' in payload:
        result['max_output_tokens'] = payload['max_tokens']


def _append_responses_input_item(
    message: Any,
    instructions: list[str],
    input_items: list[JsonDict],
) -> None:
    """将单条 Chat Completions 消息追加为 Responses `input` 项。"""
    if not isinstance(message, dict):
        return

    role = message.get('role', '')
    content = message.get('content')

    if role == 'system':
        text = _content_to_text(content)
        if text:
            instructions.append(text)
        return

    if role == 'tool':
        input_items.append({
            'type': 'function_call_output',
            'call_id': message.get('tool_call_id', ''),
            'output': _stringify_output(content),
        })
        return

    item: JsonDict = {
        'type': 'message',
        'role': role or 'user',
        'content': _content_to_responses_parts(content),
    }
    input_items.append(item)

    if role == 'assistant':
        for tool_call in message.get('tool_calls') or []:
            input_items.append(_build_responses_function_call_item(tool_call))


def _convert_input_items(items: list[Any], messages: list[JsonDict]) -> None:
    """将 Responses `input` 数组重建为 Chat Completions `messages` 列表。"""
    index = 0
    while index < len(items):
        item = items[index]

        if isinstance(item, str):
            messages.append({'role': 'user', 'content': item})
            index += 1
            continue

        if not isinstance(item, dict):
            index += 1
            continue

        item_type = item.get('type', '')
        role = item.get('role', '')

        if role and not item_type:
            messages.append({
                'role': role,
                'content': _normalize_simple_content(item.get('content', '')),
            })
            index += 1
            continue

        if item_type == 'message':
            consumed = _append_message_item(items, start=index, messages=messages)
            index += consumed
            continue

        if item_type == 'function_call':
            _append_function_call_item(item, messages)
            index += 1
            continue

        if item_type == 'function_call_output':
            messages.append(_convert_function_call_output_item(item))
            index += 1
            continue

        if role:
            messages.append({'role': role, 'content': str(item.get('content', ''))})
        index += 1


def _append_message_item(items: list[Any], *, start: int, messages: list[JsonDict]) -> int:
    """将一个 message 项及其后续连续 function_call 项合并成一条消息。"""
    item = items[start]
    role = item.get('role', 'assistant')
    content = _extract_text(item.get('content', []))
    message: JsonDict = {'role': role, 'content': content or ''}

    if role == 'assistant':
        tool_calls, consumed = _collect_function_calls(items, start + 1)
        if tool_calls:
            message['tool_calls'] = tool_calls
            if not message['content']:
                message['content'] = None
            messages.append(message)
            return 1 + consumed

    messages.append(message)
    return 1


def _append_function_call_item(item: JsonDict, messages: list[JsonDict]) -> None:
    """将独立的 Responses `function_call` 项挂接到最近的 assistant 消息上。"""
    tool_call = _build_cc_tool_call(item)

    if messages and messages[-1]['role'] == 'assistant':
        messages[-1].setdefault('tool_calls', []).append(tool_call)
        if not messages[-1].get('content'):
            messages[-1]['content'] = None
        return

    messages.append({'role': 'assistant', 'content': None, 'tool_calls': [tool_call]})


def _convert_function_call_output_item(item: JsonDict) -> JsonDict:
    """将 Responses 的 `function_call_output` 项转换为 OpenAI `tool` 消息。"""
    output = item.get('output', '')
    if not isinstance(output, str):
        output = json.dumps(output, ensure_ascii=False)
    return {
        'role': 'tool',
        'tool_call_id': item.get('call_id', ''),
        'content': output,
    }


def _normalize_simple_content(content: Any) -> str:
    """将简单 content 载荷规范化为纯文本字符串。"""
    if isinstance(content, list):
        return _extract_text(content) or ''
    return str(content) if content is not None else ''


def _collect_function_calls(items: list[Any], start: int) -> tuple[list[JsonDict], int]:
    """收集从指定位置开始连续出现的 `function_call` 项。"""
    tool_calls: list[JsonDict] = []
    index = start
    while index < len(items):
        next_item = items[index]
        if isinstance(next_item, dict) and next_item.get('type') == 'function_call':
            tool_calls.append(_build_cc_tool_call(next_item))
            index += 1
        else:
            break
    return tool_calls, index - start


def _build_cc_tool_call(item: JsonDict) -> JsonDict:
    """将单个 Responses `function_call` 项转换为 CC `tool_call` 结构。"""
    return {
        'id': item.get('call_id') or gen_id('call_'),
        'type': 'function',
        'function': {
            'name': item.get('name', ''),
            'arguments': item.get('arguments', '{}'),
        },
    }


# ═══════════════════════════════════════════════════════════
#  非流式响应转换辅助
# ═══════════════════════════════════════════════════════════


def _build_responses_output(message: JsonDict) -> list[JsonDict]:
    """将单条聊天补全消息展开为 Responses `output` 数组。"""
    output: list[JsonDict] = []

    if message.get('reasoning_content'):
        output.append(_make_reasoning_output_item(message['reasoning_content']))
    if message.get('content'):
        output.append(_make_message_output_item(message['content']))
    for tool_call in message.get('tool_calls') or []:
        output.append(_make_function_call_output_item(tool_call))

    return output


def _make_reasoning_output_item(text: str) -> JsonDict:
    """构造单个 Responses reasoning 输出项。"""
    return {
        'type': 'reasoning',
        'id': gen_id('rs_'),
        'summary': [{'type': 'summary_text', 'text': text}],
    }


def _make_message_output_item(text: str) -> JsonDict:
    """构造单个 Responses message 输出项。"""
    return {
        'type': 'message',
        'id': gen_id('msg_'),
        'status': 'completed',
        'role': 'assistant',
        'content': [{'type': 'output_text', 'text': text}],
    }


def _make_function_call_output_item(tool_call: JsonDict) -> JsonDict:
    """构造单个 Responses function_call 输出项。"""
    function_data = tool_call.get('function') or {}
    return {
        'type': 'function_call',
        'id': gen_id('fc_'),
        'status': 'completed',
        'call_id': tool_call.get('id', gen_id('call_')),
        'name': function_data.get('name', ''),
        'arguments': function_data.get('arguments', '{}'),
    }


def _build_responses_usage(usage: JsonDict) -> JsonDict:
    """将 Chat Completions 的 usage 字段映射为 Responses usage 结构。"""
    return {
        'input_tokens': usage.get('prompt_tokens', 0),
        'output_tokens': usage.get('completion_tokens', 0),
        'total_tokens': usage.get('total_tokens', 0),
    }


def _collect_cc_parts_from_responses_output(output_items: Any) -> tuple[str, str, list[JsonDict]]:
    """从 Responses `output` 中提取文本、思考摘要和工具调用。"""
    content_text = ''
    reasoning_text = ''
    tool_calls: list[JsonDict] = []

    if not isinstance(output_items, list):
        return content_text, reasoning_text, tool_calls

    for item in output_items:
        if not isinstance(item, dict):
            continue
        item_type = item.get('type', '')
        if item_type == 'message':
            content_text += _extract_text(item.get('content', []))
        elif item_type == 'reasoning':
            reasoning_text += _extract_reasoning_text(item)
        elif item_type == 'function_call':
            tool_calls.append(_build_cc_tool_call_from_responses_output(item, index=len(tool_calls)))

    return content_text, reasoning_text, tool_calls


def _extract_reasoning_text(item: JsonDict) -> str:
    """从 Responses reasoning 项中拼接出完整的摘要文本。"""
    summary = item.get('summary', [])
    if not isinstance(summary, list):
        return ''
    texts: list[str] = []
    for part in summary:
        if isinstance(part, dict) and part.get('type') == 'summary_text':
            texts.append(part.get('text', ''))
    return ''.join(texts)


def _build_cc_tool_call_from_responses_output(item: JsonDict, *, index: int) -> JsonDict:
    """将 Responses `function_call` 输出项转换为 CC `tool_call`。"""
    return {
        'index': index,
        'id': item.get('call_id') or gen_id('call_'),
        'type': 'function',
        'function': {
            'name': item.get('name', ''),
            'arguments': item.get('arguments', '{}'),
        },
    }


def _cc_finish_reason_from_responses(response_data: JsonDict, tool_calls: list[JsonDict]) -> str:
    """根据 Responses 完成状态推断聊天补全的 finish_reason。"""
    if tool_calls:
        return 'tool_calls'
    if response_data.get('status') == 'incomplete':
        return 'length'
    return 'stop'


def _response_status_from_finish_reason(finish_reason: str) -> str:
    """将聊天补全 finish_reason 映射为 Responses 顶层状态。"""
    return 'incomplete' if finish_reason == 'length' else 'completed'


def _map_anthropic_stop_reason(stop_reason: str) -> str:
    """将 Anthropic 的 stop_reason 映射为聊天补全风格的结束原因。"""
    return {'tool_use': 'tool_calls', 'max_tokens': 'length'}.get(stop_reason, 'stop')


# ═══════════════════════════════════════════════════════════
#  通用辅助
# ═══════════════════════════════════════════════════════════


def _extract_text(content: Any) -> str:
    """从多种内容块结构中提取并拼接纯文本。"""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return str(content) if content else ''

    texts: list[str] = []
    for part in content:
        if isinstance(part, str):
            texts.append(part)
        elif isinstance(part, dict):
            part_type = part.get('type', '')
            if part_type in ('output_text', 'input_text', 'text'):
                texts.append(part.get('text', ''))
            elif part_type == 'refusal':
                texts.append(part.get('refusal', ''))
    return '\n'.join(texts) if texts else ''


def _content_to_text(content: Any) -> str:
    """将任意 content 载荷转换为单个字符串。"""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return _extract_text(content)
    return str(content) if content is not None else ''


def _content_to_responses_parts(content: Any) -> list[JsonDict]:
    """将普通消息内容转换为 Responses `input_text` 数组。"""
    if isinstance(content, list):
        text = _extract_text(content)
    else:
        text = _content_to_text(content)
    return [{'type': 'input_text', 'text': text}] if text else []


def _stringify_output(content: Any) -> str:
    """将工具输出统一序列化为字符串，便于放入 `function_call_output`。"""
    if isinstance(content, str):
        return content
    if content is None:
        return ''
    return json.dumps(content, ensure_ascii=False) if not isinstance(content, str) else content


def _build_responses_function_call_item(tool_call: JsonDict) -> JsonDict:
    """将 CC `tool_call` 结构转换为 Responses `function_call` 输入项。"""
    function_data = tool_call.get('function') or {}
    return {
        'type': 'function_call',
        'call_id': tool_call.get('id', gen_id('call_')),
        'name': function_data.get('name', ''),
        'arguments': function_data.get('arguments', '{}'),
    }


def _convert_cc_tools_to_responses(tools: Any) -> list[JsonDict]:
    """将聊天补全风格的工具定义转换为 Responses `tools` 列表。"""
    if not isinstance(tools, list):
        return []

    result: list[JsonDict] = []
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        if tool.get('type') == 'function' and 'function' in tool:
            function_data = tool['function']
            result.append({
                'type': 'function',
                'name': function_data.get('name', ''),
                'description': function_data.get('description', ''),
                'parameters': function_data.get('parameters', {'type': 'object', 'properties': {}}),
            })
        elif tool.get('type') == 'function':
            result.append(tool)
    return result


def _convert_tools(tools: Any) -> list[JsonDict]:
    """规范化 Responses 请求中的工具定义列表。"""
    if not isinstance(tools, list):
        return []

    result: list[JsonDict] = []
    for tool in tools:
        converted = _convert_tool_definition(tool)
        if converted is not None:
            result.append(converted)
    return result


def _convert_tool_definition(tool: Any) -> JsonDict | None:
    """将扁平工具定义补成标准 Chat Completions `function` 工具格式。"""
    if not isinstance(tool, dict):
        return None
    if tool.get('type') != 'function':
        return None
    if 'function' in tool:
        return tool
    return {
        'type': 'function',
        'function': {
            'name': tool.get('name', ''),
            'description': tool.get('description', ''),
            'parameters': tool.get('parameters', {'type': 'object', 'properties': {}}),
        },
    }
