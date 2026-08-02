"""Cursor system 提示词注入。"""

from __future__ import annotations

import re
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Literal

from llm_rosetta.types.ir import IRRequest

from .exchange import CursorDialect
from ..settings.schema import DialectInstruction

InjectionMode = Literal['prepend', 'append', 'replace']
InjectionState = Literal['applied', 'disabled', 'failed']


@dataclass(frozen=True, slots=True)
class SystemBlock:
    id: str
    label: str
    description: str


@dataclass(frozen=True, slots=True)
class InjectionResult:
    state: InjectionState
    dialect: CursorDialect
    target: str
    mode: InjectionMode
    message: str

    @property
    def warnings(self) -> list[str]:
        return [self.message] if self.state == 'failed' else []


class InstructionStatusTracker:
    def __init__(self):
        self._lock = threading.RLock()
        self._statuses: dict[str, dict[CursorDialect, dict[str, str]]] = {}

    def record(self, model: str, result: InjectionResult) -> None:
        with self._lock:
            self._statuses.setdefault(model, {})[result.dialect] = {
                'state': result.state,
                'target': result.target,
                'mode': result.mode,
                'message': result.message,
                'updated_at': _now(),
            }

    def get(self, model: str) -> dict[str, dict[str, str]]:
        with self._lock:
            return {
                dialect: dict(status)
                for dialect, status in self._statuses.get(model, {}).items()
            }

    def clear(self, *models: str) -> None:
        with self._lock:
            for model in models:
                self._statuses.pop(model, None)

    def reconcile(
        self,
        model: str,
        dialect: CursorDialect,
        rule: DialectInstruction,
    ) -> dict[str, str] | None:
        with self._lock:
            status = self._statuses.get(model, {}).get(dialect)
            if not status:
                return None
            if (
                status['target'] != rule.target
                or status['mode'] != rule.mode
            ):
                return None
            return dict(status)


ALL_BLOCK = SystemBlock(
    id='all',
    label='全部',
    description='整段 system（Responses 上游即整个 instructions 字段）',
)

FUNCTION_BLOCKS: tuple[SystemBlock, ...] = (
    ALL_BLOCK,
    SystemBlock(
        id='system-communication',
        label='system-communication',
        description='系统附加上下文怎么处理（system_reminder 等勿对用户提及、忽略 timestamp 继续工作）',
    ),
    SystemBlock(
        id='tone_and_style',
        label='tone_and_style',
        description='回复语气与排版（代码反引号、数学公式、PR 链接、图片嵌入等）',
    ),
    SystemBlock(
        id='tool_calling',
        label='tool_calling',
        description='优先用专用工具而非终端命令；文件操作工具选用规则',
    ),
    SystemBlock(
        id='citing_code',
        label='citing_code',
        description='代码引用格式（CODE REFERENCES vs markdown 代码块）',
    ),
    SystemBlock(
        id='inline_line_numbers',
        label='inline_line_numbers',
        description='如何理解工具/用户内容里的 LINE_NUMBER| 行号前缀',
    ),
    SystemBlock(
        id='terminal_files_information',
        label='terminal_files_information',
        description='terminals 文件夹用途；勿向用户提及该路径',
    ),
    SystemBlock(
        id='ask_question_guidance',
        label='ask_question_guidance',
        description='何时用 AskQuestion 收集结构化选择题',
    ),
    SystemBlock(
        id='dynamic_tools',
        label='dynamic_tools',
        description='动态工具/MCP 发现与调用（GetDynamicTools / CallDynamicTool）',
    ),
)

CUSTOM_GRAMMAR_BLOCKS: tuple[SystemBlock, ...] = (
    ALL_BLOCK,
    SystemBlock(
        id='epistemic_rigor',
        label='epistemic_rigor',
        description='认知严谨——不盲从用户前提、核实主张、按真实目标而非字面执行',
    ),
    SystemBlock(
        id='general',
        label='general',
        description='通用工作约定（附加上下文、Shell 会话、优先专用工具、行号前缀等）',
    ),
    SystemBlock(
        id='getting_work_done',
        label='getting_work_done',
        description='推进任务——搜索工具优先、并行工具调用、避免嘈杂 shell 输出',
    ),
    SystemBlock(
        id='technical_communication',
        label='technical_communication',
        description='技术表达——先结论、控制细节、少用黑话',
    ),
    SystemBlock(
        id='system-communication',
        label='system-communication',
        description='系统附加上下文与 @ 引用；勿对用户复述系统标签',
    ),
    SystemBlock(
        id='autonomy_and_persistence',
        label='autonomy_and_persistence',
        description='自主与边界——按请求模式决定是否可改文件/继续推进',
    ),
    SystemBlock(
        id='editing_constraints',
        label='editing_constraints',
        description='编辑约束（优先 ApplyPatch、禁止擅自破坏性 git）',
    ),
    SystemBlock(
        id='mode_selection',
        label='mode_selection',
        description='何时切换 Plan/Agent 等交互模式',
    ),
    SystemBlock(
        id='dynamic_tools',
        label='dynamic_tools',
        description='动态工具/MCP 发现与调用',
    ),
    SystemBlock(
        id='linter_errors',
        label='linter_errors',
        description='实质编辑后用 ReadLints 检查并修复易修问题',
    ),
    SystemBlock(
        id='terminal_files_information',
        label='terminal_files_information',
        description='terminals 文件夹用途；勿向用户提及该路径',
    ),
    SystemBlock(
        id='working_with_the_user',
        label='working_with_the_user',
        description='与用户沟通通道（commentary 中途更新 / final 最终回复）',
    ),
    SystemBlock(
        id='visualizations',
        label='visualizations',
        description='何时用可视化/Canvas，避免无意义图表',
    ),
    SystemBlock(
        id='main_goal',
        label='main_goal',
        description='主目标——遵循 <user_query> 中的用户指令',
    ),
)

SYSTEM_BLOCKS: dict[CursorDialect, tuple[SystemBlock, ...]] = {
    'function': FUNCTION_BLOCKS,
    'custom_grammar': CUSTOM_GRAMMAR_BLOCKS,
}

_VALID_TARGETS: dict[CursorDialect, frozenset[str]] = {
    dialect: frozenset(block.id for block in blocks)
    for dialect, blocks in SYSTEM_BLOCKS.items()
}


def valid_targets(dialect: CursorDialect) -> frozenset[str]:
    return _VALID_TARGETS[dialect]


def blocks_as_dicts() -> dict[str, list[dict[str, str]]]:
    return {
        dialect: [
            {'id': block.id, 'label': block.label, 'description': block.description}
            for block in blocks
        ]
        for dialect, blocks in SYSTEM_BLOCKS.items()
    }


def apply_system_injection(
    request: IRRequest,
    dialect: CursorDialect,
    rule: DialectInstruction,
) -> InjectionResult:
    """就地改写 IR system_instruction。"""
    if not rule.text:
        return InjectionResult(
            'disabled',
            dialect,
            rule.target,
            rule.mode,
            '未配置自定义指令',
        )

    if rule.target not in _VALID_TARGETS[dialect]:
        return _failed(dialect, rule, f'无效的注入目标 {rule.target!r}')
    if rule.target != 'all' and f'</{rule.target}>' in rule.text:
        return _failed(
            dialect,
            rule,
            f'注入文案不能包含 </{rule.target}>',
        )

    system = request.get('system_instruction')
    if not isinstance(system, list):
        system = []

    text_part = next(
        (
            part for part in system
            if isinstance(part, dict)
            and part.get('type') == 'text'
            and isinstance(part.get('text'), str)
        ),
        None,
    )

    if rule.target == 'all':
        if text_part is None:
            request['system_instruction'] = [
                {'type': 'text', 'text': rule.text},
                *system,
            ]
        else:
            text_part['text'] = _apply_mode(text_part['text'], rule.text, rule.mode)
        return _applied(dialect, rule)

    if text_part is None:
        return _failed(dialect, rule, 'system 中没有可编辑的文本')

    updated, found = _inject_block(text_part['text'], rule.target, rule.text, rule.mode)
    if not found:
        return _failed(dialect, rule, f'system 中未找到块 <{rule.target}>')
    text_part['text'] = updated
    return _applied(dialect, rule)


def _apply_mode(original: str, text: str, mode: InjectionMode) -> str:
    if mode == 'replace':
        return text
    if mode == 'append':
        if not original:
            return text
        separator = '' if original.endswith('\n') or text.startswith('\n') else '\n'
        return f'{original}{separator}{text}'
    if not original:
        return text
    separator = '' if text.endswith('\n') or original.startswith('\n') else '\n'
    return f'{text}{separator}{original}'


def _merge_block_content(
    original: str,
    text: str,
    mode: InjectionMode,
) -> str:
    if mode == 'replace':
        return text
    if mode == 'append':
        return f'{original}\n{text}' if original else text
    return f'{text}\n{original}' if original else text


def _inject_block(
    content: str,
    tag: str,
    text: str,
    mode: InjectionMode,
) -> tuple[str, bool]:
    pattern = re.compile(
        rf'(<{re.escape(tag)}>)([\s\S]*?)(</{re.escape(tag)}>)',
    )
    match = pattern.search(content)
    if not match:
        return content, False

    inner = match.group(2)
    leading = '\n' if inner.startswith('\n') else ''
    trailing = '\n' if inner.endswith('\n') else ''
    core = inner[len(leading):len(inner) - len(trailing) if trailing else None]
    updated_core = _merge_block_content(core, text, mode)
    new_inner = f'{leading}{updated_core}{trailing}'

    start, end = match.span()
    updated = content[:start] + match.group(1) + new_inner + match.group(3) + content[end:]
    return updated, True


def _applied(
    dialect: CursorDialect,
    rule: DialectInstruction,
) -> InjectionResult:
    return InjectionResult(
        'applied',
        dialect,
        rule.target,
        rule.mode,
        f'已向 {rule.target} 注入自定义指令',
    )


def _failed(
    dialect: CursorDialect,
    rule: DialectInstruction,
    message: str,
) -> InjectionResult:
    return InjectionResult(
        'failed',
        dialect,
        rule.target,
        rule.mode,
        message,
    )


def _now() -> str:
    return datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
