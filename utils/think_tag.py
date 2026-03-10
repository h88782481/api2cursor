"""<think> 标签提取器

部分上游 API（如 DeepSeek）使用 <think>...</think> 标签包裹思考内容，
而 Cursor 期望 reasoning_content 字段。本模块在流式和非流式响应中
提取 <think> 标签内容并转为 reasoning_content。
"""

import re

_THINK_RE = re.compile(r'<think>(.*?)</think>', re.DOTALL)


def extract_from_text(content):
    """从文本中提取 <think> 标签（非流式）

    返回: (cleaned_content, reasoning_content)
    """
    if not isinstance(content, str) or '<think>' not in content:
        return content, None
    m = _THINK_RE.search(content)
    if not m:
        return content, None
    reasoning = m.group(1).strip()
    cleaned = (content[:m.start()] + content[m.end():]).strip() or None
    return cleaned, reasoning


class ThinkTagExtractor:
    """流式 <think> 标签提取器

    处理跨 chunk 的 <think>...</think> 标签，将标签内的文本
    转为 reasoning_content delta，标签外的文本保持为 content delta。
    """

    def __init__(self):
        """初始化跨 chunk 的 thinking 状态跟踪。"""
        self._in_thinking = False

    def process_chunk(self, chunk):
        """处理一个流式 chunk，返回转换后的 chunk 列表"""
        for choice in (chunk.get('choices') or []):
            delta = choice.get('delta') or {}
            if delta.get('reasoning_content'):
                return [chunk]
            content = delta.get('content')
            if content is None or content == '':
                return [chunk]
            return self._split(chunk, content)
        return [chunk]

    def _split(self, chunk, text):
        """根据 <think> 标签拆分文本为多个 chunk"""
        results = []

        if self._in_thinking:
            end = text.find('</think>')
            if end >= 0:
                self._in_thinking = False
                if text[:end]:
                    results.append(self._make(chunk, reasoning=text[:end]))
                rest = text[end + 8:].lstrip('\n')
                if rest:
                    results.append(self._make(chunk, content=rest))
            else:
                results.append(self._make(chunk, reasoning=text))
        else:
            start = text.find('<think>')
            if start >= 0:
                before = text[:start]
                after = text[start + 7:]
                if before:
                    results.append(self._make(chunk, content=before))
                end = after.find('</think>')
                if end >= 0:
                    if after[:end]:
                        results.append(self._make(chunk, reasoning=after[:end]))
                    rest = after[end + 8:].lstrip('\n')
                    if rest:
                        results.append(self._make(chunk, content=rest))
                else:
                    self._in_thinking = True
                    if after:
                        results.append(self._make(chunk, reasoning=after))
            else:
                results.append(chunk)

        return results or [chunk]

    @staticmethod
    def _make(template, content=None, reasoning=None):
        """根据模板 chunk 构造新的 delta chunk"""
        delta = {}
        if content is not None:
            delta['content'] = content
        if reasoning is not None:
            delta['reasoning_content'] = reasoning
        return {
            'id': template.get('id', ''),
            'object': 'chat.completion.chunk',
            'model': template.get('model', ''),
            'choices': [{'index': 0, 'delta': delta, 'finish_reason': None}],
        }
