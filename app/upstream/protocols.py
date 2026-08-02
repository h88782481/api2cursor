"""四种上游协议的 wire 规则。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from urllib.parse import quote

from ..protocol import WireProtocol


def _base(url: str) -> str:
    value = url.rstrip('/')
    for suffix in ('/v1beta', '/v1'):
        if value.endswith(suffix):
            return value[:-len(suffix)]
    return value


@dataclass(frozen=True, slots=True)
class WireSpec:
    protocol: WireProtocol

    def url(self, base_url: str, model: str, stream: bool) -> str:
        base = _base(base_url)
        if self.protocol == 'chat':
            return f'{base}/v1/chat/completions'
        if self.protocol == 'responses':
            return f'{base}/v1/responses'
        if self.protocol == 'messages':
            return f'{base}/v1/messages'
        method = 'streamGenerateContent?alt=sse' if stream else 'generateContent'
        return f'{base}/v1beta/models/{quote(model, safe="")}:{method}'

    def headers(self, api_key: str) -> dict[str, str]:
        headers = {'Content-Type': 'application/json'}
        if self.protocol == 'messages':
            headers['anthropic-version'] = '2023-06-01'
            if api_key.startswith('sk-'):
                headers['x-api-key'] = api_key
            elif api_key:
                headers['Authorization'] = f'Bearer {api_key}'
        elif self.protocol == 'gemini' and api_key.startswith('AIza'):
            headers['x-goog-api-key'] = api_key
        elif api_key:
            headers['Authorization'] = f'Bearer {api_key}'
        return headers

    def prepare_body(self, body: dict[str, Any]) -> dict[str, Any]:
        if self.protocol != 'gemini':
            return body

        body.pop('model', None)
        for source, target in {
            'system_instruction': 'systemInstruction',
            'tool_config': 'toolConfig',
            'response_mime_type': 'responseMimeType',
            'response_schema': 'responseSchema',
            'thinking_config': 'thinkingConfig',
        }.items():
            if source in body:
                body[target] = body.pop(source)
        thinking = body.get('thinkingConfig')
        if isinstance(thinking, dict):
            if 'thinking_level' in thinking:
                thinking['thinkingLevel'] = thinking.pop('thinking_level')
            if 'thinking_budget' in thinking:
                thinking['thinkingBudget'] = thinking.pop('thinking_budget')
        return body


SPECS = {
    protocol: WireSpec(protocol)
    for protocol in ('chat', 'responses', 'messages', 'gemini')
}


def spec(protocol: WireProtocol) -> WireSpec:
    return SPECS[protocol]
