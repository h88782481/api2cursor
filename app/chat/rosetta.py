"""llm-rosetta 转换门面。"""

from __future__ import annotations

from typing import Any, cast

from llm_rosetta import AnthropicConverter, GoogleGenAIConverter
from llm_rosetta import OpenAIChatConverter, OpenAIResponsesConverter
from llm_rosetta.converters import BaseConverter
from llm_rosetta.types.ir import IRRequest, IRResponse, IRStreamEvent

from ..errors import ApiError
from ..protocol import WireProtocol


class Rosetta:
    def __init__(self):
        self._converters: dict[WireProtocol, BaseConverter] = {
            'chat': OpenAIChatConverter(),
            'responses': OpenAIResponsesConverter(),
            'messages': AnthropicConverter(),
            'gemini': GoogleGenAIConverter(),
        }

    @property
    def chat(self) -> OpenAIChatConverter:
        return cast(OpenAIChatConverter, self._converters['chat'])

    def request_from(self, protocol: WireProtocol, payload: dict[str, Any]) -> IRRequest:
        try:
            return self._converters[protocol].request_from_provider(payload)
        except Exception as exc:
            raise ApiError(f'无法解析 Cursor 请求: {exc}') from exc

    def request_to(
        self,
        protocol: WireProtocol,
        request: IRRequest,
    ) -> tuple[dict[str, Any], list[str]]:
        try:
            kwargs = {'output_format': 'rest'} if protocol == 'gemini' else {}
            return self._converters[protocol].request_to_provider(request, **kwargs)
        except Exception as exc:
            raise ApiError(
                f'无法转换为 {protocol} 请求: {exc}',
                error_type='conversion_error',
                status_code=422,
            ) from exc

    def response_from(self, protocol: WireProtocol, payload: dict[str, Any]) -> IRResponse:
        try:
            return self._converters[protocol].response_from_provider(payload)
        except Exception as exc:
            raise ApiError(
                f'无法解析 {protocol} 响应: {exc}',
                error_type='upstream_response_error',
                status_code=502,
            ) from exc

    def response_to_chat(self, response: IRResponse) -> dict[str, Any]:
        try:
            return self.chat.response_to_provider(response)
        except Exception as exc:
            raise ApiError(
                f'无法生成 Cursor 响应: {exc}',
                error_type='conversion_error',
                status_code=502,
            ) from exc

    def stream_context(self, protocol: WireProtocol):
        return self._converters[protocol].create_stream_context()

    def stream_from(
        self,
        protocol: WireProtocol,
        chunk: dict[str, Any],
        context: Any,
    ) -> list[IRStreamEvent]:
        try:
            return self._converters[protocol].stream_response_from_provider(chunk, context)
        except Exception as exc:
            raise ApiError(
                f'无法解析 {protocol} 流事件: {exc}',
                error_type='upstream_stream_error',
                status_code=502,
            ) from exc

    def stream_to_chat(self, event: IRStreamEvent, context: Any) -> list[dict[str, Any]]:
        try:
            output = self.chat.stream_response_to_provider(event, context)
        except Exception as exc:
            raise ApiError(
                f'无法生成 Cursor 流事件: {exc}',
                error_type='conversion_error',
                status_code=502,
            ) from exc
        items = output if isinstance(output, list) else [output]
        return [item for item in items if item]
