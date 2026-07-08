"""协议编解码器注册表"""

from .anthropic import AnthropicCodec
from .base import Codec
from .chat_completions import ChatCompletionsCodec
from .gemini import GeminiCodec
from .responses_api import ResponsesCodec

CODECS: dict[str, Codec] = {
    'chat': ChatCompletionsCodec(),
    'responses': ResponsesCodec(),
    'messages': AnthropicCodec(),
    'gemini': GeminiCodec(),
}


def get_codec(fmt: str) -> Codec:
    """按格式名获取编解码器。"""
    return CODECS[fmt]
