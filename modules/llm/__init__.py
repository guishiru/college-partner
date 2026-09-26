"""大模型客户端边界：窄接口 + 厂商适配器。"""

from modules.llm.service import (
    DEFAULT_QWEN_MODEL,
    QWEN_BASE_URL,
    FakeClient,
    LLMClient,
    LLMError,
    LLMNotConfigured,
    LLMResponse,
    NullClient,
    QwenClient,
    client_from_environment,
)

__all__ = [
    "DEFAULT_QWEN_MODEL",
    "QWEN_BASE_URL",
    "FakeClient",
    "LLMClient",
    "LLMError",
    "LLMNotConfigured",
    "LLMResponse",
    "NullClient",
    "QwenClient",
    "client_from_environment",
]
