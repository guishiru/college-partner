"""大模型客户端：一个窄接口，几个适配器。

## 为什么接口要这么窄

只有一个方法：消息进，文本出。不暴露厂商的流式、函数调用、多模态这些能力——
用得上的时候再说。接口越窄，换厂商越便宜，测试也越好写。

## 三个实现

``QwenClient``  千问，走 DashScope 的 OpenAI 兼容端点。
``FakeClient``  返回预置回复，供测试。整条链路用它就能跑通，不花钱不联网。
``NullClient``  没配置时的占位。调用它会明确报错，**不会静默降级**——
                「模型没配置」和「模型说识别不出」是两件事，混在一起会让
                线上出问题时查不出原因。

## 密钥

只从环境变量读，绝不落盘、绝不进日志、绝不进异常消息。仓库里有 `.env.example`
说明要配哪些变量，`.env` 本身在 `.gitignore` 里。
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Protocol

#: 千问的 OpenAI 兼容端点。换厂商通常只需要改这里和模型名。
QWEN_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DEFAULT_QWEN_MODEL = "qwen-plus"
DEFAULT_TIMEOUT = 30.0
DEFAULT_RETRIES = 2


class LLMError(RuntimeError):
    """模型调用失败。消息里不含密钥。"""


class LLMNotConfigured(LLMError):
    """没有配置模型。与「模型识别不出」区分开。"""


@dataclass(frozen=True)
class LLMResponse:
    text: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


class LLMClient(Protocol):
    def complete(self, *, system: str, user: str, temperature: float = 0.0) -> LLMResponse:
        ...


class NullClient:
    """没配置模型时的占位。"""

    name = "null"

    def complete(self, *, system: str, user: str, temperature: float = 0.0) -> LLMResponse:
        raise LLMNotConfigured(
            "没有配置大模型。设置环境变量 LLM_PROVIDER 与对应的密钥后重试。"
        )


@dataclass
class FakeClient:
    """测试用：按调用顺序返回预置回复。

    真调 API 的测试慢、贵、每次结果还不一样，CI 里跑不了。桩不是将就，是这类
    测试唯一站得住的做法——要验的是「模型乱说的时候系统会不会被污染」，
    那就得能精确控制模型说什么。
    """

    replies: list[str] = field(default_factory=list)
    model: str = "fake"
    calls: list[dict[str, str]] = field(default_factory=list)
    name: str = "fake"

    def complete(self, *, system: str, user: str, temperature: float = 0.0) -> LLMResponse:
        self.calls.append({"system": system, "user": user})
        if not self.replies:
            raise LLMError("FakeClient 没有更多预置回复了。")
        text = self.replies.pop(0)
        if isinstance(text, Exception):
            raise text
        return LLMResponse(text=text, model=self.model,
                           input_tokens=len(system) + len(user), output_tokens=len(text))


@dataclass
class QwenClient:
    """千问，走 DashScope 的 OpenAI 兼容端点。

    用标准库发请求，不引入新依赖——这里只需要一次 POST 和一次 JSON 解析，
    为此拉一个 SDK 进来不划算。
    """

    api_key: str
    model: str = DEFAULT_QWEN_MODEL
    base_url: str = QWEN_BASE_URL
    timeout: float = DEFAULT_TIMEOUT
    retries: int = DEFAULT_RETRIES
    name: str = "qwen"

    def complete(self, *, system: str, user: str, temperature: float = 0.0) -> LLMResponse:
        payload = json.dumps(
            {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "temperature": temperature,
            },
            ensure_ascii=False,
        ).encode("utf-8")

        request = urllib.request.Request(
            f"{self.base_url.rstrip('/')}/chat/completions",
            data=payload,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )

        last_error: Exception | None = None
        for attempt in range(self.retries + 1):
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    body = json.loads(response.read().decode("utf-8"))
                break
            except urllib.error.HTTPError as exc:
                # 4xx 重试没有意义，直接抛；5xx 和超时值得再试一次。
                if exc.code < 500:
                    raise LLMError(f"模型调用被拒绝（HTTP {exc.code}）。") from None
                last_error = exc
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                last_error = exc
            if attempt < self.retries:
                time.sleep(0.5 * (attempt + 1))
        else:
            raise LLMError(f"模型调用失败：{type(last_error).__name__}") from None

        try:
            text = body["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError):
            raise LLMError("模型返回的结构不符合预期。") from None

        usage = body.get("usage") or {}
        return LLMResponse(
            text=text,
            model=body.get("model", self.model),
            input_tokens=int(usage.get("prompt_tokens", 0)),
            output_tokens=int(usage.get("completion_tokens", 0)),
        )


def client_from_environment() -> LLMClient:
    """按环境变量装配客户端。密钥只在这里读一次。"""

    provider = (os.environ.get("LLM_PROVIDER") or "").strip().lower()
    if not provider or provider == "none":
        return NullClient()
    if provider == "fake":
        # 端到端测试里服务跑在另一个进程，预置回复通过文件传入。
        path = (os.environ.get("E2E_FAKE_REPLIES") or "").strip()
        replies: list[str] = []
        if path and os.path.isfile(path):
            with open(path, encoding="utf-8") as handle:
                replies = json.load(handle)
        return FakeClient(replies=replies)
    if provider in {"qwen", "dashscope"}:
        api_key = (os.environ.get("DASHSCOPE_API_KEY") or "").strip()
        if not api_key:
            raise LLMNotConfigured(
                "LLM_PROVIDER 设为 qwen，但没有设置 DASHSCOPE_API_KEY。"
            )
        return QwenClient(
            api_key=api_key,
            model=(os.environ.get("LLM_MODEL") or DEFAULT_QWEN_MODEL).strip(),
            base_url=(os.environ.get("LLM_BASE_URL") or QWEN_BASE_URL).strip(),
            timeout=float(os.environ.get("LLM_TIMEOUT") or DEFAULT_TIMEOUT),
        )
    raise LLMNotConfigured(f"不支持的 LLM_PROVIDER：{provider}")
