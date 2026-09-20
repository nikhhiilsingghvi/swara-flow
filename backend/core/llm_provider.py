from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Optional, Protocol

from core.config import LLMConfig, get_config

@dataclass(frozen=True)
class LLMResponse:
    text: str
    provider: str
    available: bool
    error: Optional[str] = None

class LLMProvider(Protocol):
    def generate(self, prompt: str, system: Optional[str] = None) -> LLMResponse: ...
    def is_available(self) -> bool: ...

def _post_json(url: str, payload: dict, headers: Optional[dict] = None, timeout: float = 15.0) -> dict:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, method="POST")
    req.add_header("Content-Type", "application/json")
    for k, v in (headers or {}).items():
        req.add_header(k, v)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))

class NoOpLLMProvider:
    def generate(self, prompt: str, system: Optional[str] = None) -> LLMResponse:
        return LLMResponse(text="", provider="none", available=False, error="No LLM provider configured.")

    def is_available(self) -> bool:
        return False

class LMStudioProvider:

    def __init__(self, config: Optional[LLMConfig] = None):
        self.config = config or get_config().llm

    def is_available(self) -> bool:
        try:
            req = urllib.request.Request(f"{self.config.lmstudio_base_url}/models")
            with urllib.request.urlopen(req, timeout=2.0):
                return True
        except Exception:
            return False

    def generate(self, prompt: str, system: Optional[str] = None) -> LLMResponse:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        try:
            result = _post_json(
                f"{self.config.lmstudio_base_url}/chat/completions",
                {"messages": messages, "max_tokens": 500, "temperature": 0.7},
                timeout=self.config.request_timeout_s,
            )
            text = result["choices"][0]["message"]["content"]
            return LLMResponse(text=text, provider="lmstudio", available=True)
        except Exception as e:
            return LLMResponse(text="", provider="lmstudio", available=False, error=str(e))

class OllamaProvider:
    def __init__(self, config: Optional[LLMConfig] = None):
        self.config = config or get_config().llm

    def is_available(self) -> bool:
        try:
            req = urllib.request.Request(f"{self.config.ollama_base_url}/api/tags")
            with urllib.request.urlopen(req, timeout=2.0):
                return True
        except Exception:
            return False

    def generate(self, prompt: str, system: Optional[str] = None) -> LLMResponse:
        full_prompt = f"{system}\n\n{prompt}" if system else prompt
        try:
            result = _post_json(
                f"{self.config.ollama_base_url}/api/generate",
                {"model": self.config.ollama_model, "prompt": full_prompt, "stream": False},
                timeout=self.config.request_timeout_s,
            )
            return LLMResponse(text=result.get("response", ""), provider="ollama", available=True)
        except Exception as e:
            return LLMResponse(text="", provider="ollama", available=False, error=str(e))

class OpenAICompatibleProvider:
    def __init__(self, config: Optional[LLMConfig] = None):
        self.config = config or get_config().llm

    def is_available(self) -> bool:
        return bool(self.config.openai_compat_base_url and self.config.openai_compat_api_key)

    def generate(self, prompt: str, system: Optional[str] = None) -> LLMResponse:
        if not self.is_available():
            return LLMResponse(text="", provider="openai_compatible", available=False,
                                error="OpenAI-compatible provider not configured.")
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        try:
            result = _post_json(
                f"{self.config.openai_compat_base_url}/chat/completions",
                {"model": self.config.openai_compat_model, "messages": messages, "max_tokens": 500},
                headers={"Authorization": f"Bearer {self.config.openai_compat_api_key}"},
                timeout=self.config.request_timeout_s,
            )
            text = result["choices"][0]["message"]["content"]
            return LLMResponse(text=text, provider="openai_compatible", available=True)
        except Exception as e:
            return LLMResponse(text="", provider="openai_compatible", available=False, error=str(e))

def build_llm_provider(config: Optional[LLMConfig] = None) -> LLMProvider:
    cfg = config or get_config().llm
    providers = {
        "lmstudio": LMStudioProvider,
        "ollama": OllamaProvider,
        "openai_compatible": OpenAICompatibleProvider,
        "none": NoOpLLMProvider,
    }
    provider_cls = providers.get(cfg.provider.lower(), NoOpLLMProvider)
    return provider_cls(cfg) if provider_cls is not NoOpLLMProvider else NoOpLLMProvider()
