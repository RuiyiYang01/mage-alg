"""LLM backbone utilities shared across MAGE agents.

Supports both local models (via transformers) and API models (OpenAI-compatible).
"""

from __future__ import annotations

import json
import logging
import os
from abc import ABC, abstractmethod
from typing import Any

logger = logging.getLogger(__name__)


class LLMBackend(ABC):
    """Abstract LLM backend."""

    @abstractmethod
    def generate(self, prompt: str, system_prompt: str = "", **kwargs: Any) -> str:
        ...

    @abstractmethod
    def generate_json(self, prompt: str, system_prompt: str = "", **kwargs: Any) -> dict | list:
        """Generate and parse JSON output."""
        ...


class APIBackend(LLMBackend):
    """OpenAI-compatible API backend (works with OpenAI, vLLM, Ollama, etc.)."""

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        base_url: str | None = None,
        api_key: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

        try:
            from openai import OpenAI
            self.client = OpenAI(
                base_url=base_url or os.environ.get("OPENAI_BASE_URL"),
                api_key=api_key or os.environ.get("OPENAI_API_KEY", "no-key"),
            )
        except ImportError:
            raise ImportError("openai package required for APIBackend: pip install openai")

    def generate(self, prompt: str, system_prompt: str = "", **kwargs: Any) -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        # Disable Qwen3's <think>...</think> reasoning blocks via vLLM's
        # chat_template_kwargs. Sequential agents (Crafter, WebShop) need
        # a single concrete action per turn — thinking blocks burn the
        # max_tokens budget without ever emitting an action, then the
        # parser falls through to a no-op fallback. The user can override
        # via kwargs["enable_thinking"]=True if needed.
        extra: dict[str, Any] = {}
        if "Qwen3" in self.model or "qwen3" in self.model.lower():
            extra["chat_template_kwargs"] = {
                "enable_thinking": kwargs.get("enable_thinking", False),
            }

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=kwargs.get("temperature", self.temperature),
            max_tokens=kwargs.get("max_tokens", self.max_tokens),
            extra_body=extra or None,
        )
        return response.choices[0].message.content or ""

    def generate_json(self, prompt: str, system_prompt: str = "", **kwargs: Any) -> dict | list:
        text = self.generate(prompt, system_prompt, **kwargs)
        return _parse_json_from_text(text)


class LocalBackend(LLMBackend):
    """Local transformers model backend with optional LoRA.

    Supports Qwen3 no-think mode: if model name contains 'Qwen3',
    appends /no_think suffix to disable thinking for faster generation.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-7B-Instruct",
        device: str = "auto",
        load_in_4bit: bool = True,
        lora_path: str | None = None,
        temperature: float = 0.7,
        max_new_tokens: int = 2048,
        no_think: bool = True,
    ):
        self.model_name = model_name
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self._model = None
        self._tokenizer = None
        self._device = device
        self._load_in_4bit = load_in_4bit
        self._lora_path = lora_path
        self._is_qwen3 = "Qwen3" in model_name or "qwen3" in model_name
        self._no_think = no_think and self._is_qwen3

    def _load_model(self) -> None:
        """Lazy load model on first use."""
        if self._model is not None:
            return

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        logger.info(f"Loading model: {self.model_name}")
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        kwargs: dict[str, Any] = {"device_map": self._device}
        if self._load_in_4bit:
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
            )

        self._model = AutoModelForCausalLM.from_pretrained(self.model_name, **kwargs)

        if self._lora_path:
            from peft import PeftModel
            logger.info(f"Loading LoRA adapter: {self._lora_path}")
            self._model = PeftModel.from_pretrained(self._model, self._lora_path)

        logger.info(f"Model loaded: {self.model_name}")

    def generate(self, prompt: str, system_prompt: str = "", **kwargs: Any) -> str:
        self._load_model()

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        # Qwen3 no-think: append /no_think to user content
        user_content = prompt
        if self._no_think:
            user_content = prompt + " /no_think"

        messages.append({"role": "user", "content": user_content})

        text = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )

        # Determine device for inputs
        if hasattr(self._model, "device"):
            target_device = self._model.device
        elif hasattr(self._model, "hf_device_map"):
            # Multi-device model — use first device
            first_device = next(iter(self._model.hf_device_map.values()))
            target_device = first_device
        else:
            target_device = "cpu"

        inputs = self._tokenizer(text, return_tensors="pt").to(target_device)

        import torch
        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=kwargs.get("max_new_tokens", self.max_new_tokens),
                temperature=kwargs.get("temperature", self.temperature),
                do_sample=True,
            )

        generated = outputs[0][inputs["input_ids"].shape[1]:]
        result = self._tokenizer.decode(generated, skip_special_tokens=True)

        # Qwen3: strip any <think>...</think> tags that may leak through
        if self._is_qwen3 and "<think>" in result:
            import re
            result = re.sub(r"<think>.*?</think>", "", result, flags=re.DOTALL).strip()

        return result

    def generate_json(self, prompt: str, system_prompt: str = "", **kwargs: Any) -> dict | list:
        text = self.generate(prompt, system_prompt, **kwargs)
        return _parse_json_from_text(text)


class AnthropicBackend(LLMBackend):
    """Anthropic Claude API backend for strong LLM calls (Navigator, Critic, Curator)."""

    def __init__(
        self,
        model: str = "claude-sonnet-4-6-20250514",
        api_key: str | None = None,
        api_key_file: str | None = None,
        temperature: float = 0.3,
        max_tokens: int = 4096,
    ):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

        # Load API key from file if provided
        key = api_key
        if not key and api_key_file:
            from pathlib import Path
            key_path = Path(api_key_file)
            if key_path.exists():
                key = key_path.read_text().strip()

        if not key:
            key = os.environ.get("ANTHROPIC_API_KEY")

        if not key:
            raise ValueError("Anthropic API key required: set api_key, api_key_file, or ANTHROPIC_API_KEY env var")

        try:
            import anthropic
            self.client = anthropic.Anthropic(api_key=key)
        except ImportError:
            raise ImportError("anthropic package required: pip install anthropic")

    def generate(self, prompt: str, system_prompt: str = "", **kwargs: Any) -> str:
        # 2026-05-05: Opus 4.7 deprecated the `temperature` parameter on the
        # Messages API; including it returns HTTP 400. Older Sonnet/Haiku
        # models still accept it. Drop temperature for Opus 4.7+ models so
        # principle extraction, failure analysis, and judge calls don't
        # silently fail (which was the root cause of the long-standing
        # plateau on Opus-teacher cells: every `extract_principles` call
        # returned 400 → exception swallowed → 0 principles in KG).
        params = {
            "model": self.model,
            "max_tokens": kwargs.get("max_tokens", self.max_tokens),
            "system": system_prompt if system_prompt else "You are a helpful assistant.",
            "messages": [{"role": "user", "content": prompt}],
        }
        if not self.model.startswith("claude-opus-4-7"):
            params["temperature"] = kwargs.get("temperature", self.temperature)
        message = self.client.messages.create(**params)
        return message.content[0].text

    def generate_json(self, prompt: str, system_prompt: str = "", **kwargs: Any) -> dict | list:
        text = self.generate(prompt, system_prompt, **kwargs)
        return _parse_json_from_text(text)


class MockBackend(LLMBackend):
    """Mock backend for testing — returns predefined or echo responses."""

    def __init__(self, responses: dict[str, str] | None = None):
        self.responses = responses or {}
        self.call_log: list[dict[str, str]] = []

    def generate(self, prompt: str, system_prompt: str = "", **kwargs: Any) -> str:
        self.call_log.append({"prompt": prompt, "system_prompt": system_prompt})
        # Check if any key in responses matches a substring of the prompt
        for key, response in self.responses.items():
            if key in prompt:
                return response
        return f"Mock response to: {prompt[:100]}"

    def generate_json(self, prompt: str, system_prompt: str = "", **kwargs: Any) -> dict | list:
        text = self.generate(prompt, system_prompt, **kwargs)
        try:
            return _parse_json_from_text(text)
        except (json.JSONDecodeError, ValueError):
            return {}


def _parse_json_from_text(text: str) -> dict | list:
    """Extract JSON from LLM output that may contain markdown fences or extra text."""
    text = text.strip()

    # Try direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try extracting from ```json ... ``` blocks
    if "```json" in text:
        start = text.index("```json") + 7
        end = text.index("```", start)
        return json.loads(text[start:end].strip())

    if "```" in text:
        start = text.index("```") + 3
        end = text.index("```", start)
        candidate = text[start:end].strip()
        if candidate.startswith(("{", "[")):
            return json.loads(candidate)

    # Try finding first { or [ and matching bracket
    for start_char, end_char in [("{", "}"), ("[", "]")]:
        if start_char in text:
            start = text.index(start_char)
            depth = 0
            for i in range(start, len(text)):
                if text[i] == start_char:
                    depth += 1
                elif text[i] == end_char:
                    depth -= 1
                if depth == 0:
                    return json.loads(text[start : i + 1])

    raise ValueError(f"Could not parse JSON from: {text[:200]}")


def create_backend(config: dict[str, Any]) -> LLMBackend:
    """Factory to create an LLM backend from config."""
    backend_type = config.get("type", "mock")

    if backend_type == "api":
        return APIBackend(
            model=config.get("model", "gpt-4o-mini"),
            base_url=config.get("base_url"),
            api_key=config.get("api_key"),
            temperature=config.get("temperature", 0.7),
            max_tokens=config.get("max_tokens", 2048),
        )
    elif backend_type == "local":
        return LocalBackend(
            model_name=config.get("model", "Qwen/Qwen2.5-7B-Instruct"),
            device=config.get("device", "auto"),
            load_in_4bit=config.get("load_in_4bit", True),
            lora_path=config.get("lora_path"),
            temperature=config.get("temperature", 0.7),
            max_new_tokens=config.get("max_new_tokens", 2048),
            no_think=config.get("no_think", True),
        )
    elif backend_type == "anthropic":
        return AnthropicBackend(
            model=config.get("model", "claude-sonnet-4-6-20250514"),
            api_key=config.get("api_key"),
            api_key_file=config.get("api_key_file"),
            temperature=config.get("temperature", 0.3),
            max_tokens=config.get("max_tokens", 4096),
        )
    elif backend_type == "mock":
        return MockBackend(config.get("responses", {}))
    else:
        raise ValueError(f"Unknown backend type: {backend_type}")
