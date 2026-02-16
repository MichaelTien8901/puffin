"""Concrete LLM provider implementations."""

import os
import json

from puffin.ai.llm_provider import LLMProvider


class ClaudeProvider(LLMProvider):
    """Anthropic Claude LLM provider."""

    def __init__(self, api_key: str | None = None, model: str = "claude-sonnet-4-5-20250929", **kwargs):
        super().__init__(**kwargs)
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        self.model = model
        self._client = None

    def _get_client(self):
        if self._client is None:
            import anthropic
            self._client = anthropic.Anthropic(api_key=self.api_key)
        return self._client

    def analyze(self, text: str, prompt: str) -> dict:
        cache_key = self._cache_key("claude", self.model, text[:500], prompt)
        cached = self._get_cached(cache_key)
        if cached:
            return cached

        client = self._get_client()
        message = client.messages.create(
            model=self.model,
            max_tokens=1024,
            messages=[
                {"role": "user", "content": f"{prompt}\n\nText to analyze:\n{text}"}
            ],
        )

        result = {
            "response": message.content[0].text,
            "model": self.model,
            "usage": {
                "input_tokens": message.usage.input_tokens,
                "output_tokens": message.usage.output_tokens,
            },
        }

        self._set_cached(cache_key, result)
        return result

    def generate(self, prompt: str, system: str | None = None) -> str:
        client = self._get_client()
        kwargs = {
            "model": self.model,
            "max_tokens": 2048,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system:
            kwargs["system"] = system

        message = client.messages.create(**kwargs)
        return message.content[0].text


class OpenAIProvider(LLMProvider):
    """OpenAI GPT LLM provider."""

    def __init__(self, api_key: str | None = None, model: str = "gpt-4o", **kwargs):
        super().__init__(**kwargs)
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self.model = model
        self._client = None

    def _get_client(self):
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI(api_key=self.api_key)
        return self._client

    def analyze(self, text: str, prompt: str) -> dict:
        cache_key = self._cache_key("openai", self.model, text[:500], prompt)
        cached = self._get_cached(cache_key)
        if cached:
            return cached

        client = self._get_client()
        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "user", "content": f"{prompt}\n\nText to analyze:\n{text}"}
            ],
            max_tokens=1024,
        )

        result = {
            "response": response.choices[0].message.content,
            "model": self.model,
            "usage": {
                "input_tokens": response.usage.prompt_tokens,
                "output_tokens": response.usage.completion_tokens,
            },
        }

        self._set_cached(cache_key, result)
        return result

    def generate(self, prompt: str, system: str | None = None) -> str:
        client = self._get_client()
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        response = client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=2048,
        )
        return response.choices[0].message.content
