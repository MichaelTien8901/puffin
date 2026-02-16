"""LLM provider abstraction for AI-assisted trading."""

import hashlib
import json
import time
from abc import ABC, abstractmethod
from pathlib import Path


class LLMProvider(ABC):
    """Base class for LLM providers."""

    def __init__(self, cache_dir: str | None = None, cache_ttl: int = 3600):
        self.cache_dir = Path(cache_dir) if cache_dir else Path(".cache/llm")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_ttl = cache_ttl

    @abstractmethod
    def analyze(self, text: str, prompt: str) -> dict:
        """Analyze text with a specific prompt.

        Args:
            text: Text to analyze.
            prompt: Instruction for analysis.

        Returns:
            Dict with 'response', 'model', 'usage' keys.
        """

    @abstractmethod
    def generate(self, prompt: str, system: str | None = None) -> str:
        """Generate text from a prompt.

        Args:
            prompt: User prompt.
            system: Optional system prompt.

        Returns:
            Generated text.
        """

    def _cache_key(self, *args) -> str:
        content = json.dumps(args, sort_keys=True)
        return hashlib.md5(content.encode()).hexdigest()

    def _get_cached(self, key: str) -> dict | None:
        cache_file = self.cache_dir / f"{key}.json"
        if cache_file.exists():
            with open(cache_file) as f:
                cached = json.load(f)
            if time.time() - cached.get("timestamp", 0) < self.cache_ttl:
                return cached.get("data")
        return None

    def _set_cached(self, key: str, data: dict):
        cache_file = self.cache_dir / f"{key}.json"
        with open(cache_file, "w") as f:
            json.dump({"timestamp": time.time(), "data": data}, f)
