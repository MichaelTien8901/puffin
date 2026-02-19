---
layout: default
title: "LLM Providers"
parent: "Part 22: AI-Assisted Trading"
nav_order: 1
---

# LLM Providers

## Overview

Puffin uses an abstract `LLMProvider` interface so you can swap between Claude, OpenAI, or any custom provider without changing your trading code. Every provider inherits from the same base class, shares a unified caching layer, and exposes two core methods: `analyze()` for structured text analysis and `generate()` for free-form text generation.

The `puffin.ai.llm_provider` module defines the contract. The `puffin.ai.providers` module supplies concrete implementations for Anthropic Claude and OpenAI GPT.

## The LLMProvider Abstract Base Class

```python
from puffin.ai.llm_provider import LLMProvider

# The abstract interface that all providers implement
class LLMProvider(ABC):
    def __init__(self, cache_dir: str | None = None, cache_ttl: int = 3600):
        """Initialize with optional cache directory and TTL in seconds."""

    @abstractmethod
    def analyze(self, text: str, prompt: str) -> dict:
        """Analyze text with a specific prompt.

        Returns:
            Dict with 'response', 'model', 'usage' keys.
        """

    @abstractmethod
    def generate(self, prompt: str, system: str | None = None) -> str:
        """Generate text from a prompt."""
```

The base class provides three caching helpers used by all subclasses:

- `_cache_key(*args)` -- builds an MD5 hash from the provider name, model, text, and prompt
- `_get_cached(key)` -- returns cached data if the entry exists and has not expired
- `_set_cached(key, data)` -- writes a timestamped JSON cache entry to disk

{: .note }
The cache key is based on the provider, model, text (first 500 characters), and prompt. Two identical requests within the TTL window will return the cached response without making an API call.

## Using ClaudeProvider

Claude excels at nuanced financial analysis and structured JSON output. The `ClaudeProvider` wraps the Anthropic Python SDK with lazy client initialization and automatic response caching.

```python
from puffin.ai.providers import ClaudeProvider

# Initialize with API key (or use ANTHROPIC_API_KEY env var)
provider = ClaudeProvider(
    api_key="your-api-key",
    model="claude-sonnet-4-5-20250929",  # Date-versioned for reproducibility; use "claude-sonnet-4-5" for latest
    cache_dir=".cache/llm",             # Response caching directory
    cache_ttl=3600,                      # 1 hour cache TTL
)

# Analyze text -- returns structured dict
result = provider.analyze(
    text="Apple reported Q4 earnings beating estimates by 15%...",
    prompt="Extract sentiment and key financial metrics from this earnings report."
)

print(result["response"])
print(f"Model: {result['model']}")
print(f"Tokens: {result['usage']['input_tokens']} in, {result['usage']['output_tokens']} out")
```

Under the hood, `ClaudeProvider` lazily imports the `anthropic` SDK the first time `_get_client()` is called. This means you only need the Anthropic package installed if you actually use this provider.

```python
# The lazy client pattern inside ClaudeProvider
def _get_client(self):
    if self._client is None:
        import anthropic
        self._client = anthropic.Anthropic(api_key=self.api_key)
    return self._client
```

## Using OpenAIProvider

OpenAI's GPT models are supported through the same interface:

```python
from puffin.ai.providers import OpenAIProvider

provider = OpenAIProvider(
    api_key="your-api-key",
    model="gpt-4o",  # Default model
)

result = provider.generate(
    prompt="Summarize the key technical levels for SPY right now.",
    system="You are a technical analysis expert."
)
print(result)
```

The `OpenAIProvider` maps the Puffin `usage` dict keys to match the standard format:

```python
# OpenAI SDK uses prompt_tokens/completion_tokens
# Puffin normalizes to input_tokens/output_tokens
result = {
    "response": response.choices[0].message.content,
    "model": self.model,
    "usage": {
        "input_tokens": response.usage.prompt_tokens,
        "output_tokens": response.usage.completion_tokens,
    },
}
```

{: .tip }
Both providers normalize their usage dictionaries to `input_tokens` and `output_tokens`, so downstream code does not need to know which provider is active.

## Cost Management with Caching

LLM API calls can get expensive quickly, especially when analyzing many articles. Every provider inherits built-in disk caching from the `LLMProvider` base class.

```python
from puffin.ai.providers import ClaudeProvider

# Cache responses for 1 hour (default)
provider = ClaudeProvider(cache_ttl=3600)

# Longer cache for historical analysis that won't change
provider = ClaudeProvider(cache_ttl=86400)  # 24 hours
```

The cache stores JSON files under `cache_dir` (defaulting to `.cache/llm/`). Each file contains a timestamp and the response data:

```json
{
    "timestamp": 1706000000.0,
    "data": {
        "response": "...",
        "model": "claude-sonnet-4-5-20250929",
        "usage": {"input_tokens": 120, "output_tokens": 85}
    }
}
```

## Rate Limiting

Avoid hitting API rate limits by batching requests with delays:

```python
import time
from puffin.ai.sentiment import analyze_sentiment

def analyze_articles_batch(provider, articles, batch_size=10, delay=1.0):
    """Process articles in batches with delays between batches."""
    results = []
    for i in range(0, len(articles), batch_size):
        batch = articles[i:i + batch_size]
        for article in batch:
            results.append(analyze_sentiment(provider, article["text"]))
        time.sleep(delay)  # Delay between batches
    return results
```

{: .warning }
Anthropic and OpenAI both enforce rate limits on tokens per minute and requests per minute. If you exceed them, the SDK raises a rate-limit error. Use caching and batching to stay within your tier's limits.

## Fallback Providers

For production systems, switch to a backup provider if the primary fails:

```python
from puffin.ai.providers import ClaudeProvider, OpenAIProvider

class RobustProvider:
    """Wraps a primary and fallback LLM provider."""

    def __init__(self, primary, fallback):
        self.primary = primary
        self.fallback = fallback

    def analyze(self, text, prompt):
        try:
            return self.primary.analyze(text, prompt)
        except Exception as e:
            print(f"Primary failed: {e}. Using fallback.")
            return self.fallback.analyze(text, prompt)

    def generate(self, prompt, system=None):
        try:
            return self.primary.generate(prompt, system)
        except Exception as e:
            print(f"Primary failed: {e}. Using fallback.")
            return self.fallback.generate(prompt, system)

provider = RobustProvider(
    primary=ClaudeProvider(),
    fallback=OpenAIProvider(),
)
```

## Async Performance

LLM calls typically take 1-5 seconds per request. For batch analysis, use async to process multiple articles in parallel:

```python
import asyncio
from anthropic import AsyncAnthropic
from puffin.ai.providers import ClaudeProvider

class AsyncClaudeProvider(ClaudeProvider):
    """Extends ClaudeProvider with async analysis."""

    async def analyze_async(self, text, prompt):
        """Async version of analyze for parallel processing."""
        # Check cache first
        cache_key = self._cache_key("claude", self.model, text[:500], prompt)
        cached = self._get_cached(cache_key)
        if cached:
            return cached

        client = AsyncAnthropic(api_key=self.api_key)
        message = await client.messages.create(
            model=self.model,
            max_tokens=1024,
            messages=[{"role": "user", "content": f"{prompt}\n\nText:\n{text}"}],
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


# Process multiple articles in parallel
async def analyze_parallel(provider, articles, prompt):
    tasks = [provider.analyze_async(a["text"], prompt) for a in articles]
    return await asyncio.gather(*tasks)


# Run the parallel analysis
provider = AsyncClaudeProvider()
results = asyncio.run(analyze_parallel(provider, articles, "Analyze sentiment."))
```

{: .tip }
Async analysis is most useful when processing 10+ articles. For smaller batches, the synchronous API with caching is simpler and equally fast on cache hits.

## Token Usage Monitoring

Track your total token consumption across requests to stay within budget:

```python
total_tokens = 0

def track_usage(result):
    """Accumulate token counts from provider results."""
    global total_tokens
    usage = result.get("usage", {})
    tokens = usage.get("input_tokens", 0) + usage.get("output_tokens", 0)
    total_tokens += tokens
    print(f"Tokens used: {tokens} (total: {total_tokens})")

result = provider.analyze(text, prompt)
track_usage(result)
```

## Output Validation

Always validate LLM outputs before passing them to trading logic:

```python
def validate_sentiment(sentiment):
    """Ensure sentiment output has required fields and valid ranges."""
    required_keys = {"sentiment", "score", "confidence", "reasoning"}
    if not required_keys.issubset(sentiment.keys()):
        raise ValueError("Missing required sentiment fields")

    if not -1.0 <= sentiment["score"] <= 1.0:
        raise ValueError("Score out of range [-1.0, 1.0]")

    if not 0.0 <= sentiment["confidence"] <= 1.0:
        raise ValueError("Confidence out of range [0.0, 1.0]")

    return sentiment
```

{: .warning }
LLMs can return malformed JSON, hallucinated tickers, or scores outside expected ranges. Always validate before acting on the output.

## Source Code

- Provider base class: [`puffin/ai/llm_provider.py`](https://github.com/MichaelTien8901/puffin/tree/main/puffin/ai/llm_provider.py)
- Claude and OpenAI implementations: [`puffin/ai/providers.py`](https://github.com/MichaelTien8901/puffin/tree/main/puffin/ai/providers.py)

## Next Steps

Now that you have providers configured, move on to [Sentiment & Signals](02-sentiment-signals) to extract actionable trading signals from financial news.
