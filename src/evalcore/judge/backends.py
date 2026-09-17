"""Pluggable judge backend interface. The orchestrator/telemetry layer
depends only on this protocol, never on a specific LLM client -- so
swapping local Ollama for a real rate-limited cloud API (OpenAI, Anthropic,
Bedrock, etc.) at scale is a config change, not a rewrite. MockCloudBackend
proves this out end-to-end (concurrency, retries, latency variance) without
needing a paid API key.
"""

from __future__ import annotations

import asyncio
import random
from typing import Protocol

from evalcore.judge.ollama_client import call_ollama


class JudgeBackend(Protocol):
    name: str

    async def call(self, prompt: str, model: str) -> str: ...


class OllamaBackend:
    """Wraps the existing synchronous call_ollama() in a thread so it can
    be awaited alongside other backends. max_concurrency is advisory here
    (local Ollama has its own internal queuing) but is surfaced so the
    orchestrator's semaphore sizing logic is uniform across backends.
    """

    name = "ollama"

    def __init__(self, max_concurrency: int = 4, call_fn=call_ollama):
        self.max_concurrency = max_concurrency
        self._call_fn = call_fn

    async def call(self, prompt: str, model: str) -> str:
        return await asyncio.to_thread(self._call_fn, prompt, model=model)


class MockCloudBackend:
    """Simulates a rate-limited, high-concurrency cloud LLM API: variable
    latency, an injectable transient-failure rate (to exercise retry
    logic), and high nominal concurrency. This is what a real cloud
    backend (OpenAI/Anthropic/Bedrock) would look like from the
    orchestrator's point of view -- implement JudgeBackend and nothing
    else in the pipeline changes.

    Response content is a crude format-matching stub (not a real model),
    purely so downstream parsers have well-formed input to exercise the
    full pipeline end-to-end.
    """

    name = "mock_cloud"

    def __init__(
        self,
        max_concurrency: int = 200,
        mean_latency_s: float = 0.3,
        latency_jitter_s: float = 0.15,
        failure_rate: float = 0.03,
        seed: int | None = None,
    ):
        self.max_concurrency = max_concurrency
        self.mean_latency_s = mean_latency_s
        self.latency_jitter_s = latency_jitter_s
        self.failure_rate = failure_rate
        self._rng = random.Random(seed)

    async def call(self, prompt: str, model: str) -> str:
        latency = max(0.01, self._rng.gauss(self.mean_latency_s, self.latency_jitter_s))
        await asyncio.sleep(latency)
        if self._rng.random() < self.failure_rate:
            raise RuntimeError("mock_cloud: simulated transient API error (e.g. HTTP 429/503)")
        return self._stub_response(prompt)

    def _stub_response(self, prompt: str) -> str:
        if "VERDICT:" in prompt:
            return f"Brief simulated analysis.\nVERDICT: {self._rng.choice(['A', 'B', 'Tie'])}"
        if "Summary A:" in prompt:
            return self._rng.choice(["A", "B", "Tie"])
        return str(self._rng.randint(1, 5))

