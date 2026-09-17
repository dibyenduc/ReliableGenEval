"""Unit tests for the pluggable judge backend interface. Async calls are
run via asyncio.run() directly (no extra pytest-asyncio plugin needed,
matching this repo's existing plugin set)."""

import asyncio

import pytest

from evalcore.judge.backends import MockCloudBackend, OllamaBackend


def test_ollama_backend_delegates_to_call_fn():
    calls = []

    def fake_call_ollama(prompt, model):
        calls.append((prompt, model))
        return "3"

    backend = OllamaBackend(call_fn=fake_call_ollama)
    result = asyncio.run(backend.call("some prompt", "llama3.1:latest"))
    assert result == "3"
    assert calls == [("some prompt", "llama3.1:latest")]


def test_ollama_backend_has_name():
    assert OllamaBackend().name == "ollama"


def test_mock_cloud_backend_deterministic_given_seed():
    b1 = MockCloudBackend(seed=42, failure_rate=0.0, mean_latency_s=0.01, latency_jitter_s=0.001)
    b2 = MockCloudBackend(seed=42, failure_rate=0.0, mean_latency_s=0.01, latency_jitter_s=0.001)
    r1 = asyncio.run(b1.call("Rate this 1-5", "fake-model"))
    r2 = asyncio.run(b2.call("Rate this 1-5", "fake-model"))
    assert r1 == r2


def test_mock_cloud_backend_pointwise_style_prompt_returns_digit():
    backend = MockCloudBackend(seed=1, failure_rate=0.0, mean_latency_s=0.01, latency_jitter_s=0.001)
    result = asyncio.run(backend.call("Rate this on a scale of 1 to 5.", "fake-model"))
    assert result.strip() in {"1", "2", "3", "4", "5"}


def test_mock_cloud_backend_cot_pairwise_prompt_contains_verdict_marker():
    backend = MockCloudBackend(seed=1, failure_rate=0.0, mean_latency_s=0.01, latency_jitter_s=0.001)
    prompt = "Summary A:\n...\nSummary B:\n...\nVERDICT: A"
    result = asyncio.run(backend.call(prompt, "fake-model"))
    assert "VERDICT:" in result


def test_mock_cloud_backend_bare_pairwise_prompt_returns_letter_or_tie():
    backend = MockCloudBackend(seed=1, failure_rate=0.0, mean_latency_s=0.01, latency_jitter_s=0.001)
    prompt = "Summary A:\nfoo\nSummary B:\nbar\nWhich is better?"
    result = asyncio.run(backend.call(prompt, "fake-model"))
    assert result in {"A", "B", "Tie"}


def test_mock_cloud_backend_always_fails_when_failure_rate_is_one():
    backend = MockCloudBackend(seed=1, failure_rate=1.0, mean_latency_s=0.01, latency_jitter_s=0.001)
    with pytest.raises(RuntimeError, match="simulated transient"):
        asyncio.run(backend.call("prompt", "fake-model"))


def test_mock_cloud_backend_never_fails_when_failure_rate_is_zero():
    backend = MockCloudBackend(seed=1, failure_rate=0.0, mean_latency_s=0.01, latency_jitter_s=0.001)
    for _ in range(20):
        asyncio.run(backend.call("prompt", "fake-model"))  # should not raise


def test_mock_cloud_backend_has_name():
    assert MockCloudBackend().name == "mock_cloud"

