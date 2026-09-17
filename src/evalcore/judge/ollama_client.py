"""Thin client for calling a local Ollama model. Network-dependent and
non-deterministic -- kept separate from parsing logic so the parser can be
unit-tested without a live server. Not exercised by the default (offline)
pytest run; see tests/test_ollama_client.py for the manually-run integration test.
"""

from __future__ import annotations

import requests

OLLAMA_BASE_URL = "http://localhost:11434"
DEFAULT_TIMEOUT_SECONDS = 60


def call_ollama(prompt: str, model: str = "llama3.1:latest", temperature: float = 0.0) -> str:
    """Call a local Ollama model with a prompt, return the raw text response.

    temperature=0.0 for maximum determinism -- judge scoring should be as
    reproducible as possible, not creative.
    """
    response = requests.post(
        f"{OLLAMA_BASE_URL}/api/generate",
        json={"model": model, "prompt": prompt, "stream": False, "options": {"temperature": temperature}},
        timeout=DEFAULT_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    return response.json()["response"]
