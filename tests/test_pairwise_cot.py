"""Unit tests for the chain-of-thought pairwise prompt/parser."""

import pytest

from evalcore.judge.pairwise_cot import (
    build_cot_pairwise_prompt,
    parse_cot_pairwise_verdict,
)


def test_build_prompt_contains_verdict_instruction():
    prompt = build_cot_pairwise_prompt("Article.", "Summary one.", "Summary two.", "coherence")
    assert "VERDICT:" in prompt
    assert "Summary one." in prompt
    assert "Summary two." in prompt


def test_build_prompt_unknown_dimension_raises():
    with pytest.raises(ValueError, match="unknown dimension"):
        build_cot_pairwise_prompt("article", "a", "b", "readability")


def test_parses_verdict_marker_at_end():
    response = "Summary A is more concise. Summary B has more detail.\nVERDICT: B"
    result = parse_cot_pairwise_verdict(response)
    assert result.winner == "B"
    assert result.parse_ok


def test_parses_tie_verdict_marker():
    response = "Both are comparable in quality.\nVERDICT: Tie"
    result = parse_cot_pairwise_verdict(response)
    assert result.winner == "tie"


def test_takes_last_verdict_if_repeated():
    response = "VERDICT: A\n\nActually, reconsidering the evidence...\nVERDICT: B"
    result = parse_cot_pairwise_verdict(response)
    assert result.winner == "B"


def test_ignores_summary_mentions_in_reasoning_uses_marker():
    # reasoning text mentions both A and B extensively, but the marker
    # should be authoritative, not the reasoning-text mention-counting
    # heuristic that the legacy parser relies on
    response = (
        "Summary A discusses the earnings report in detail. Summary B also "
        "covers Summary A's points but adds redundant information.\nVERDICT: A"
    )
    result = parse_cot_pairwise_verdict(response)
    assert result.winner == "A"


def test_falls_back_to_legacy_parser_when_no_marker():
    response = "Summary A is clearly better on this dimension."
    result = parse_cot_pairwise_verdict(response)
    assert result.winner == "A"
    assert "fallback" in result.parse_note


def test_case_insensitive_marker():
    response = "reasoning here\nverdict: b"
    result = parse_cot_pairwise_verdict(response)
    assert result.winner == "B"


def test_unparseable_fallback_fails_gracefully():
    response = "This is genuinely too close to call without more context."
    result = parse_cot_pairwise_verdict(response)
    assert result.winner is None
    assert not result.parse_ok

