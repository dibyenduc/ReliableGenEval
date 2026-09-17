"""Unit tests for pairwise judge prompt building and verdict parsing."""

import pytest

from evalcore.judge.pairwise import build_pairwise_prompt, parse_pairwise_verdict


def test_bare_a_parses():
    result = parse_pairwise_verdict("A")
    assert result.winner == "A"
    assert result.parse_ok


def test_bare_b_with_punctuation_parses():
    result = parse_pairwise_verdict("B.")
    assert result.winner == "B"


def test_bare_tie_parses_case_insensitive():
    result = parse_pairwise_verdict("tie")
    assert result.winner == "tie"


def test_leading_letter_with_explanation_parses():
    result = parse_pairwise_verdict("A, because it is more concise and accurate.")
    assert result.winner == "A"


def test_tie_keyword_anywhere_parses():
    result = parse_pairwise_verdict("Both summaries are equally good, so it's a tie.")
    assert result.winner == "tie"


def test_only_summary_a_mentioned_parses():
    result = parse_pairwise_verdict("Summary A is better because it captures the key facts.")
    assert result.winner == "A"


def test_only_summary_b_mentioned_parses():
    result = parse_pairwise_verdict("Summary B is more coherent overall.")
    assert result.winner == "B"


def test_unparseable_response_fails_gracefully():
    result = parse_pairwise_verdict("This is a difficult comparison to make.")
    assert result.winner is None
    assert not result.parse_ok


def test_build_prompt_contains_both_summaries():
    prompt = build_pairwise_prompt("Article.", "Summary one.", "Summary two.", "coherence")
    assert "Summary one." in prompt
    assert "Summary two." in prompt


def test_build_prompt_unknown_dimension_raises():
    with pytest.raises(ValueError, match="unknown dimension"):
        build_pairwise_prompt("article", "a", "b", "readability")
