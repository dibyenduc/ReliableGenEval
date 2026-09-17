"""Unit tests for pointwise judge prompt building and score parsing.
No network calls -- all inputs are hand-crafted strings simulating the
range of outputs a real LLM judge might produce."""

import pytest

from evalcore.judge.pointwise import build_pointwise_prompt, parse_pointwise_score


def test_bare_integer_parses_correctly():
    result = parse_pointwise_score("2")
    assert result.score == 2
    assert result.parse_ok


def test_integer_with_whitespace_parses():
    result = parse_pointwise_score("  4  \n")
    assert result.score == 4
    assert result.parse_ok


def test_number_embedded_in_sentence_parses():
    result = parse_pointwise_score("I would rate this a 3 out of 5 given the fluency issues.")
    assert result.score == 3
    assert result.parse_ok


def test_out_of_range_number_fails_gracefully():
    result = parse_pointwise_score("I'd give this a 7.")
    assert result.score is None
    assert not result.parse_ok
    assert "not in valid range" in result.parse_note or "none in valid range" in result.parse_note


def test_no_number_fails_gracefully():
    result = parse_pointwise_score("This summary is quite good overall.")
    assert result.score is None
    assert not result.parse_ok


def test_ambiguous_multiple_numbers_fails_gracefully():
    result = parse_pointwise_score("Coherence is 2 but fluency might be closer to 4.")
    assert result.score is None
    assert not result.parse_ok
    assert "ambiguous" in result.parse_note


def test_trailing_punctuation_parses():
    result = parse_pointwise_score("3.")
    assert result.score == 3


def test_build_prompt_contains_article_and_summary():
    prompt = build_pointwise_prompt("Article text here.", "Summary text here.", "coherence")
    assert "Article text here." in prompt
    assert "Summary text here." in prompt
    assert "coherence" in prompt.lower()


def test_build_prompt_unknown_dimension_raises():
    with pytest.raises(ValueError, match="unknown dimension"):
        build_pointwise_prompt("article", "summary", "readability")

def test_out_of_five_phrasing_parses_correctly():
    result = parse_pointwise_score("I would rate this a 3 out of 5 given the fluency issues.")
    assert result.score == 3
    assert result.parse_ok


def test_slash_five_notation_parses_correctly():
    result = parse_pointwise_score("Score: 4/5")
    assert result.score == 4
    assert result.parse_ok

