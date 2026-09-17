"""Unit tests for verbosity-variant generation (pure, no network)."""

from evalcore.judge.style_bias import (
    add_filler_phrases,
    add_redundant_restatement,
    compare_verbosity_scores,
    make_verbose_variant,
)

SAMPLE = "The company reported strong earnings. Revenue grew by 12 percent. Analysts were surprised by the results."


def test_filler_phrases_increase_length():
    padded = add_filler_phrases(SAMPLE, n_insertions=2, seed=0)
    assert len(padded) > len(SAMPLE)


def test_filler_phrases_deterministic_given_seed():
    a = add_filler_phrases(SAMPLE, n_insertions=2, seed=3)
    b = add_filler_phrases(SAMPLE, n_insertions=2, seed=3)
    assert a == b


def test_filler_phrases_different_seeds_can_differ():
    a = add_filler_phrases(SAMPLE, n_insertions=2, seed=0)
    b = add_filler_phrases(SAMPLE, n_insertions=2, seed=1)
    assert a != b


def test_redundant_restatement_preserves_original_text():
    restated = add_redundant_restatement(SAMPLE, seed=0)
    assert SAMPLE.strip() in restated


def test_redundant_restatement_adds_no_new_numbers():
    restated = add_redundant_restatement(SAMPLE, seed=0)
    import re
    original_numbers = set(re.findall(r"\d+", SAMPLE))
    new_numbers = set(re.findall(r"\d+", restated))
    assert new_numbers == original_numbers


def test_make_verbose_variant_substantially_longer():
    verbose = make_verbose_variant(SAMPLE, seed=0)
    assert len(verbose) > len(SAMPLE) * 1.3


def test_make_verbose_variant_deterministic():
    a = make_verbose_variant(SAMPLE, seed=5)
    b = make_verbose_variant(SAMPLE, seed=5)
    assert a == b


def test_compare_verbosity_scores_computes_positive_delta():
    comparison = compare_verbosity_scores(3, 5, "coherence", SAMPLE, "x" * 200)
    assert comparison.score_delta == 2
    assert comparison.length_ratio > 1


def test_compare_verbosity_scores_handles_none_scores():
    comparison = compare_verbosity_scores(None, 5, "fluency", SAMPLE, "padded")
    assert comparison.score_delta is None


def test_empty_summary_handled_gracefully():
    assert add_filler_phrases("", n_insertions=2, seed=0) == ""
    assert add_redundant_restatement("", seed=0) == ""

