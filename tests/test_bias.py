"""Unit tests for position-bias resolution logic (pure, no network)."""

from evalcore.judge.bias import resolve_pairwise_winner


def test_resolve_winner_a_maps_to_first_identity():
    assert resolve_pairwise_winner("A", ("summary_1", "summary_2")) == "summary_1"


def test_resolve_winner_b_maps_to_second_identity():
    assert resolve_pairwise_winner("B", ("summary_1", "summary_2")) == "summary_2"


def test_resolve_tie_stays_tie():
    assert resolve_pairwise_winner("tie", ("summary_1", "summary_2")) == "tie"


def test_resolve_none_stays_none():
    assert resolve_pairwise_winner(None, ("summary_1", "summary_2")) is None


def test_resolve_respects_swapped_order():
    # when order is swapped, "A" now refers to summary_2
    assert resolve_pairwise_winner("A", ("summary_2", "summary_1")) == "summary_2"
    assert resolve_pairwise_winner("B", ("summary_2", "summary_1")) == "summary_1"

