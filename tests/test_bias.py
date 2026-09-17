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

import pandas as pd
import pytest

from evalcore.judge.bias import summarize_position_bias


def _make_results(dimension: str, consistent_flags: list[bool]) -> pd.DataFrame:
    return pd.DataFrame({
        "dimension": [dimension] * len(consistent_flags),
        "consistent": consistent_flags,
    })


def test_all_consistent_is_not_escalated():
    df = _make_results("relevance", [True] * 10)
    summary = summarize_position_bias(df)
    row = summary.iloc[0]
    assert row["escalate"] == False
    assert row["inconsistency_rate"] == 0.0


def test_high_inconsistency_is_escalated():
    df = _make_results("consistency", [False] * 8 + [True] * 2)
    summary = summarize_position_bias(df)
    row = summary.iloc[0]
    assert row["escalate"] == True
    assert row["inconsistency_rate"] == pytest.approx(0.8)


def test_borderline_below_threshold_not_escalated():
    # 1/10 = 10% inconsistency, threshold is 20%
    df = _make_results("fluency", [False] + [True] * 9)
    summary = summarize_position_bias(df, threshold=0.2)
    assert summary.iloc[0]["escalate"] == False


def test_multiple_dimensions_produce_one_row_each():
    df = pd.concat([
        _make_results("coherence", [True, False, True]),
        _make_results("fluency", [False, False, False]),
    ])
    summary = summarize_position_bias(df)
    assert len(summary) == 2
    assert set(summary["dimension"]) == {"coherence", "fluency"}


def test_reason_strings_mention_threshold_and_rate():
    df = _make_results("coherence", [False] * 5)
    summary = summarize_position_bias(df, threshold=0.2)
    reason = summary.iloc[0]["reason"]
    assert "100.0%" in reason
    assert "20%" in reason

