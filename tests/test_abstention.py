"""Unit tests for the calibration-driven abstention policy."""

import pandas as pd
import pytest

from evalcore.judge.abstention import build_abstention_policy


@pytest.fixture
def synthetic_calibration_df():
    return pd.DataFrame([
        {"dimension": "coherence", "pearson_r": 0.55, "pearson_p": 1e-9, "n": 100},
        {"dimension": "consistency", "pearson_r": 0.55, "pearson_p": 1e-9, "n": 100},
        {"dimension": "fluency", "pearson_r": 0.155, "pearson_p": 0.123, "n": 100},
        {"dimension": "relevance", "pearson_r": 0.48, "pearson_p": 1e-6, "n": 100},
    ])


def test_significant_strong_correlation_is_trusted(synthetic_calibration_df):
    policy = build_abstention_policy(synthetic_calibration_df)
    assert not policy.should_escalate("coherence")
    assert not policy.should_escalate("consistency")
    assert not policy.should_escalate("relevance")


def test_nonsignificant_correlation_is_escalated(synthetic_calibration_df):
    policy = build_abstention_policy(synthetic_calibration_df)
    assert policy.should_escalate("fluency")


def test_unknown_dimension_escalates_by_default(synthetic_calibration_df):
    policy = build_abstention_policy(synthetic_calibration_df)
    assert policy.should_escalate("some_new_dimension_never_calibrated")


def test_weak_but_significant_correlation_still_escalates():
    df = pd.DataFrame([{"dimension": "weak_metric", "pearson_r": 0.1, "pearson_p": 0.001, "n": 10000}])
    policy = build_abstention_policy(df, min_pearson_r=0.3)
    assert policy.should_escalate("weak_metric")


def test_annotate_adds_escalation_column(synthetic_calibration_df):
    policy = build_abstention_policy(synthetic_calibration_df)
    judge_results = pd.DataFrame({
        "dimension": ["coherence", "fluency", "consistency"],
        "judge_score": [4, 2, 4],
    })
    annotated = policy.annotate(judge_results)
    assert list(annotated["escalate_to_human"]) == [False, True, False]


def test_reason_strings_are_informative(synthetic_calibration_df):
    policy = build_abstention_policy(synthetic_calibration_df)
    assert "not statistically significant" in policy.dimension_reliability["fluency"].reason
    assert "significant" in policy.dimension_reliability["coherence"].reason

