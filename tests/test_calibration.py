"""Unit tests for judge-vs-human calibration metrics, using synthetic
judge/human score pairs with known agreement properties."""

import numpy as np
import pandas as pd
import pytest

from evalcore.analysis.calibration import (
    compute_calibration_by_dimension,
    compute_calibration_metrics,
)


def test_perfect_agreement_gives_correlation_one():
    human = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    judge = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    result = compute_calibration_metrics(judge, human)
    assert result.pearson_r == pytest.approx(1.0)
    assert result.spearman_r == pytest.approx(1.0)
    assert result.mean_absolute_error == pytest.approx(0.0)
    assert result.mean_signed_error == pytest.approx(0.0)


def test_systematic_positive_bias_detected():
    human = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    judge = human + 1.0
    result = compute_calibration_metrics(judge, human)
    assert result.pearson_r == pytest.approx(1.0)
    assert result.mean_signed_error == pytest.approx(1.0)
    assert result.mean_absolute_error == pytest.approx(1.0)


def test_systematic_negative_bias_detected():
    human = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    judge = human - 1.0
    result = compute_calibration_metrics(judge, human)
    assert result.mean_signed_error == pytest.approx(-1.0)


def test_no_correlation_gives_near_zero_r():
    rng = np.random.default_rng(0)
    human = pd.Series(rng.uniform(1, 5, size=200))
    judge = pd.Series(rng.uniform(1, 5, size=200))
    result = compute_calibration_metrics(judge, human)
    assert abs(result.pearson_r) < 0.2


def test_mismatched_lengths_raise():
    with pytest.raises(ValueError, match="equal length"):
        compute_calibration_metrics(pd.Series([1.0, 2.0]), pd.Series([1.0]))


def test_too_few_observations_raise():
    with pytest.raises(ValueError, match="at least 2"):
        compute_calibration_metrics(pd.Series([1.0]), pd.Series([1.0]))


def test_by_dimension_produces_one_row_per_dimension():
    df = pd.DataFrame({
        "dimension": ["coherence"] * 5 + ["fluency"] * 5,
        "judge_score": [1, 2, 3, 4, 5, 5, 4, 3, 2, 1],
        "human_score": [1, 2, 3, 4, 5, 1, 2, 3, 4, 5],
    })
    result = compute_calibration_by_dimension(df)
    assert len(result) == 2
    assert set(result["dimension"]) == {"coherence", "fluency"}
    coherence_row = result[result["dimension"] == "coherence"].iloc[0]
    fluency_row = result[result["dimension"] == "fluency"].iloc[0]
    assert coherence_row["pearson_r"] > 0.9
    assert fluency_row["pearson_r"] < -0.9

