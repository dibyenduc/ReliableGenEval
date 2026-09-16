"""Unit tests for the small-sample sensitivity analysis."""

import numpy as np
import pandas as pd
import pytest

from evalcore.analysis.sensitivity import (
    run_sensitivity_analysis,
    summarize_sensitivity,
)


@pytest.fixture
def synthetic_paired_data():
    rng = np.random.default_rng(0)
    n = 500
    doc_ids = pd.Series([f"doc-{i}" for i in range(n)])
    a = pd.Series(rng.normal(4.0, 1.0, size=n))
    b = pd.Series(rng.normal(3.5, 1.0, size=n))
    return a, b, doc_ids


def test_runs_and_returns_expected_row_count(synthetic_paired_data):
    a, b, doc_ids = synthetic_paired_data
    df = run_sensitivity_analysis(a, b, doc_ids, sample_sizes=[25, 50], n_trials=10, random_state=1)
    assert len(df) == 2 * 10
    assert set(df["sample_size"].unique()) == {25, 50}


def test_ci_width_shrinks_as_sample_size_grows(synthetic_paired_data):
    a, b, doc_ids = synthetic_paired_data
    df = run_sensitivity_analysis(a, b, doc_ids, sample_sizes=[25, 250], n_trials=30, random_state=1)
    mean_width_25 = df[df["sample_size"] == 25]["ci_width"].mean()
    mean_width_250 = df[df["sample_size"] == 250]["ci_width"].mean()
    assert mean_width_250 < mean_width_25


def test_sample_size_exceeding_full_data_raises(synthetic_paired_data):
    a, b, doc_ids = synthetic_paired_data
    with pytest.raises(ValueError, match="exceeds full dataset size"):
        run_sensitivity_analysis(a, b, doc_ids, sample_sizes=[10_000], n_trials=5, random_state=1)


def test_summarize_produces_one_row_per_sample_size(synthetic_paired_data):
    a, b, doc_ids = synthetic_paired_data
    df = run_sensitivity_analysis(a, b, doc_ids, sample_sizes=[25, 50, 100], n_trials=10, random_state=1)
    full_est = float(np.asarray(a).mean() - np.asarray(b).mean())
    summary = summarize_sensitivity(df, full_sample_estimate=full_est)
    assert len(summary) == 3
    assert set(summary.columns) >= {"sample_size", "mean_ci_width", "empirical_coverage", "fraction_significant"}


def test_coverage_is_between_zero_and_one(synthetic_paired_data):
    a, b, doc_ids = synthetic_paired_data
    df = run_sensitivity_analysis(a, b, doc_ids, sample_sizes=[50], n_trials=50, random_state=1)
    summary = summarize_sensitivity(df, full_sample_estimate=0.5)
    assert 0.0 <= summary["empirical_coverage"].iloc[0] <= 1.0

