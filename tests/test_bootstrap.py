"""Unit tests for paired bootstrap CI, using synthetic data with known
properties so we can assert on statistical behavior, not just 'runs without error'."""

import numpy as np
import pandas as pd
import pytest

from evalcore.stats.bootstrap import paired_bootstrap_ci


def test_zero_difference_ci_contains_zero():
    rng = np.random.default_rng(0)
    a = pd.Series(rng.normal(3.0, 1.0, size=500))
    b = pd.Series(rng.normal(3.0, 1.0, size=500))
    result = paired_bootstrap_ci(a, b, n_resamples=2000, random_state=1)
    assert result.ci_lower < 0 < result.ci_upper


def test_large_true_difference_ci_excludes_zero():
    rng = np.random.default_rng(0)
    a = pd.Series(rng.normal(4.5, 0.5, size=500))
    b = pd.Series(rng.normal(2.5, 0.5, size=500))
    result = paired_bootstrap_ci(a, b, n_resamples=2000, random_state=1)
    assert result.ci_lower > 0
    assert result.point_estimate == pytest.approx(2.0, abs=0.2)


def test_reproducible_with_fixed_seed():
    a = pd.Series(np.arange(100, dtype=float))
    b = pd.Series(np.arange(100, dtype=float) * 0.9)
    r1 = paired_bootstrap_ci(a, b, n_resamples=500, random_state=7)
    r2 = paired_bootstrap_ci(a, b, n_resamples=500, random_state=7)
    assert r1.ci_lower == r2.ci_lower
    assert r1.ci_upper == r2.ci_upper


def test_mismatched_lengths_raise():
    a = pd.Series([1.0, 2.0, 3.0])
    b = pd.Series([1.0, 2.0])
    with pytest.raises(ValueError, match="equal length"):
        paired_bootstrap_ci(a, b)


def test_cluster_by_article_requires_doc_ids():
    a = pd.Series([1.0, 2.0])
    b = pd.Series([1.0, 2.0])
    with pytest.raises(ValueError, match="doc_ids is required"):
        paired_bootstrap_ci(a, b, cluster_by="article")


def test_cluster_bootstrap_gives_wider_ci_than_naive_when_clustered():
    """Core statistical claim: when the two models' scores are BOTH driven
    by shared per-article difficulty, but respond to it differently (so the
    article effect doesn't cancel in the paired difference), cluster
    bootstrap should produce a wider (more honest) CI than naive per-row
    resampling."""
    rng = np.random.default_rng(3)
    n_docs = 20
    rows_per_doc = 16
    doc_ids = np.repeat([f"doc-{i}" for i in range(n_docs)], rows_per_doc)

    effect_a = np.repeat(rng.normal(0, 1.5, size=n_docs), rows_per_doc)
    effect_b = np.repeat(rng.normal(0, 1.5, size=n_docs), rows_per_doc)
    noise_a = rng.normal(0, 0.2, size=n_docs * rows_per_doc)
    noise_b = rng.normal(0, 0.2, size=n_docs * rows_per_doc)

    a = pd.Series(3.0 + effect_a + noise_a)
    b = pd.Series(3.0 + effect_b + noise_b)
    doc_ids = pd.Series(doc_ids)

    naive = paired_bootstrap_ci(a, b, n_resamples=3000, random_state=5, cluster_by=None)
    clustered = paired_bootstrap_ci(a, b, doc_ids=doc_ids, n_resamples=3000, random_state=5, cluster_by="article")

    naive_width = naive.ci_upper - naive.ci_lower
    clustered_width = clustered.ci_upper - clustered.ci_lower
    assert clustered_width > naive_width
    assert clustered.n_units_resampled == n_docs
    assert naive.n_units_resampled == n_docs * rows_per_doc

def test_median_diff_statistic():
    a = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    b = pd.Series([1.0, 2.0, 3.0, 4.0, 100.0])
    result = paired_bootstrap_ci(a, b, statistic="median_diff", n_resamples=1000, random_state=1)
    assert result.point_estimate == pytest.approx(0.0, abs=0.01)

