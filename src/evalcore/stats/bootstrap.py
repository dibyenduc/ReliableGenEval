"""Paired bootstrap confidence intervals for comparing two models' scores.

Supports both naive per-row resampling and cluster bootstrap (resampling
whole articles, keeping all of an article's summaries together). SummEval
has 16 summaries per article, and those 16 aren't independent observations
— they share a source document, and article-level difficulty/topic can
shift all 16 scores together. Naive resampling treats them as independent
and will understate CI width; cluster resampling respects that structure.
Both are implemented so the choice is explicit and inspectable, not buried.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class BootstrapResult:
    point_estimate: float
    ci_lower: float
    ci_upper: float
    n_resamples: int
    cluster_by: str | None
    n_units_resampled: int


def paired_bootstrap_ci(
    scores_a: pd.Series,
    scores_b: pd.Series,
    doc_ids: pd.Series | None = None,
    cluster_by: str | None = None,
    n_resamples: int = 10_000,
    ci_level: float = 0.95,
    statistic: str = "mean_diff",
    random_state: int | None = 42,
) -> BootstrapResult:
    """Bootstrap CI for the difference in a statistic between two paired
    score series (e.g. model A's coherence scores vs model B's, same articles).

    Args:
        scores_a, scores_b: paired score series, same length, same order.
        doc_ids: article id for each row; required if cluster_by="article".
        cluster_by: None for naive per-row resampling, or "article" to
            resample whole articles (all their rows move together).
        n_resamples: number of bootstrap resamples.
        ci_level: confidence level, e.g. 0.95 for a 95% CI.
        statistic: "mean_diff" (mean(a) - mean(b)) or "median_diff".
        random_state: seed for reproducibility.

    Returns:
        BootstrapResult with point estimate, CI bounds, and metadata.
    """
    if len(scores_a) != len(scores_b):
        raise ValueError(f"scores_a and scores_b must have equal length, got {len(scores_a)} vs {len(scores_b)}")
    if cluster_by == "article" and doc_ids is None:
        raise ValueError("doc_ids is required when cluster_by='article'")
    if statistic not in ("mean_diff", "median_diff"):
        raise ValueError(f"unsupported statistic: {statistic}")

    a = np.asarray(scores_a, dtype=float)
    b = np.asarray(scores_b, dtype=float)
    rng = np.random.default_rng(random_state)

    def compute_stat(x: np.ndarray, y: np.ndarray) -> float:
        if statistic == "mean_diff":
            return float(np.mean(x) - np.mean(y))
        return float(np.median(x) - np.median(y))

    point_estimate = compute_stat(a, b)

    if cluster_by is None:
        n = len(a)
        n_units = n
        boot_stats = np.empty(n_resamples)
        for i in range(n_resamples):
            idx = rng.integers(0, n, size=n)
            boot_stats[i] = compute_stat(a[idx], b[idx])
    elif cluster_by == "article":
        doc_arr = np.asarray(doc_ids)
        unique_docs = np.unique(doc_arr)
        n_units = len(unique_docs)
        doc_to_rows = {d: np.where(doc_arr == d)[0] for d in unique_docs}
        boot_stats = np.empty(n_resamples)
        for i in range(n_resamples):
            sampled_docs = rng.choice(unique_docs, size=n_units, replace=True)
            idx = np.concatenate([doc_to_rows[d] for d in sampled_docs])
            boot_stats[i] = compute_stat(a[idx], b[idx])
    else:
        raise ValueError(f"unsupported cluster_by: {cluster_by}")

    alpha = 1 - ci_level
    ci_lower = float(np.percentile(boot_stats, 100 * (alpha / 2)))
    ci_upper = float(np.percentile(boot_stats, 100 * (1 - alpha / 2)))

    return BootstrapResult(
        point_estimate=point_estimate,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        n_resamples=n_resamples,
        cluster_by=cluster_by,
        n_units_resampled=n_units,
    )

