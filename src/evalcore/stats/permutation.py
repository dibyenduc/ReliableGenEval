"""Paired permutation test (sign-flip test) for comparing two models.

Uses sign-flip permutation rather than group-label shuffling because our
data is paired: model A and model B are scored on the SAME articles. Under
the null hypothesis of no true difference, each article's observed
difference is equally likely to be positive or negative, so we build the
null distribution by randomly flipping signs of the per-article differences
rather than shuffling which rows belong to which model.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class PermutationResult:
    observed_statistic: float
    p_value: float
    n_permutations: int
    null_distribution_std: float


def paired_permutation_test(
    scores_a: pd.Series,
    scores_b: pd.Series,
    n_permutations: int = 10_000,
    statistic: str = "mean_diff",
    alternative: str = "two-sided",
    random_state: int | None = 42,
) -> PermutationResult:
    """Sign-flip paired permutation test on scores_a - scores_b.

    Args:
        scores_a, scores_b: paired score series, same length, same order.
        n_permutations: number of sign-flip resamples for the null distribution.
        statistic: "mean_diff" or "median_diff" of (a - b).
        alternative: "two-sided", "greater" (a > b), or "less" (a < b).
        random_state: seed for reproducibility.

    Returns:
        PermutationResult with the observed statistic, p-value, and metadata.
    """
    if len(scores_a) != len(scores_b):
        raise ValueError(f"scores_a and scores_b must have equal length, got {len(scores_a)} vs {len(scores_b)}")
    if statistic not in ("mean_diff", "median_diff"):
        raise ValueError(f"unsupported statistic: {statistic}")
    if alternative not in ("two-sided", "greater", "less"):
        raise ValueError(f"unsupported alternative: {alternative}")

    diffs = np.asarray(scores_a, dtype=float) - np.asarray(scores_b, dtype=float)
    n = len(diffs)
    rng = np.random.default_rng(random_state)

    def compute_stat(x: np.ndarray) -> float:
        return float(np.mean(x)) if statistic == "mean_diff" else float(np.median(x))

    observed = compute_stat(diffs)

    signs = rng.choice([-1.0, 1.0], size=(n_permutations, n))
    null_stats = np.empty(n_permutations)
    for i in range(n_permutations):
        null_stats[i] = compute_stat(diffs * signs[i])

    if alternative == "two-sided":
        p_value = float(np.mean(np.abs(null_stats) >= np.abs(observed)))
    elif alternative == "greater":
        p_value = float(np.mean(null_stats >= observed))
    else:
        p_value = float(np.mean(null_stats <= observed))

    return PermutationResult(
        observed_statistic=observed,
        p_value=p_value,
        n_permutations=n_permutations,
        null_distribution_std=float(np.std(null_stats)),
    )

