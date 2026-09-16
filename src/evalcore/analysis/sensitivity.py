"""Small-sample sensitivity analysis: how much can you trust a bootstrap CI
or permutation p-value as sample size shrinks?

For each target sample size, draws many independent random subsamples,
runs the paired bootstrap CI and permutation test on each, and records:
- CI width (does it get properly wider as n shrinks, or misleadingly narrow?)
- whether the CI contains the full-sample ("ground truth") point estimate
  (empirical coverage — should be close to the nominal CI level if calibrated)
- permutation test p-value (does significance become unstable at small n?)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from evalcore.stats.bootstrap import paired_bootstrap_ci
from evalcore.stats.permutation import paired_permutation_test


@dataclass
class SensitivityRow:
    sample_size: int
    trial: int
    point_estimate: float
    ci_lower: float
    ci_upper: float
    ci_width: float
    contains_full_sample_estimate: bool
    p_value: float
    significant_at_05: bool


def run_sensitivity_analysis(
    scores_a: pd.Series,
    scores_b: pd.Series,
    doc_ids: pd.Series,
    sample_sizes: list[int],
    n_trials: int = 200,
    n_bootstrap_resamples: int = 2000,
    n_permutations: int = 2000,
    ci_level: float = 0.95,
    random_state: int = 42,
) -> pd.DataFrame:
    """Run repeated-subsampling sensitivity analysis across sample sizes.

    Args:
        scores_a, scores_b: paired full-sample score series for two models.
        doc_ids: article id per row (kept for potential cluster-aware
            subsampling in a later iteration; unused for naive subsampling here).
        sample_sizes: list of subsample sizes to test, e.g. [25, 50, 100, 250].
            The full sample size is handled separately as the reference.
        n_trials: number of independent random subsamples per sample size.
        random_state: base seed; each (size, trial) combination gets a
            deterministic derived seed for reproducibility.

    Returns:
        Long-format DataFrame, one row per (sample_size, trial).
    """
    a_full = np.asarray(scores_a, dtype=float)
    b_full = np.asarray(scores_b, dtype=float)
    n_full = len(a_full)

    full_estimate = float(a_full.mean() - b_full.mean())

    rng = np.random.default_rng(random_state)
    rows: list[SensitivityRow] = []

    for size in sample_sizes:
        if size > n_full:
            raise ValueError(f"sample_size {size} exceeds full dataset size {n_full}")
        for trial in range(n_trials):
            trial_seed = int(rng.integers(0, 2**31 - 1))
            trial_rng = np.random.default_rng(trial_seed)
            idx = trial_rng.choice(n_full, size=size, replace=False)
            a_sub = pd.Series(a_full[idx])
            b_sub = pd.Series(b_full[idx])

            boot_result = paired_bootstrap_ci(
                a_sub, b_sub, n_resamples=n_bootstrap_resamples, ci_level=ci_level, random_state=trial_seed
            )
            perm_result = paired_permutation_test(
                a_sub, b_sub, n_permutations=n_permutations, random_state=trial_seed
            )

            rows.append(
                SensitivityRow(
                    sample_size=size,
                    trial=trial,
                    point_estimate=boot_result.point_estimate,
                    ci_lower=boot_result.ci_lower,
                    ci_upper=boot_result.ci_upper,
                    ci_width=boot_result.ci_upper - boot_result.ci_lower,
                    contains_full_sample_estimate=(boot_result.ci_lower <= full_estimate <= boot_result.ci_upper),
                    p_value=perm_result.p_value,
                    significant_at_05=perm_result.p_value < 0.05,
                )
            )

    return pd.DataFrame([vars(r) for r in rows])

def summarize_sensitivity(df: pd.DataFrame, full_sample_estimate: float, ci_level: float = 0.95) -> pd.DataFrame:
    """Aggregate per-trial results into per-sample-size summary statistics.

    Includes coverage_se: the standard error of the empirical coverage
    estimate (binomial SE), so callers can judge whether an observed
    coverage gap from the nominal CI level is statistically meaningful
    or just noise from a limited number of trials.
    """
    summary = (
        df.groupby("sample_size")
        .agg(
            n_trials=("trial", "count"),
            mean_ci_width=("ci_width", "mean"),
            std_ci_width=("ci_width", "std"),
            empirical_coverage=("contains_full_sample_estimate", "mean"),
            mean_point_estimate=("point_estimate", "mean"),
            std_point_estimate=("point_estimate", "std"),
            fraction_significant=("significant_at_05", "mean"),
            median_p_value=("p_value", "median"),
        )
        .reset_index()
    )
    p = summary["empirical_coverage"]
    n = summary["n_trials"]
    summary["coverage_se"] = np.sqrt(p * (1 - p) / n)
    summary["nominal_ci_level"] = ci_level
    summary["full_sample_estimate"] = full_sample_estimate
    return summary

