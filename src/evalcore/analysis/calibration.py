"""Judge-vs-human calibration metrics: how well do LLM judge scores agree
with human annotations, overall and per dimension."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class CalibrationResult:
    dimension: str
    n: int
    pearson_r: float
    pearson_p: float
    spearman_r: float
    spearman_p: float
    mean_absolute_error: float
    mean_signed_error: float


def compute_calibration_metrics(judge_scores: pd.Series, human_scores: pd.Series, dimension: str = "overall") -> CalibrationResult:
    """Compute agreement metrics between judge scores and human scores.

    mean_signed_error = mean(judge - human): positive means judge scores
    systematically higher than humans, negative means systematically lower
    (a bias check, distinct from correlation which only measures whether
    they move together, not whether they're offset).
    """
    judge = np.asarray(judge_scores, dtype=float)
    human = np.asarray(human_scores, dtype=float)
    if len(judge) != len(human):
        raise ValueError(f"judge_scores and human_scores must have equal length, got {len(judge)} vs {len(human)}")
    if len(judge) < 2:
        raise ValueError("need at least 2 paired observations to compute correlation")

    pearson_r, pearson_p = stats.pearsonr(judge, human)
    spearman_r, spearman_p = stats.spearmanr(judge, human)
    mae = float(np.mean(np.abs(judge - human)))
    mse = float(np.mean(judge - human))

    return CalibrationResult(
        dimension=dimension,
        n=len(judge),
        pearson_r=float(pearson_r),
        pearson_p=float(pearson_p),
        spearman_r=float(spearman_r),
        spearman_p=float(spearman_p),
        mean_absolute_error=mae,
        mean_signed_error=mse,
    )


def compute_calibration_by_dimension(results_df: pd.DataFrame) -> pd.DataFrame:
    """Given a long-format DataFrame with columns [dimension, judge_score,
    human_score], compute calibration metrics separately for each dimension."""
    rows = []
    for dimension, group in results_df.groupby("dimension"):
        valid = group.dropna(subset=["judge_score", "human_score"])
        if len(valid) < 2:
            continue
        result = compute_calibration_metrics(valid["judge_score"], valid["human_score"], dimension=dimension)
        rows.append(vars(result))
    return pd.DataFrame(rows)

