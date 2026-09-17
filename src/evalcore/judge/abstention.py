"""Human-escalation / abstention policy for LLM judge scores.

Built directly from empirical judge-vs-human calibration statistics (see
evalcore.analysis.calibration), not an arbitrary threshold. A dimension is
flagged for human escalation if the judge shows no statistically
significant correlation with human scores on that dimension, or if the
correlation is too weak to be practically useful even if significant.

This operationalizes a specific finding: our llama3.1:8b judge showed
r=0.155 (p=0.12, not significant) on fluency, versus r=0.48-0.55 (all
p<1e-6) on coherence/consistency/relevance. A fixed correlation threshold
alone could miss this: a dimension could have p<0.05 by chance with a
weak effect on more data, so both a significance test AND a minimum
effect-size floor are required.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class DimensionReliability:
    dimension: str
    pearson_r: float
    pearson_p: float
    n: int
    trusted: bool
    reason: str


@dataclass
class AbstentionPolicy:
    dimension_reliability: dict[str, DimensionReliability]

    def should_escalate(self, dimension: str) -> bool:
        """Return True if this dimension's judge scores should be escalated
        to human review rather than trusted at face value."""
        if dimension not in self.dimension_reliability:
            return True  # unknown dimension: escalate by default (fail safe)
        return not self.dimension_reliability[dimension].trusted

    def annotate(self, df: pd.DataFrame, dimension_col: str = "dimension") -> pd.DataFrame:
        """Add an 'escalate_to_human' column to a dataframe of judge results."""
        result = df.copy()
        result["escalate_to_human"] = result[dimension_col].apply(self.should_escalate)
        return result


def build_abstention_policy(
    calibration_df: pd.DataFrame,
    min_pearson_r: float = 0.3,
    max_p_value: float = 0.05,
) -> AbstentionPolicy:
    """Build an abstention policy from empirical calibration metrics.

    A dimension is trusted only if BOTH conditions hold:
    - statistically significant correlation with human scores (p < max_p_value)
    - the correlation is at least min_pearson_r (weak-but-significant
      correlations, e.g. from a very large n, are still not practically useful)

    Args:
        calibration_df: output of evalcore.analysis.calibration.compute_calibration_by_dimension
        min_pearson_r: minimum |Pearson r| to consider a dimension usable
        max_p_value: maximum p-value to consider the correlation statistically real
    """
    reliability: dict[str, DimensionReliability] = {}
    for _, row in calibration_df.iterrows():
        dimension = row["dimension"]
        r, p, n = row["pearson_r"], row["pearson_p"], row["n"]

        significant = p < max_p_value
        strong_enough = abs(r) >= min_pearson_r
        trusted = significant and strong_enough

        if not significant:
            reason = f"not statistically significant (p={p:.4f} >= {max_p_value})"
        elif not strong_enough:
            reason = f"correlation too weak (|r|={abs(r):.3f} < {min_pearson_r})"
        else:
            reason = f"significant (p={p:.4f}) and sufficiently strong (r={r:.3f})"

        reliability[dimension] = DimensionReliability(
            dimension=dimension, pearson_r=r, pearson_p=p, n=n, trusted=trusted, reason=reason
        )

    return AbstentionPolicy(dimension_reliability=reliability)

