"""Adapts the calibration-driven AbstentionPolicy to the orchestrator's
escalate_fn interface, so real, empirically-grounded escalation logic
(not a placeholder) drives the observability dashboard's escalation
metrics.
"""

from __future__ import annotations

from typing import Any

from evalcore.judge.abstention import AbstentionPolicy


def make_pointwise_escalate_fn(policy: AbstentionPolicy):
    """Returns an escalate_fn(dimension, parsed) -> bool suitable for
    run_jobs()/run_jobs_async(). Escalates if either:
    - the parse itself failed (never trust an unparseable score, regardless
      of how reliable the dimension is on average), or
    - the dimension is not calibration-trusted per the abstention policy
      (e.g. fluency, per the Week 3 finding: r=0.155, p=0.12, not significant).
    """

    def escalate_fn(dimension: str, parsed: Any) -> bool:
        if parsed is None or not getattr(parsed, "parse_ok", False):
            return True
        return policy.should_escalate(dimension)

    return escalate_fn
