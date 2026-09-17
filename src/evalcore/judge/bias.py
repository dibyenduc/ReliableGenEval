"""Bias stress tests for the pairwise judge.

Position bias: does the judge's preference track the actual content, or
does it systematically favor whichever summary happens to be shown first
(or second)? We test this by calling the judge twice on the same pair --
once in each order -- and checking whether the *content* preference
(not the positional A/B label) stays consistent.
"""

from __future__ import annotations

from dataclasses import dataclass

from evalcore.judge.ollama_client import call_ollama
from evalcore.judge.pairwise import (
    build_pairwise_prompt,
    parse_pairwise_verdict,
)


@dataclass
class PositionBiasResult:
    dimension: str
    winner_original_order: str | None  # 'summary_1', 'summary_2', or 'tie'
    winner_swapped_order: str | None
    consistent: bool | None  # None if either verdict failed to parse
    note: str


def resolve_pairwise_winner(verdict_winner: str | None, order: tuple[str, str]) -> str | None:
    """Map a positional verdict ('A'/'B'/'tie') back to which actual summary
    won, given which summary was shown in slot A and which in slot B.

    Args:
        verdict_winner: 'A', 'B', 'tie', or None (parse failure).
        order: (identity_shown_as_A, identity_shown_as_B), e.g.
            ('summary_1', 'summary_2').
    """
    if verdict_winner == "A":
        return order[0]
    if verdict_winner == "B":
        return order[1]
    if verdict_winner == "tie":
        return "tie"
    return None


def run_position_bias_check(
    article: str,
    summary_1: str,
    summary_2: str,
    dimension: str,
    model: str = "llama3.1:latest",
    prompt_builder=build_pairwise_prompt,
    verdict_parser=parse_pairwise_verdict,
) -> PositionBiasResult:
    """... (same docstring as before) ...
    prompt_builder/verdict_parser let this be reused with alternative
    pairwise prompt designs (e.g. chain-of-thought) without duplicating
    the order-swap/resolve logic.
    """
    prompt_original = prompt_builder(article, summary_1, summary_2, dimension)
    verdict_original = verdict_parser(call_ollama(prompt_original, model=model))
    winner_original = resolve_pairwise_winner(verdict_original.winner, ("summary_1", "summary_2"))

    prompt_swapped = prompt_builder(article, summary_2, summary_1, dimension)
    verdict_swapped = verdict_parser(call_ollama(prompt_swapped, model=model))
    winner_swapped = resolve_pairwise_winner(verdict_swapped.winner, ("summary_2", "summary_1"))

    if winner_original is None or winner_swapped is None:
        consistent = None
        note = "one or both verdicts failed to parse; cannot assess consistency"
    elif winner_original == winner_swapped:
        consistent = True
        note = f"consistent: preferred '{winner_original}' regardless of position"
    else:
        consistent = False
        note = f"POSITION BIAS DETECTED: preferred '{winner_original}' when shown first, '{winner_swapped}' when shown second"

    return PositionBiasResult(
        dimension=dimension,
        winner_original_order=winner_original,
        winner_swapped_order=winner_swapped,
        consistent=consistent,
        note=note,
    )

import pandas as pd

POSITION_BIAS_THRESHOLD = 0.2  # escalate a dimension if >20% of comparisons flip with position


def summarize_position_bias(results: pd.DataFrame, threshold: float = POSITION_BIAS_THRESHOLD) -> pd.DataFrame:
    """Aggregate a position-bias study's raw comparisons into a per-dimension
    verdict: is the pairwise judge reliable enough (on position) to use as-is,
    or should it be escalated / order-debiased before trusting its output?

    Expects `results` to have columns 'dimension' and 'consistent' (bool),
    as produced by scripts/run_position_bias_study.py.
    """
    rows = []
    for dimension, group in results.groupby("dimension"):
        n_comparisons = len(group)
        n_inconsistent = int((~group["consistent"]).sum())
        inconsistency_rate = n_inconsistent / n_comparisons if n_comparisons else 0.0
        escalate = inconsistency_rate > threshold
        if escalate:
            reason = (
                f"position-inconsistency {inconsistency_rate:.1%} exceeds threshold "
                f"{threshold:.0%}; pairwise verdicts on this dimension are not reliable "
                f"without order-debiasing (e.g. majority vote across both orders)"
            )
        else:
            reason = (
                f"position-inconsistency {inconsistency_rate:.1%} is within threshold "
                f"{threshold:.0%}; pairwise verdicts are usable as-is"
            )
        rows.append({
            "dimension": dimension,
            "n_comparisons": n_comparisons,
            "n_inconsistent": n_inconsistent,
            "inconsistency_rate": inconsistency_rate,
            "escalate": escalate,
            "reason": reason,
        })
    return pd.DataFrame(rows).sort_values("dimension").reset_index(drop=True)

from collections import Counter


@dataclass
class DebiasedVerdict:
    winner: str | None  # 'summary_1', 'summary_2', 'tie', or None
    agreement: bool  # True if all calls used agreed without needing a tie-break
    n_calls: int
    note: str


def combine_two_orders(winner_original: str | None, winner_swapped: str | None) -> tuple[str | None, bool]:
    """Combine winners from two presentation orders of the same pair.

    Returns (combined_winner, agreement). If either call failed to parse,
    the combined result is None (inconclusive). If both orders agree on
    the same content-identity winner (including 'tie'), that's returned
    with agreement=True. If they disagree, returns ('tie', False) --
    treating position-dependent flips as inconclusive rather than
    arbitrarily picking one side.
    """
    if winner_original is None or winner_swapped is None:
        return None, False
    if winner_original == winner_swapped:
        return winner_original, True
    return "tie", False


def combine_three_verdicts(v1: str | None, v2: str | None, v3: str | None) -> tuple[str | None, bool]:
    """Majority vote across three verdicts (used as a tie-break when the
    first two orders disagree). Returns (winner, unanimous)."""
    votes = [v for v in (v1, v2, v3) if v is not None]
    if not votes:
        return None, False
    counts = Counter(votes)
    top_winner, top_count = counts.most_common(1)[0]
    unanimous = top_count == len(votes) and len(votes) == 3
    if top_count > len(votes) / 2:
        return top_winner, unanimous
    return "tie", False  # no majority -> inconclusive

import random


def run_debiased_pairwise_judge(
    article: str,
    summary_1: str,
    summary_2: str,
    dimension: str,
    model: str = "llama3.1:latest",
    use_tiebreak: bool = True,
    rng_seed: int | None = None,
    prompt_builder=build_pairwise_prompt,
    verdict_parser=parse_pairwise_verdict,
) -> DebiasedVerdict:
    """... (same docstring, plus:)
    prompt_builder/verdict_parser let this be reused with alternative
    pairwise prompt designs (e.g. chain-of-thought) without duplicating
    the order-swap/tie-break/majority-vote logic.
    """
    prompt_original = prompt_builder(article, summary_1, summary_2, dimension)
    verdict_original = verdict_parser(call_ollama(prompt_original, model=model))
    winner_original = resolve_pairwise_winner(verdict_original.winner, ("summary_1", "summary_2"))

    prompt_swapped = prompt_builder(article, summary_2, summary_1, dimension)
    verdict_swapped = verdict_parser(call_ollama(prompt_swapped, model=model))
    winner_swapped = resolve_pairwise_winner(verdict_swapped.winner, ("summary_2", "summary_1"))

    combined, agreement = combine_two_orders(winner_original, winner_swapped)

    if agreement or not use_tiebreak or combined is None:
        note = (
            f"orders agreed on '{combined}'" if agreement
            else "one or both verdicts failed to parse; inconclusive"
        )
        return DebiasedVerdict(winner=combined, agreement=agreement, n_calls=2, note=note)

    rng = random.Random(rng_seed)
    show_1_first = rng.random() < 0.5
    if show_1_first:
        prompt_tiebreak = prompt_original
        tiebreak_order = ("summary_1", "summary_2")
    else:
        prompt_tiebreak = prompt_swapped
        tiebreak_order = ("summary_2", "summary_1")

    verdict_tiebreak = verdict_parser(call_ollama(prompt_tiebreak, model=model))
    winner_tiebreak = resolve_pairwise_winner(verdict_tiebreak.winner, tiebreak_order)

    final, unanimous = combine_three_verdicts(winner_original, winner_swapped, winner_tiebreak)
    note = (
        f"orders disagreed ({winner_original} vs {winner_swapped}); "
        f"random tie-break ({'orig' if show_1_first else 'swapped'} order): '{final}'"
        f"{' (unanimous)' if unanimous else ''}"
    )
    return DebiasedVerdict(winner=final, agreement=unanimous, n_calls=3, note=note)

