
"""Verbosity/style bias stress test: does the pointwise judge reward length
and hedging language over actual content? We construct a 'verbose variant'
of each summary that adds filler and redundant restatement -- but zero new
facts -- then compare judge scores on original vs. padded versions.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

OPENING_FILLERS = [
    "It is important to note that",
    "As the article clearly indicates,",
    "In summary, it can be observed that",
    "Notably, and as one might expect,",
]

CLOSING_HEDGES = [
    "This appears to be the case based on the available information.",
    "That said, further context could add nuance to this point.",
    "Overall, this reflects the key takeaway from the source material.",
]

RESTATEMENT_LEAD = "To put it another way,"


def add_filler_phrases(summary: str, n_insertions: int = 2, seed: int = 0) -> str:
    """Insert hedging/filler phrases before sentences, without changing any
    factual content. Deterministic given seed (cycles through filler lists
    rather than using randomness, so results are reproducible).
    """
    
    stripped = summary.strip()
    if not stripped:
        return summary
    
    sentences = re.split(r"(?<=[.!?])\s+", summary.strip())
    if not sentences:
        return summary

    n_insertions = min(n_insertions, len(sentences))
    out = list(sentences)
    for i in range(n_insertions):
        idx = (i * max(1, len(sentences) // max(n_insertions, 1))) % len(out)
        filler = OPENING_FILLERS[(seed + i) % len(OPENING_FILLERS)]
        out[idx] = f"{filler} {out[idx][0].lower()}{out[idx][1:]}" if out[idx] else out[idx]
    return " ".join(out)


def add_redundant_restatement(summary: str, seed: int = 0) -> str:
    """Append a verbatim restatement of the first sentence, wrapped in a
    'to put it another way' framing. Adds length and the appearance of
    elaboration with literally zero new information.
    """
    stripped = summary.strip()
    if not stripped:
        return summary
    sentences = re.split(r"(?<=[.!?])\s+", stripped)
    first = sentences[0].rstrip(".!?")
    if not first:
        return summary
    restatement = f"{RESTATEMENT_LEAD} {first[0].lower()}{first[1:]}."
    hedge = CLOSING_HEDGES[seed % len(CLOSING_HEDGES)]
    return f"{summary.strip()} {restatement} {hedge}"

def make_verbose_variant(summary: str, seed: int = 0) -> str:
    """Combine filler insertion and redundant restatement to produce a
    substantially longer version of the same summary with no new facts.
    This is the stress-test artifact: a judge that scores this higher than
    the original on quality dimensions is exhibiting length/verbosity bias.
    """
    padded = add_filler_phrases(summary, n_insertions=2, seed=seed)
    padded = add_redundant_restatement(padded, seed=seed)
    return padded


@dataclass
class VerbosityComparison:
    dimension: str
    original_score: int | None
    verbose_score: int | None
    score_delta: int | None  # verbose - original; positive means verbosity was rewarded
    length_ratio: float


def compare_verbosity_scores(
    original_score: int | None,
    verbose_score: int | None,
    dimension: str,
    original_text: str,
    verbose_text: str,
) -> VerbosityComparison:
    delta = None
    if original_score is not None and verbose_score is not None:
        delta = verbose_score - original_score
    length_ratio = len(verbose_text) / len(original_text) if original_text else float("nan")
    return VerbosityComparison(
        dimension=dimension,
        original_score=original_score,
        verbose_score=verbose_score,
        score_delta=delta,
        length_ratio=length_ratio,
    )

