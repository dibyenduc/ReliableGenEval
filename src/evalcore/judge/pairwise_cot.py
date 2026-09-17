"""Chain-of-thought pairwise judge: forces the model to briefly justify its
comparison before giving a verdict, to test whether requiring reasoning
recovers a real content signal (as opposed to the bare A/B/Tie prompt in
pairwise.py, which showed zero recoverable content signal on disagreement --
see position-bias debiasing study)."""

from __future__ import annotations

import re

from evalcore.judge.pairwise import ParsedVerdict, parse_pairwise_verdict
from evalcore.judge.pointwise import DIMENSION_DEFINITIONS

COT_PAIRWISE_PROMPT_TEMPLATE = """You are comparing two summaries of the same article on \
{dimension} ONLY. Ignore all other qualities.

{dimension_definition}

Article:
{article}

Summary A:
{summary_a}

Summary B:
{summary_b}

First, in 1-2 sentences each, briefly analyze Summary A's {dimension} and Summary B's \
{dimension}, citing specific evidence from the article or summaries. Then, on its own \
final line, give your verdict in EXACTLY this format (nothing else on that line):

VERDICT: A
(or VERDICT: B, or VERDICT: Tie)"""

VERDICT_LINE_RE = re.compile(r"verdict:\s*(a|b|tie)\b", re.IGNORECASE)


def build_cot_pairwise_prompt(article: str, summary_a: str, summary_b: str, dimension: str) -> str:
    if dimension not in DIMENSION_DEFINITIONS:
        raise ValueError(f"unknown dimension: {dimension}. Must be one of {list(DIMENSION_DEFINITIONS)}")
    return COT_PAIRWISE_PROMPT_TEMPLATE.format(
        dimension=dimension,
        dimension_definition=DIMENSION_DEFINITIONS[dimension],
        article=article,
        summary_a=summary_a,
        summary_b=summary_b,
    )


def parse_cot_pairwise_verdict(raw_response: str) -> ParsedVerdict:
    """Extract the verdict from a 'VERDICT: X' marker line. Takes the LAST
    such marker if multiple appear (some models restate it). Falls back to
    the legacy bare-token/mention-based parser (from pairwise.py) if no
    marker is found, so responses that ignore the format instruction are
    still recoverable where possible.
    """
    matches = VERDICT_LINE_RE.findall(raw_response)
    if matches:
        token = matches[-1]
        winner = "tie" if token.lower() == "tie" else token.upper()
        return ParsedVerdict(raw_response, winner, True, "found 'VERDICT:' marker")

    fallback = parse_pairwise_verdict(raw_response)
    return ParsedVerdict(
        raw_response,
        fallback.winner,
        fallback.parse_ok,
        f"no VERDICT marker found; fallback parse: {fallback.parse_note}",
    )

