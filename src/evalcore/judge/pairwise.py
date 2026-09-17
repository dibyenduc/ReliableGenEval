"""Pairwise LLM-as-judge wrapper: compare two summaries head-to-head on
one dimension. Parsing logic kept separate from the network call, same
pattern as evalcore.judge.pointwise."""

from __future__ import annotations

import re
from dataclasses import dataclass

from evalcore.judge.pointwise import DIMENSION_DEFINITIONS

PAIRWISE_PROMPT_TEMPLATE = """You are comparing two summaries of the same article on \
{dimension} ONLY. Ignore all other qualities.

{dimension_definition}

Article:
{article}

Summary A:
{summary_a}

Summary B:
{summary_b}

Which summary is better on {dimension}? Respond with ONLY "A", "B", or "Tie" if they \
are equal quality on {dimension}. Do not explain your answer."""


@dataclass
class ParsedVerdict:
    raw_response: str
    winner: str | None  # "A", "B", "tie", or None if unparseable
    parse_ok: bool
    parse_note: str


def build_pairwise_prompt(article: str, summary_a: str, summary_b: str, dimension: str) -> str:
    if dimension not in DIMENSION_DEFINITIONS:
        raise ValueError(f"unknown dimension: {dimension}. Must be one of {list(DIMENSION_DEFINITIONS)}")
    return PAIRWISE_PROMPT_TEMPLATE.format(
        dimension=dimension,
        dimension_definition=DIMENSION_DEFINITIONS[dimension],
        article=article,
        summary_a=summary_a,
        summary_b=summary_b,
    )


def parse_pairwise_verdict(raw_response: str) -> ParsedVerdict:
    """Extract a winner ('A', 'B', or 'tie') from free-form LLM output.

    Handles: bare tokens ("A", "B.", "Tie"), leading-token responses
    ("A, because..."), 'tie' mentioned anywhere, and responses that only
    name one summary explicitly ("Summary A is better because...").
    """
    stripped = raw_response.strip()

    bare_match = re.fullmatch(r"[\s.]*([AB]|Tie)[\s.]*", stripped, re.IGNORECASE)
    if bare_match:
        token = bare_match.group(1)
        winner = "tie" if token.lower() == "tie" else token.upper()
        return ParsedVerdict(raw_response, winner, True, "bare token")

    if re.search(r"\btie\b", stripped, re.IGNORECASE):
        return ParsedVerdict(raw_response, "tie", True, "'tie' keyword found")

    leading_match = re.match(r"^\W*([AB])\b", stripped, re.IGNORECASE)
    if leading_match:
        return ParsedVerdict(raw_response, leading_match.group(1).upper(), True, "leading letter token")

    a_mentions = len(re.findall(r"\bsummary\s+a\b", stripped, re.IGNORECASE))
    b_mentions = len(re.findall(r"\bsummary\s+b\b", stripped, re.IGNORECASE))
    if a_mentions > 0 and b_mentions == 0:
        return ParsedVerdict(raw_response, "A", True, "only Summary A mentioned")
    if b_mentions > 0 and a_mentions == 0:
        return ParsedVerdict(raw_response, "B", True, "only Summary B mentioned")

    return ParsedVerdict(raw_response, None, False, "could not determine winner from response")

