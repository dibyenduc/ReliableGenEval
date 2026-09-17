"""Pointwise LLM-as-judge wrapper: score a single summary on one dimension.

Design principle: parsing/validation logic (pure, testable, no network) is
kept separate from the Ollama call itself (network-dependent, non-deterministic,
excluded from CI). This lets us unit-test the hard part -- robustly extracting
a numeric score from messy free-form LLM output -- without needing a live
Ollama server in CI.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

SCORE_MIN, SCORE_MAX = 1, 5

POINTWISE_PROMPT_TEMPLATE = """You are evaluating the {dimension} of a summary.

{dimension_definition}

Article:
{article}

Summary:
{summary}

Rate the {dimension} of this summary on a scale of 1 to 5, where 1 is very poor \
and 5 is excellent. Respond with ONLY the integer score, nothing else."""

DIMENSION_DEFINITIONS = {
    "coherence": "Coherence measures whether the summary is well-structured and "
                 "organized, building from sentence to sentence into a coherent body of information.",
    "consistency": "Consistency measures whether the facts in the summary are "
                    "consistent with the facts in the article, with no contradictions or fabrications.",
    "fluency": "Fluency measures the quality of individual sentences: are they "
               "grammatical, well-formed, and easy to read.",
    "relevance": "Relevance measures whether the summary includes only important "
                  "information from the article, without redundant or irrelevant content.",
}


@dataclass
class ParsedScore:
    raw_response: str
    score: int | None
    parse_ok: bool
    parse_note: str


def build_pointwise_prompt(article: str, summary: str, dimension: str) -> str:
    if dimension not in DIMENSION_DEFINITIONS:
        raise ValueError(f"unknown dimension: {dimension}. Must be one of {list(DIMENSION_DEFINITIONS)}")
    return POINTWISE_PROMPT_TEMPLATE.format(
        dimension=dimension,
        dimension_definition=DIMENSION_DEFINITIONS[dimension],
        article=article,
        summary=summary,
    )


def parse_pointwise_score(raw_response: str) -> ParsedScore:
    """Extract an integer score in [SCORE_MIN, SCORE_MAX] from free-form LLM output.

    Handles: bare numbers ("2"), numbers with trailing punctuation ("2."),
    "X out of 5" / "X/5" phrasing (common LLM pattern -- the "5" here is
    boilerplate, not a second candidate score), numbers embedded in text
    ("I'd rate this a 2"), and failure cases (no number found, out of range,
    genuinely multiple conflicting numbers).
    """
    stripped = raw_response.strip()

    bare_match = re.fullmatch(r"[1-5]", stripped)
    if bare_match:
        return ParsedScore(raw_response, int(bare_match.group()), True, "bare integer")

    out_of_five_match = re.search(r"\b([1-5])\s*(?:out of|/)\s*5\b", stripped, re.IGNORECASE)
    if out_of_five_match:
        return ParsedScore(
            raw_response, int(out_of_five_match.group(1)), True, "extracted from 'X out of 5' / 'X/5' pattern"
        )

    numbers = re.findall(r"\b[1-5]\b", stripped)
    if len(numbers) == 1:
        return ParsedScore(raw_response, int(numbers[0]), True, "single valid number embedded in text")

    if len(numbers) > 1:
        return ParsedScore(raw_response, None, False, f"ambiguous: found {len(numbers)} candidate numbers {numbers}")

    any_digit = re.findall(r"\d+", stripped)
    if any_digit:
        return ParsedScore(raw_response, None, False, f"found number(s) {any_digit} but none in valid range [1,5]")

    return ParsedScore(raw_response, None, False, "no number found in response")

