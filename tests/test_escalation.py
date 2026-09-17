"""Unit tests for the AbstentionPolicy -> escalate_fn adapter."""

from dataclasses import dataclass

from evalcore.judge.abstention import AbstentionPolicy, DimensionReliability
from evalcore.orchestration.escalation import make_pointwise_escalate_fn


@dataclass
class FakeParsed:
    parse_ok: bool
    score: int | None = None


def _make_policy() -> AbstentionPolicy:
    return AbstentionPolicy(dimension_reliability={
        "coherence": DimensionReliability("coherence", 0.55, 1e-9, 100, True, "trusted"),
        "consistency": DimensionReliability("consistency", 0.55, 1e-9, 100, True, "trusted"),
        "relevance": DimensionReliability("relevance", 0.48, 1e-6, 100, True, "trusted"),
        "fluency": DimensionReliability("fluency", 0.155, 0.12, 100, False, "not significant"),
    })


def test_trusted_dimension_with_successful_parse_does_not_escalate():
    escalate_fn = make_pointwise_escalate_fn(_make_policy())
    assert escalate_fn("coherence", FakeParsed(parse_ok=True, score=4)) is False


def test_untrusted_dimension_escalates_even_with_successful_parse():
    escalate_fn = make_pointwise_escalate_fn(_make_policy())
    assert escalate_fn("fluency", FakeParsed(parse_ok=True, score=4)) is True


def test_failed_parse_escalates_regardless_of_dimension_trust():
    escalate_fn = make_pointwise_escalate_fn(_make_policy())
    assert escalate_fn("coherence", FakeParsed(parse_ok=False, score=None)) is True


def test_none_parsed_escalates():
    escalate_fn = make_pointwise_escalate_fn(_make_policy())
    assert escalate_fn("coherence", None) is True


def test_unknown_dimension_escalates_by_default():
    escalate_fn = make_pointwise_escalate_fn(_make_policy())
    assert escalate_fn("readability", FakeParsed(parse_ok=True, score=3)) is True


def test_all_trusted_dimensions_do_not_escalate():
    policy = _make_policy()
    escalate_fn = make_pointwise_escalate_fn(policy)
    for dim in ["coherence", "consistency", "relevance"]:
        assert escalate_fn(dim, FakeParsed(parse_ok=True, score=3)) is False

