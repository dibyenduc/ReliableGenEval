"""Unit tests for the JSONL telemetry schema and writer."""

import json

from evalcore.orchestration.telemetry import JudgeCallEvent, TelemetryWriter


def _make_event(item_id="doc-1", **overrides):
    defaults = {
        "item_id": item_id,
        "dimension": "coherence",
        "judge_type": "pointwise",
        "backend": "mock_cloud",
        "model": "fake-model",
        "started_at": 1000.0,
        "latency_ms": 250.0,
        "parse_ok": True,
        "escalated": False,
        "retry_count": 0,
        "error": None,
    }
    defaults.update(overrides)
    return JudgeCallEvent(**defaults)


def test_event_to_json_round_trips():
    event = _make_event()
    parsed = json.loads(event.to_json())
    assert parsed["item_id"] == "doc-1"
    assert parsed["dimension"] == "coherence"
    assert parsed["parse_ok"] is True


def test_writer_emits_and_reads_back(tmp_path):
    path = tmp_path / "telemetry.jsonl"
    writer = TelemetryWriter(path)
    writer.emit(_make_event(item_id="doc-1"))
    writer.emit(_make_event(item_id="doc-2"))

    events = writer.read_all()
    assert len(events) == 2
    assert {e.item_id for e in events} == {"doc-1", "doc-2"}


def test_writer_appends_across_instances_not_truncates(tmp_path):
    path = tmp_path / "telemetry.jsonl"
    TelemetryWriter(path).emit(_make_event(item_id="doc-1"))
    TelemetryWriter(path).emit(_make_event(item_id="doc-2"))  # simulates a resumed run

    events = TelemetryWriter(path).read_all()
    assert len(events) == 2


def test_read_all_on_nonexistent_path_returns_empty_list(tmp_path):
    path = tmp_path / "does_not_exist.jsonl"
    assert TelemetryWriter(path).read_all() == []


def test_writer_creates_parent_directories(tmp_path):
    path = tmp_path / "nested" / "dir" / "telemetry.jsonl"
    writer = TelemetryWriter(path)
    writer.emit(_make_event())
    assert path.exists()


def test_event_preserves_error_and_extra_fields():
    event = _make_event(error="timeout", extra={"prompt_tokens": 512})
    parsed = json.loads(event.to_json())
    assert parsed["error"] == "timeout"
    assert parsed["extra"] == {"prompt_tokens": 512}

