"""Unit tests for the HTML dashboard generator (pure string/HTML output,
no browser rendering needed)."""

from evalcore.orchestration.dashboard import build_dashboard_html, generate_dashboard
from evalcore.orchestration.telemetry import JudgeCallEvent, TelemetryWriter


def _make_event(item_id, dimension="coherence", latency_ms=200.0, parse_ok=True, escalated=False, retry_count=0, error=None, started_at=1000.0):
    return JudgeCallEvent(
        item_id=item_id, dimension=dimension, judge_type="pointwise", backend="mock_cloud",
        model="fake-model", started_at=started_at, latency_ms=latency_ms, parse_ok=parse_ok,
        escalated=escalated, retry_count=retry_count, error=error,
    )


def test_empty_events_produces_placeholder_html():
    html = build_dashboard_html([])
    assert "No telemetry events found" in html


def test_kpi_values_reflect_event_counts():
    events = [_make_event("a", parse_ok=True), _make_event("b", parse_ok=False, error="timeout")]
    html = build_dashboard_html(events)
    assert "2" in html  # total calls
    assert "50.0%" in html  # parse success rate


def test_latency_percentiles_appear_in_output():
    events = [_make_event(f"item-{i}", latency_ms=float(i * 10)) for i in range(1, 101)]
    html = build_dashboard_html(events)
    assert "p50" in html
    assert "p99" in html


def test_dimension_breakdown_includes_all_dimensions():
    events = [_make_event("a", dimension="coherence"), _make_event("b", dimension="fluency")]
    html = build_dashboard_html(events)
    assert "coherence" in html
    assert "fluency" in html


def test_error_rows_render_when_errors_present():
    events = [_make_event("a", error="rate limited", retry_count=2)]
    html = build_dashboard_html(events)
    assert "rate limited" in html


def test_no_errors_shows_placeholder_message():
    events = [_make_event("a", parse_ok=True)]
    html = build_dashboard_html(events)
    assert "No retries or errors recorded" in html


def test_escalation_count_appears():
    events = [_make_event("a", escalated=True), _make_event("b", escalated=False)]
    html = build_dashboard_html(events)
    assert "1 escalated" in html


def test_generate_dashboard_writes_file(tmp_path):
    telemetry_path = tmp_path / "telemetry.jsonl"
    writer = TelemetryWriter(telemetry_path)
    writer.emit(_make_event("a"))
    writer.emit(_make_event("b"))

    output_path = tmp_path / "dashboard.html"
    result_path = generate_dashboard(telemetry_path, output_path)

    assert result_path == output_path
    assert output_path.exists()
    content = output_path.read_text()
    assert "<html>" in content


def test_generate_dashboard_creates_parent_dirs(tmp_path):
    telemetry_path = tmp_path / "telemetry.jsonl"
    TelemetryWriter(telemetry_path).emit(_make_event("a"))

    output_path = tmp_path / "nested" / "report" / "dashboard.html"
    generate_dashboard(telemetry_path, output_path)
    assert output_path.exists()

