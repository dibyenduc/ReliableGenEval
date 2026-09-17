"""Generates a self-contained HTML observability dashboard from a
JudgeCallEvent telemetry log. Pure stdlib + simple HTML/CSS -- no chart
library dependency, so the output is a single portable file.
"""

from __future__ import annotations

from pathlib import Path

from evalcore.orchestration.telemetry import JudgeCallEvent, TelemetryWriter


def _percentile(sorted_values: list[float], pct: float) -> float:
    if not sorted_values:
        return 0.0
    idx = round(pct / 100 * (len(sorted_values) - 1))
    return sorted_values[idx]


def _bucket_throughput(events: list[JudgeCallEvent], n_buckets: int = 20) -> list[tuple[float, int]]:
    if not events:
        return []
    starts = [e.started_at for e in events]
    t0, t1 = min(starts), max(starts)
    span = max(t1 - t0, 1e-6)
    bucket_width = span / n_buckets
    counts = [0] * n_buckets
    for t in starts:
        idx = min(int((t - t0) / bucket_width), n_buckets - 1)
        counts[idx] += 1
    return [(t0 + i * bucket_width, c) for i, c in enumerate(counts)]


def build_dashboard_html(events: list[JudgeCallEvent], title: str = "Judge Pipeline Observability") -> str:
    n = len(events)
    if n == 0:
        return f"<html><body><h1>{title}</h1><p>No telemetry events found.</p></body></html>"

    n_ok = sum(1 for e in events if e.parse_ok)
    n_retried = sum(1 for e in events if e.retry_count > 0)
    n_error = sum(1 for e in events if e.error is not None)
    n_escalated = sum(1 for e in events if e.escalated)

    t0, t1 = min(e.started_at for e in events), max(e.started_at for e in events)
    duration = max(t1 - t0, 1e-6)
    throughput = n / duration

    latencies = sorted(e.latency_ms for e in events)
    p50, p90, p95, p99 = (_percentile(latencies, p) for p in (50, 90, 95, 99))

    dims = sorted({e.dimension for e in events})
    dim_rows = []
    for d in dims:
        d_events = [e for e in events if e.dimension == d]
        d_ok = sum(1 for e in d_events if e.parse_ok)
        d_esc = sum(1 for e in d_events if e.escalated)
        d_lat = sum(e.latency_ms for e in d_events) / len(d_events)
        dim_rows.append((d, len(d_events), d_ok / len(d_events), d_esc / len(d_events), d_lat))

    buckets = _bucket_throughput(events, n_buckets=20)
    max_bucket = max((c for _, c in buckets), default=1) or 1
    bars = "".join(
        f'<div class="bar" style="height:{(c / max_bucket) * 100:.0f}%" title="{c} calls"></div>'
        for _, c in buckets
    )

    error_rows = "".join(
        f"<tr><td>{e.item_id}</td><td>{e.dimension}</td><td>{e.retry_count}</td><td>{e.error or '&mdash;'}</td></tr>"
        for e in events if e.error is not None or e.retry_count > 0
    )
    if not error_rows:
        error_rows = '<tr><td colspan="4"><em>No retries or errors recorded.</em></td></tr>'

    dim_table_rows = "".join(
        f"<tr><td>{d}</td><td>{cnt}</td><td>{ok:.1%}</td><td>{esc:.1%}</td><td>{lat:.0f} ms</td></tr>"
        for d, cnt, ok, esc, lat in dim_rows
    )

    return f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>{title}</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 40px; background: #f7f8fa; color: #1a1a2e; }}
  h1 {{ font-size: 22px; margin-bottom: 4px; }}
  .subtitle {{ color: #666; margin-bottom: 32px; font-size: 13px; }}
  .kpi-grid {{ display: grid; grid-template-columns: repeat(5, 1fr); gap: 16px; margin-bottom: 32px; }}
  .kpi {{ background: white; border-radius: 8px; padding: 16px; box-shadow: 0 1px 3px rgba(0,0,0,0.08); }}
  .kpi .value {{ font-size: 28px; font-weight: 700; }}
  .kpi .label {{ font-size: 12px; color: #666; text-transform: uppercase; letter-spacing: 0.05em; }}
  .kpi.good .value {{ color: #16a34a; }}
  .kpi.warn .value {{ color: #d97706; }}
  section {{ background: white; border-radius: 8px; padding: 20px 24px; margin-bottom: 24px; box-shadow: 0 1px 3px rgba(0,0,0,0.08); }}
  section h2 {{ font-size: 15px; margin-top: 0; margin-bottom: 16px; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
  th, td {{ text-align: left; padding: 8px 10px; border-bottom: 1px solid #eee; }}
  th {{ color: #666; font-weight: 600; }}
  .latency-grid {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; }}
  .latency-cell {{ text-align: center; padding: 12px; background: #f0f2f5; border-radius: 6px; }}
  .latency-cell .val {{ font-size: 20px; font-weight: 700; }}
  .latency-cell .lbl {{ font-size: 11px; color: #666; }}
  .throughput-chart {{ display: flex; align-items: flex-end; height: 100px; gap: 3px; }}
  .bar {{ flex: 1; background: #4f46e5; border-radius: 2px 2px 0 0; min-height: 2px; }}
  code {{ background: #f0f2f5; padding: 2px 6px; border-radius: 4px; font-size: 12px; }}
</style>
</head>
<body>
  <h1>{title}</h1>
  <p class="subtitle">{n} events &middot; backend: {events[0].backend} &middot; model: {events[0].model} &middot; duration: {duration:.1f}s</p>

  <div class="kpi-grid">
    <div class="kpi good"><div class="value">{n:,}</div><div class="label">Total Calls</div></div>
    <div class="kpi good"><div class="value">{throughput:.0f}/s</div><div class="label">Throughput</div></div>
    <div class="kpi good"><div class="value">{n_ok/n:.1%}</div><div class="label">Parse Success</div></div>
    <div class="kpi warn"><div class="value">{n_retried/n:.1%}</div><div class="label">Needed Retry</div></div>
    <div class="kpi warn"><div class="value">{n_error/n:.1%}</div><div class="label">Failed (post-retry)</div></div>
  </div>

  <section>
    <h2>Latency Distribution</h2>
    <div class="latency-grid">
      <div class="latency-cell"><div class="val">{p50:.0f}ms</div><div class="lbl">p50</div></div>
      <div class="latency-cell"><div class="val">{p90:.0f}ms</div><div class="lbl">p90</div></div>
      <div class="latency-cell"><div class="val">{p95:.0f}ms</div><div class="lbl">p95</div></div>
      <div class="latency-cell"><div class="val">{p99:.0f}ms</div><div class="lbl">p99</div></div>
    </div>
  </section>

  <section>
    <h2>Throughput Over Time ({n_escalated} escalated to human review)</h2>
    <div class="throughput-chart">{bars}</div>
  </section>

  <section>
    <h2>Breakdown by Dimension</h2>
    <table>
      <tr><th>Dimension</th><th>Calls</th><th>Parse OK</th><th>Escalated</th><th>Avg Latency</th></tr>
      {dim_table_rows}
    </table>
  </section>

  <section>
    <h2>Retries &amp; Errors</h2>
    <table>
      <tr><th>Item ID</th><th>Dimension</th><th>Retry Count</th><th>Error</th></tr>
      {error_rows}
    </table>
  </section>
</body>
</html>"""


def generate_dashboard(telemetry_path: str | Path, output_path: str | Path, title: str = "Judge Pipeline Observability") -> Path:
    events = TelemetryWriter(telemetry_path).read_all()
    html = build_dashboard_html(events, title=title)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html)
    return output_path

