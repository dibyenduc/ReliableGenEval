"""Structured per-call telemetry for judge invocations, in the shape a
production observability backend (Prometheus, a data warehouse, Grafana
dashboards) would ingest. Emitted as JSONL: streamable, append-only, safe
across resumed/checkpointed runs.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class JudgeCallEvent:
    item_id: str
    dimension: str
    judge_type: str  # 'pointwise', 'pairwise', 'pairwise_cot'
    backend: str      # 'ollama', 'mock_cloud', etc.
    model: str
    started_at: float
    latency_ms: float
    parse_ok: bool
    escalated: bool
    retry_count: int
    error: str | None = None
    extra: dict = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(asdict(self))


class TelemetryWriter:
    """Appends JudgeCallEvent records to a JSONL file. Always appends,
    never truncates -- multiple runs (including resumed ones) against the
    same path accumulate a single continuous event log, the way a real
    observability pipeline would.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def emit(self, event: JudgeCallEvent) -> None:
        with open(self.path, "a") as f:
            f.write(event.to_json() + "\n")

    def read_all(self) -> list[JudgeCallEvent]:
        if not self.path.exists():
            return []
        events = []
        with open(self.path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                events.append(JudgeCallEvent(**json.loads(line)))
        return events


def now_s() -> float:
    return time.time()

