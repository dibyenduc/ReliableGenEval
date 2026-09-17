
"""Unit tests for the async orchestrator: concurrency bounding, retries,
checkpoint/resume, and escalation wiring. Uses a FakeBackend for full
control over failure timing and concurrency observation."""

import asyncio
from dataclasses import dataclass

from evalcore.orchestration.runner import JudgeJob, get_completed_keys, run_jobs
from evalcore.orchestration.telemetry import TelemetryWriter


@dataclass
class FakeParsed:
    parse_ok: bool
    value: str


def fake_parser(raw: str) -> FakeParsed:
    return FakeParsed(parse_ok=True, value=raw)


class FakeBackend:
    name = "fake"

    def __init__(self, fail_times: int = 0, sleep_s: float = 0.0):
        self.fail_times = fail_times
        self.sleep_s = sleep_s
        self.call_count = 0
        self.concurrent = 0
        self.max_concurrent = 0
        self._lock = asyncio.Lock()

    async def call(self, prompt: str, model: str) -> str:
        async with self._lock:
            self.concurrent += 1
            self.max_concurrent = max(self.max_concurrent, self.concurrent)
        try:
            self.call_count += 1
            if self.sleep_s:
                await asyncio.sleep(self.sleep_s)
            if self.call_count <= self.fail_times:
                raise RuntimeError("fake transient failure")
            return f"response-for:{prompt}"
        finally:
            async with self._lock:
                self.concurrent -= 1


def _make_jobs(n: int, judge_type: str = "pointwise") -> list[JudgeJob]:
    return [
        JudgeJob(item_id=f"doc-{i}", dimension="coherence", judge_type=judge_type, prompt=f"prompt-{i}", parser=fake_parser)
        for i in range(n)
    ]


def test_runs_all_jobs_and_emits_telemetry(tmp_path):
    jobs = _make_jobs(5)
    backend = FakeBackend()
    telemetry_path = tmp_path / "telemetry.jsonl"

    results = run_jobs(jobs, backend, "fake-model", telemetry_path)

    assert len(results) == 5
    assert backend.call_count == 5
    events = TelemetryWriter(telemetry_path).read_all()
    assert len(events) == 5
    assert all(e.parse_ok for e in events)


def test_resume_skips_already_completed_jobs(tmp_path):
    jobs = _make_jobs(3)
    telemetry_path = tmp_path / "telemetry.jsonl"

    backend1 = FakeBackend()
    run_jobs(jobs, backend1, "fake-model", telemetry_path)
    assert backend1.call_count == 3

    backend2 = FakeBackend()
    results2 = run_jobs(jobs, backend2, "fake-model", telemetry_path, resume=True)
    assert backend2.call_count == 0
    assert results2 == []
    assert len(TelemetryWriter(telemetry_path).read_all()) == 3


def test_resume_false_reruns_everything(tmp_path):
    jobs = _make_jobs(3)
    telemetry_path = tmp_path / "telemetry.jsonl"

    run_jobs(jobs, FakeBackend(), "fake-model", telemetry_path)
    run_jobs(jobs, FakeBackend(), "fake-model", telemetry_path, resume=False)

    assert len(TelemetryWriter(telemetry_path).read_all()) == 6


def test_partial_completion_only_reruns_remaining_jobs(tmp_path):
    telemetry_path = tmp_path / "telemetry.jsonl"
    first_batch = _make_jobs(2)
    run_jobs(first_batch, FakeBackend(), "fake-model", telemetry_path)

    full_batch = _make_jobs(5)  # includes the original 2 plus 3 new
    backend2 = FakeBackend()
    results = run_jobs(full_batch, backend2, "fake-model", telemetry_path, resume=True)

    assert backend2.call_count == 3
    assert len(results) == 3
    assert len(TelemetryWriter(telemetry_path).read_all()) == 5


def test_retry_records_retry_count_on_eventual_success(tmp_path):
    jobs = _make_jobs(1)
    backend = FakeBackend(fail_times=2)
    telemetry_path = tmp_path / "telemetry.jsonl"

    results = run_jobs(jobs, backend, "fake-model", telemetry_path, max_retries=3)

    assert results[0].event.retry_count == 2
    assert results[0].event.error is None
    assert results[0].event.parse_ok is True


def test_exhausted_retries_records_error_and_parse_not_ok(tmp_path):
    jobs = _make_jobs(1)
    backend = FakeBackend(fail_times=999)
    telemetry_path = tmp_path / "telemetry.jsonl"

    results = run_jobs(jobs, backend, "fake-model", telemetry_path, max_retries=2)

    assert results[0].event.retry_count == 2
    assert results[0].event.error is not None
    assert results[0].event.parse_ok is False
    assert results[0].parsed is None


def test_concurrency_is_bounded_by_max_concurrency(tmp_path):
    jobs = _make_jobs(10)
    backend = FakeBackend(sleep_s=0.05)
    telemetry_path = tmp_path / "telemetry.jsonl"

    run_jobs(jobs, backend, "fake-model", telemetry_path, max_concurrency=3)

    assert backend.max_concurrent <= 3
    assert backend.max_concurrent > 1  # confirms real concurrency happened, not accidental serialization


def test_escalate_fn_sets_escalated_flag(tmp_path):
    jobs = _make_jobs(2)
    backend = FakeBackend()
    telemetry_path = tmp_path / "telemetry.jsonl"

    def escalate_fn(dimension, parsed):
        return dimension == "coherence"

    results = run_jobs(jobs, backend, "fake-model", telemetry_path, escalate_fn=escalate_fn)

    assert all(r.event.escalated for r in results)


def test_get_completed_keys_reflects_telemetry_log(tmp_path):
    jobs = _make_jobs(2)
    telemetry_path = tmp_path / "telemetry.jsonl"
    run_jobs(jobs, FakeBackend(), "fake-model", telemetry_path)

    keys = get_completed_keys(TelemetryWriter(telemetry_path))
    assert keys == {("doc-0", "coherence", "pointwise"), ("doc-1", "coherence", "pointwise")}

