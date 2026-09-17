"""Async job orchestrator: bounded concurrency, automatic retries with
backoff, and checkpoint/resume support, all backend-agnostic via the
JudgeBackend protocol. This is the layer that makes "same code, more
workers, different backend = large scale" literally true rather than
aspirational -- max_concurrency and backend are the only things that
change between a laptop-scale run and a production-scale run.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from evalcore.judge.backends import JudgeBackend
from evalcore.orchestration.telemetry import JudgeCallEvent, TelemetryWriter, now_s


@dataclass
class JudgeJob:
    item_id: str
    dimension: str
    judge_type: str  # 'pointwise', 'pairwise', 'pairwise_cot', etc.
    prompt: str
    parser: Callable[[str], Any]  # returns an object exposing .parse_ok


@dataclass
class JobResult:
    item_id: str
    dimension: str
    judge_type: str
    parsed: Any | None
    event: JudgeCallEvent


def get_completed_keys(telemetry: TelemetryWriter) -> set[tuple[str, str, str]]:
    """Keys of jobs that already reached a terminal outcome (success or
    retries exhausted) in a prior run against this telemetry log.
    """
    return {(e.item_id, e.dimension, e.judge_type) for e in telemetry.read_all()}


async def _run_one_job(
    job: JudgeJob,
    backend: JudgeBackend,
    model: str,
    telemetry: TelemetryWriter,
    semaphore: asyncio.Semaphore,
    max_retries: int,
    escalate_fn: Callable[[str, Any], bool] | None,
) -> JobResult:
    async with semaphore:
        started_at = now_s()
        raw_response: str | None = None
        error: str | None = None
        retry_count = 0

        while True:
            try:
                raw_response = await backend.call(job.prompt, model)
                error = None
                break
            except Exception as exc:  # noqa: BLE001 -- deliberately broad: any backend failure is retryable
                error = str(exc)
                if retry_count >= max_retries:
                    break
                retry_count += 1
                await asyncio.sleep(0.05 * retry_count)  # simple linear backoff

        latency_ms = (now_s() - started_at) * 1000

        parsed = None
        parse_ok = False
        if raw_response is not None:
            parsed = job.parser(raw_response)
            parse_ok = bool(getattr(parsed, "parse_ok", False))

        escalated = bool(escalate_fn(job.dimension, parsed)) if escalate_fn else False

        event = JudgeCallEvent(
            item_id=job.item_id,
            dimension=job.dimension,
            judge_type=job.judge_type,
            backend=backend.name,
            model=model,
            started_at=started_at,
            latency_ms=latency_ms,
            parse_ok=parse_ok,
            escalated=escalated,
            retry_count=retry_count,
            error=error,
        )
        telemetry.emit(event)
        return JobResult(job.item_id, job.dimension, job.judge_type, parsed, event)


async def run_jobs_async(
    jobs: list[JudgeJob],
    backend: JudgeBackend,
    model: str,
    telemetry_path: str | Path,
    max_concurrency: int = 8,
    max_retries: int = 2,
    escalate_fn: Callable[[str, Any], bool] | None = None,
    resume: bool = True,
) -> list[JobResult]:
    """Run all jobs concurrently (bounded by max_concurrency), skipping
    any job whose (item_id, dimension, judge_type) already has a terminal
    telemetry event on disk when resume=True. This is the whole
    checkpoint/resume story: the telemetry log IS the checkpoint, so
    there's no separate state file to get out of sync.
    """
    telemetry = TelemetryWriter(telemetry_path)
    completed = get_completed_keys(telemetry) if resume else set()
    pending = [j for j in jobs if (j.item_id, j.dimension, j.judge_type) not in completed]

    semaphore = asyncio.Semaphore(max_concurrency)
    tasks = [
        _run_one_job(job, backend, model, telemetry, semaphore, max_retries, escalate_fn)
        for job in pending
    ]
    return list(await asyncio.gather(*tasks))


def run_jobs(
    jobs: list[JudgeJob],
    backend: JudgeBackend,
    model: str,
    telemetry_path: str | Path,
    **kwargs: Any,
) -> list[JobResult]:
    """Synchronous convenience wrapper for scripts/CLIs that aren't
    already inside an event loop."""
    return asyncio.run(run_jobs_async(jobs, backend, model, telemetry_path, **kwargs))
