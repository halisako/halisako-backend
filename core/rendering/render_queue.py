"""In-memory lifecycle manager for rendering jobs.

RenderQueue's sole responsibility is job lifecycle management: accept
jobs, hand them out in order, track their state, and record how they
finished. It is renderer-agnostic — it has no knowledge of ComfyUI,
Stable Diffusion, video generation, FFmpeg, or GPUs, and no knowledge
of how a job's scenes are actually rendered. It only manages the
bookkeeping around a RenderJob's lifecycle.

Two notes on reconciling this module's spec with the existing
RenderJob/RenderStatus schemas (core/rendering/job_models.py), which
this module does not modify:

1. RenderStatus has no RUNNING member. RENDERING is used instead —
   it's the existing status that means "a worker is actively
   processing this job," which is what RUNNING would mean here.
2. RenderJob has no started_at, completed_at, failure-reason, or
   retry-count fields. Rather than add them to the shared schema,
   RenderQueue tracks them itself, keyed by job_id, and exposes them
   through small accessor methods (get_started_at, get_completed_at,
   get_failure_reason, get_retry_count). This is lifecycle metadata
   about a job, not a fact belonging on the job's own record.

Priority is accepted and stored on each job but does not affect
dequeue order — next_job() is strict FIFO over the internal deque,
matching the instruction to use collections.deque as the queue.
"""

from __future__ import annotations

import logging
import threading
from collections import deque
from datetime import datetime, timezone

from core.rendering.job_models import RenderJob, RenderResult, RenderStatus

logger = logging.getLogger(__name__)

# Statuses under RenderQueue's own control that count as "waiting to run"
# for cancellation purposes. Every other status this module sets
# (RENDERING, COMPLETED, FAILED, CANCELLED) is not cancellable.
_CANCELLABLE_STATUSES = (RenderStatus.NEW, RenderStatus.QUEUED)


def _utcnow() -> datetime:
    """Returns the current time in UTC — factored out for one
    consistent source of timestamps, and so it's the one place a
    future test would patch to control time."""
    return datetime.now(timezone.utc)


class RenderQueue:
    """A thread-safe, in-memory FIFO queue of RenderJobs.

    Two internal structures back every operation: a `collections.deque`
    of job IDs giving FIFO order and O(1) append/popleft, and a
    `dict[job_id, RenderJob]` giving O(1) lookup by ID. A single
    `threading.Lock` guards both, so every public method here is safe
    to call from multiple threads concurrently.

    This class holds everything in memory and is not durable — nothing
    here persists across a process restart. That's out of scope for
    this module, same as Redis, Celery, or any other backing store.
    """

    def __init__(self) -> None:
        self._queue: deque[str] = deque()
        self._jobs: dict[str, RenderJob] = {}
        self._lock = threading.Lock()

        # Lifecycle metadata not present on RenderJob itself — see the
        # module docstring for why this lives here instead of on the
        # schema.
        self._started_at: dict[str, datetime] = {}
        self._completed_at: dict[str, datetime] = {}
        self._failure_reasons: dict[str, str] = {}
        self._retry_counts: dict[str, int] = {}

    def submit(self, job: RenderJob) -> str:
        """Adds a job to the queue.

        Sets the job's status to NEW and stores it. If `job.job_id` is
        already known to this queue, the stored record is replaced but
        the job is not re-enqueued a second time, to avoid it being
        handed out twice by next_job().

        Args:
            job: The job to enqueue. Ownership of this object passes
                to the queue — callers should not mutate it directly
                afterward; use the queue's own methods instead.

        Returns:
            The job's `job_id`.
        """
        with self._lock:
            already_known = job.job_id in self._jobs
            job.update_progress(job.progress, status=RenderStatus.NEW)
            self._jobs[job.job_id] = job

            if already_known:
                logger.warning(
                    "Job %s re-submitted — replacing stored record, not re-queuing.",
                    job.job_id,
                )
            else:
                self._queue.append(job.job_id)
                logger.info(
                    "Job %s submitted (product=%s, priority=%s).",
                    job.job_id, job.product, job.priority.value,
                )

            return job.job_id

    def next_job(self) -> RenderJob | None:
        """Pops and returns the next waiting job, in FIFO order.

        Marks the job's status as RENDERING (this module's stand-in
        for "running" — see the module docstring) and records its
        start time. Returns None without side effects if no job is
        waiting.

        Returns:
            The job that was next in line, now marked RENDERING, or
            None if the queue is empty.
        """
        with self._lock:
            if not self._queue:
                return None

            job_id = self._queue.popleft()
            job = self._jobs[job_id]
            job.update_progress(job.progress, status=RenderStatus.RENDERING)
            self._started_at[job_id] = _utcnow()

            logger.info("Job %s started.", job_id)
            return job

    def get(self, job_id: str) -> RenderJob | None:
        """Returns the job with the given ID, or None if unknown."""
        with self._lock:
            return self._jobs.get(job_id)

    def list_jobs(self) -> list[RenderJob]:
        """Returns a snapshot list of every job known to this queue,
        regardless of status."""
        with self._lock:
            return list(self._jobs.values())

    def cancel(self, job_id: str) -> bool:
        """Cancels a job, if it hasn't started running yet.

        Only allowed while the job's status is NEW or QUEUED — a job
        that has already started (RENDERING) cannot be cancelled here.

        Args:
            job_id: The job to cancel.

        Returns:
            True if the job was cancelled. False if the job is
            unknown, or is not in a cancellable state.
        """
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                logger.warning("Cannot cancel unknown job %s.", job_id)
                return False

            if job.status not in _CANCELLABLE_STATUSES:
                logger.warning(
                    "Cannot cancel job %s in status %s.", job_id, job.status.value
                )
                return False

            job.update_progress(job.progress, status=RenderStatus.CANCELLED)
            self._remove_from_queue(job_id)

            logger.info("Job %s cancelled.", job_id)
            return True

    def mark_completed(self, job_id: str, result: RenderResult) -> bool:
        """Marks a job as completed and stores its result.

        Args:
            job_id: The job to complete.
            result: The render's output artifacts.

        Returns:
            True if the job was found and updated. False if the job
            is unknown.
        """
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                logger.warning("Cannot complete unknown job %s.", job_id)
                return False

            job.update_progress(100, status=RenderStatus.COMPLETED)
            job.result = result
            self._completed_at[job_id] = _utcnow()

            logger.info("Job %s completed.", job_id)
            return True

    def mark_failed(self, job_id: str, reason: str) -> bool:
        """Marks a job as failed, stores the failure reason, and
        increments its retry count.

        Args:
            job_id: The job to fail.
            reason: A human-readable description of why it failed.

        Returns:
            True if the job was found and updated. False if the job
            is unknown.
        """
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                logger.warning("Cannot fail unknown job %s.", job_id)
                return False

            job.update_progress(job.progress, status=RenderStatus.FAILED)
            self._failure_reasons[job_id] = reason
            self._retry_counts[job_id] = self._retry_counts.get(job_id, 0) + 1

            logger.info("Job %s failed: %s", job_id, reason)
            return True

    def retry(self, job_id: str) -> bool:
        """Moves a failed job back onto the queue.

        Only allowed while the job's status is FAILED. The job's
        status resets to QUEUED (distinct from the NEW a fresh
        submit() produces, so a job's history is visible in its
        current status) and it re-enters the queue at the back, in
        FIFO order with everything else waiting.

        Args:
            job_id: The job to retry.

        Returns:
            True if the job was requeued. False if the job is unknown
            or not currently FAILED.
        """
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                logger.warning("Cannot retry unknown job %s.", job_id)
                return False

            if job.status != RenderStatus.FAILED:
                logger.warning(
                    "Cannot retry job %s in status %s (must be FAILED).",
                    job_id, job.status.value,
                )
                return False

            job.update_progress(job.progress, status=RenderStatus.QUEUED)
            self._queue.append(job_id)

            logger.info("Job %s requeued for retry.", job_id)
            return True

    def stats(self) -> dict[str, int]:
        """Returns job counts by coarse lifecycle bucket.

        "queued" covers both NEW and QUEUED (not yet started); RENDERING and
        any of the schema's other in-progress statuses count as "running."

        Returns:
            A dict with keys "queued", "running", "completed", "failed",
            "cancelled", and "total".
        """
        with self._lock:
            counts = {
                "queued": 0, "running": 0, "completed": 0, "failed": 0, "cancelled": 0
            }
            for job in self._jobs.values():
                if job.status in _CANCELLABLE_STATUSES:
                    counts["queued"] += 1
                elif job.status == RenderStatus.COMPLETED:
                    counts["completed"] += 1
                elif job.status == RenderStatus.FAILED:
                    counts["failed"] += 1
                elif job.status == RenderStatus.CANCELLED:
                    counts["cancelled"] += 1
                else:
                    counts["running"] += 1
            counts["total"] = len(self._jobs)
            return counts

    def get_started_at(self, job_id: str) -> datetime | None:
        """Returns when the job most recently started running (its
        most recent next_job() call), or None if it hasn't started."""
        with self._lock:
            return self._started_at.get(job_id)

    def get_completed_at(self, job_id: str) -> datetime | None:
        """Returns when the job completed, or None if it hasn't."""
        with self._lock:
            return self._completed_at.get(job_id)

    def get_failure_reason(self, job_id: str) -> str | None:
        """Returns the most recent failure reason recorded for the
        job, or None if it has never failed."""
        with self._lock:
            return self._failure_reasons.get(job_id)

    def get_retry_count(self, job_id: str) -> int:
        """Returns how many times the job has failed (and so become
        eligible for retry). Zero if it has never failed."""
        with self._lock:
            return self._retry_counts.get(job_id, 0)

    def _remove_from_queue(self, job_id: str) -> None:
        """Removes a job ID from the waiting deque, if present.

        Assumes the caller already holds `self._lock`. A no-op if the
        ID isn't currently queued (e.g. it's already running).
        """
        try:
            self._queue.remove(job_id)
        except ValueError:
            pass


if __name__ == "__main__":
    import sys

    sys.path.insert(0, ".")
    from core.rendering.job_models import RenderPriority

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )

    queue = RenderQueue()

    print("=== Creating and submitting 3 jobs ===")
    jobs = [
        RenderJob(
            job_id=f"job_{i}",
            user_id="demo_user",
            product="chess2fight",
            priority=RenderPriority.NORMAL,
            workflow_version="v1.0.0",
        )
        for i in (1, 2, 3)
    ]
    for job in jobs:
        queue.submit(job)
    print(queue.stats())

    print("\n=== Starting all 3 jobs (next_job x3) ===")
    running = [queue.next_job() for _ in range(3)]
    for job in running:
        assert job is not None
        started = queue.get_started_at(job.job_id)
        print(f"  {job.job_id}: status={job.status.value}, started_at={started}")
    print(queue.stats())

    print("\n=== Completing job_1 ===")
    result = RenderResult(
        video_path="out/job_1.mp4",
        thumbnail_path="out/job_1.png",
        gif_preview_path="out/job_1.gif",
        render_time_seconds=42.0,
        workflow_version="v1.0.0",
        seed=123,
    )
    queue.mark_completed("job_1", result)
    print(queue.stats())

    print("\n=== Failing job_2 ===")
    queue.mark_failed("job_2", "GPU out of memory")
    print(f"  failure reason: {queue.get_failure_reason('job_2')!r}")
    print(f"  retry count: {queue.get_retry_count('job_2')}")
    print(queue.stats())

    print("\n=== Retrying job_2 ===")
    queue.retry("job_2")
    print(queue.stats())

    print("\n=== Attempting to cancel job_3 (currently RENDERING — should fail) ===")
    print(f"  cancel result: {queue.cancel('job_3')}")

    print("\n=== Final state of every job ===")
    for job in queue.list_jobs():
        print(f"  {job.job_id}: status={job.status.value}, progress={job.progress}")
