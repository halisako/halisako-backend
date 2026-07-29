"""Background runtime that continuously consumes jobs from a RenderQueue
and hands each one to a renderer.

RenderWorker knows exactly three things: a queue to pull jobs from, a
renderer to hand them to, and how to record the outcome. It has no
knowledge of what a renderer actually does — no ComfyUI, no FFmpeg, no
GPU details, no chess, no HTTP. `renderer` is any object with a
`render(job) -> RenderResult` method; this module never imports or
assumes anything about what's on the other side of that call.

One note on reconciling this module's spec with the existing
WorkerInfo schema (core/rendering/job_models.py), which this module
does not modify: `health()` is specified to return a WorkerInfo, but
that schema requires `gpu_type` and `vram_gb` (positive, no default) —
fields describing GPU hardware this worker, by design, knows nothing
about. Rather than add a way around those constraints to the shared
schema, `health()` populates them with clearly-named sentinel values
("unknown" / 1) — see `health()`'s docstring. Real GPU details, if
ever needed, belong to whatever layer actually tracks hardware
allocation, not to this worker.
"""

from __future__ import annotations

import logging
import threading
import time
import uuid
from datetime import datetime, timezone
from typing import Protocol

from core.rendering.job_models import RenderJob, RenderResult, WorkerInfo, WorkerStatus
from core.rendering.render_queue import RenderQueue

logger = logging.getLogger(__name__)

# How long to sleep after finding the queue empty before polling again.
_POLL_INTERVAL_SECONDS = 0.5

# Sentinel values for WorkerInfo fields this worker has no real data
# for — see the module docstring.
_UNKNOWN_GPU_TYPE = "unknown"
_PLACEHOLDER_VRAM_GB = 1


def _utcnow() -> datetime:
    """Returns the current time in UTC — one consistent source of
    timestamps for this module."""
    return datetime.now(timezone.utc)


class Renderer(Protocol):
    """Structural interface RenderWorker expects of a renderer.

    Any object with a matching `render` method satisfies this —
    nothing needs to inherit from this class. RenderWorker never
    checks `isinstance` against it; it exists purely for type hints.
    """

    def render(self, job: RenderJob) -> RenderResult:
        """Renders the given job and returns its result.

        Implementations should raise on failure rather than returning
        a sentinel value — RenderWorker treats any exception here as
        a failed render and records it as such.
        """
        ...


class RenderWorker:
    """Pulls jobs from a RenderQueue, one at a time, and renders them.

    A worker is single-threaded with respect to rendering: it never
    calls `renderer.render()` more than once concurrently. Running
    several workers against the same RenderQueue (each with its own
    RenderWorker instance) is how this module expects concurrent
    rendering to be achieved — RenderQueue is already thread-safe for
    exactly that reason.
    """

    def __init__(
        self,
        queue: RenderQueue,
        renderer: Renderer,
        worker_id: str | None = None,
    ) -> None:
        """Initializes a worker.

        Args:
            queue: The RenderQueue to pull jobs from.
            renderer: Anything implementing `render(job) -> RenderResult`.
            worker_id: Optional explicit ID for this worker. If omitted,
                a short random ID is generated.
        """
        self.queue = queue
        self.renderer = renderer
        self.worker_id: str = worker_id or f"worker-{uuid.uuid4().hex[:8]}"

        self.status: WorkerStatus = WorkerStatus.OFFLINE
        self.current_job: RenderJob | None = None
        self.started_at: datetime | None = None
        self.jobs_completed: int = 0
        self.jobs_failed: int = 0
        self.last_activity: datetime | None = None

        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        """Starts the worker loop on a background thread.

        Returns immediately — the loop runs until `stop()` is called.
        Calling `start()` again while already running is a no-op.
        """
        if self._thread is not None and self._thread.is_alive():
            logger.warning(
                "Worker %s already running; start() is a no-op.", self.worker_id
            )
            return

        self._stop_event.clear()
        self.started_at = _utcnow()
        self.status = WorkerStatus.IDLE
        self._thread = threading.Thread(
            target=self._run_loop, name=f"render-worker-{self.worker_id}", daemon=True
        )
        self._thread.start()
        logger.info("Worker %s started.", self.worker_id)

    def stop(self) -> None:
        """Signals the worker loop to exit, then waits for it to do so.

        If a job is currently being rendered, that render is allowed
        to finish before the loop exits — this only stops the worker
        from picking up a *new* job afterward. Blocks until the
        background thread has fully exited.
        """
        logger.info(
            "Worker %s stopping (will finish any in-progress job first)...",
            self.worker_id,
        )
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join()
        self.status = WorkerStatus.OFFLINE
        logger.info("Worker %s stopped.", self.worker_id)

    def run_once(self) -> bool:
        """Processes a single job cycle synchronously, without starting
        the background loop. Useful for tests and other callers that
        want precise control over exactly one iteration.

        Returns:
            True if a job was fetched and processed (whether it
            succeeded or failed). False if the queue was empty, in
            which case this sleeps briefly before returning, matching
            the same empty-queue pause the background loop uses.
        """
        return self._process_next()

    def is_busy(self) -> bool:
        """Returns True if the worker is currently rendering a job."""
        return self.status == WorkerStatus.BUSY

    def health(self) -> WorkerInfo:
        """Returns a snapshot of this worker's current state as a
        WorkerInfo.

        `gpu_type` and `vram_gb` are populated with fixed sentinel
        values ("unknown" / 1) rather than real hardware data — this
        worker has no GPU knowledge by design (see module docstring),
        but WorkerInfo requires both fields to be present and
        `vram_gb` to be positive.
        """
        return WorkerInfo(
            worker_id=self.worker_id,
            gpu_type=_UNKNOWN_GPU_TYPE,
            vram_gb=_PLACEHOLDER_VRAM_GB,
            status=self.status,
            started_at=self.started_at or _utcnow(),
            current_job_id=self.current_job.job_id if self.current_job else None,
        )

    def _run_loop(self) -> None:
        """Background thread target: repeatedly processes jobs until
        `stop()` signals shutdown via `self._stop_event`."""
        logger.info("Worker %s loop running.", self.worker_id)
        while not self._stop_event.is_set():
            self._process_next()
        logger.info("Worker %s loop exiting.", self.worker_id)

    def _process_next(self) -> bool:
        """Runs exactly one fetch-render-record cycle.

        This is the single entry point both `run_once()` and the
        background loop call, so both paths get identical behavior —
        including the "never crash" guarantee: any exception anywhere
        in this method (not just from the renderer) is caught, logged,
        and treated as an empty cycle rather than allowed to escape.
        """
        try:
            return self._process_next_unsafe()
        except Exception:
            logger.exception(
                "Worker %s: unexpected error in processing loop.", self.worker_id
            )
            time.sleep(_POLL_INTERVAL_SECONDS)
            return False

    def _process_next_unsafe(self) -> bool:
        """The real logic behind `_process_next()`, without the outer
        safety net — see that method's docstring."""
        job = self.queue.next_job()
        if job is None:
            logger.info("Worker %s: queue empty.", self.worker_id)
            time.sleep(_POLL_INTERVAL_SECONDS)
            return False

        logger.info("Worker %s picked job %s.", self.worker_id, job.job_id)
        self.current_job = job
        self.status = WorkerStatus.BUSY
        self.last_activity = _utcnow()

        try:
            result = self.renderer.render(job)
        except Exception as exc:
            logger.exception("Worker %s: job %s failed.", self.worker_id, job.job_id)
            self.queue.mark_failed(job.job_id, str(exc))
            self.jobs_failed += 1
        else:
            self.queue.mark_completed(job.job_id, result)
            self.jobs_completed += 1
            logger.info("Worker %s: job %s completed.", self.worker_id, job.job_id)
        finally:
            self.current_job = None
            self.status = WorkerStatus.IDLE
            self.last_activity = _utcnow()

        return True


if __name__ == "__main__":
    from core.rendering.job_models import RenderPriority

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )

    class DummySuccessRenderer:
        """Test renderer that always succeeds after a short simulated
        render delay."""

        def render(self, job: RenderJob) -> RenderResult:
            time.sleep(1)
            return RenderResult(
                video_path=f"out/{job.job_id}.mp4",
                thumbnail_path=f"out/{job.job_id}.png",
                gif_preview_path=f"out/{job.job_id}.gif",
                render_time_seconds=1.0,
                workflow_version=job.workflow_version,
                seed=1,
            )

    class DummyFailingRenderer:
        """Test renderer that always raises, to exercise the worker's
        failure-handling path."""

        def render(self, job: RenderJob) -> RenderResult:
            time.sleep(1)
            raise RuntimeError(f"Simulated render failure for job {job.job_id}")

    def make_job(job_id: str) -> RenderJob:
        return RenderJob(
            job_id=job_id,
            user_id="demo_user",
            product="chess2fight",
            priority=RenderPriority.NORMAL,
            workflow_version="v1.0.0",
        )

    print("=== Submitting 3 jobs, starting a worker with a succeeding renderer ===")
    queue = RenderQueue()
    for job_id in ("job_1", "job_2", "job_3"):
        queue.submit(make_job(job_id))

    worker = RenderWorker(queue, DummySuccessRenderer())
    worker.start()

    time.sleep(4)  # 3 jobs x ~1s render time, plus margin

    worker.stop()

    print("\nworker.health():", worker.health().model_dump_json(indent=2))
    print("\nqueue.stats():", queue.stats())

    assert queue.stats()["completed"] == 3, "expected all 3 jobs to complete"
    assert worker.jobs_completed == 3
    assert worker.jobs_failed == 0
    print("\nOK: all 3 jobs completed, worker counters match.")

    print("\n=== Testing failure handling with run_once() ===")
    queue.submit(make_job("job_4"))
    failing_worker = RenderWorker(queue, DummyFailingRenderer())

    processed = failing_worker.run_once()
    assert processed is True

    job_4 = queue.get("job_4")
    assert job_4 is not None and job_4.status.value == "failed"
    assert queue.get_failure_reason("job_4") == "Simulated render failure for job job_4"
    print("OK: job_4 failed as expected:", queue.get_failure_reason("job_4"))
    print("queue.stats():", queue.stats())

    print("\n=== Testing retry: requeue job_4, process with a succeeding renderer ===")
    retried = queue.retry("job_4")
    assert retried is True
    assert queue.get("job_4").status.value == "queued"

    recovery_worker = RenderWorker(queue, DummySuccessRenderer())
    processed_again = recovery_worker.run_once()
    assert processed_again is True

    job_4_after_retry = queue.get("job_4")
    assert job_4_after_retry is not None
    assert job_4_after_retry.status.value == "completed"
    assert queue.get_retry_count("job_4") == 1
    retry_count = queue.get_retry_count("job_4")
    print("OK: job_4 completed after retry. retry_count:", retry_count)

    print("\n=== Final queue stats ===")
    print(queue.stats())
    assert queue.stats() == {
        "queued": 0, "running": 0, "completed": 4,
        "failed": 0, "cancelled": 0, "total": 4,
    }
    print("\nALL CHECKS PASSED")
