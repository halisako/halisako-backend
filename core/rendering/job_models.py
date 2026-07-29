"""Data models for the Halisako Rendering Infrastructure.

This module defines the strongly typed records that describe a render
job as it moves through the (not yet built) rendering pipeline: what
was asked for, what state it's in, and what came out. It intentionally
contains no queueing, scheduling, worker-management, or rendering
logic — those belong to future modules that will operate on these
models, not to the models themselves.

These types are shared platform infrastructure. They know nothing
about chess, screenplays, or any other product-specific concept —
`RenderJob.product` is a plain string precisely so this module never
needs to change when a new product (beyond Chess2Fight, Song2Dance)
starts using the renderer.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, model_validator

__all__ = [
    "RenderStatus",
    "RenderPriority",
    "WorkerStatus",
    "WorkerInfo",
    "SceneJob",
    "RenderResult",
    "RenderJob",
]


def _utcnow() -> datetime:
    """Returns the current time in UTC.

    Factored out so every auto-generated timestamp in this module is
    produced the same way, and so it's the one place a future test
    would patch to control time.
    """
    return datetime.now(timezone.utc)


class RenderStatus(str, Enum):
    """Lifecycle stage of a render job or an individual scene within one.

    Ordered roughly as a job is expected to progress through them,
    though this module makes no claim about which transitions are
    valid — that's the responsibility of whatever orchestrates the
    pipeline, not this data model.
    """

    NEW = "new"
    QUEUED = "queued"
    GPU_ASSIGNED = "gpu_assigned"
    PREPARING_ASSETS = "preparing_assets"
    RENDERING = "rendering"
    UPSCALING = "upscaling"
    INTERPOLATING = "interpolating"
    COMPOSING = "composing"
    UPLOADING = "uploading"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class RenderPriority(str, Enum):
    """Scheduling priority tier requested for a render job.

    What a given tier actually buys in queue position or GPU
    allocation is a scheduler concern, not something this model
    defines.
    """

    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    PREMIUM = "premium"


class WorkerStatus(str, Enum):
    """Coarse operational state of a rendering worker.

    A worker is a GPU host or process capable of executing render
    jobs. This is recorded here as a fact about the worker; deciding
    what to do with that fact (route work to it, drain it, alert on
    it) belongs to the future queue/worker service.
    """

    IDLE = "idle"
    BUSY = "busy"
    OFFLINE = "offline"
    DRAINING = "draining"


class WorkerInfo(BaseModel):
    """A snapshot of one rendering worker's identity and current state.

    This is a status record, not a live handle to the worker process —
    it describes what a worker last reported about itself, for
    whatever future service tracks and schedules against a fleet of
    workers.
    """

    model_config = ConfigDict(validate_assignment=True)

    worker_id: str = Field(
        ..., min_length=1, description="Unique identifier for this worker."
    )
    gpu_type: str = Field(
        ...,
        min_length=1,
        description='GPU model reported by the worker, e.g. "A100", "H100".',
    )
    vram_gb: int = Field(
        ..., gt=0, description="VRAM available on the worker's GPU, in gigabytes."
    )
    status: WorkerStatus = Field(
        ..., description="The worker's current operational state."
    )
    started_at: datetime = Field(
        ...,
        description=(
            "When the worker process started, in UTC. Provided by the "
            "caller — not auto-generated, since it describes an external "
            "event that may not coincide with when this record is "
            "constructed."
        ),
    )
    current_job_id: str | None = Field(
        default=None,
        description=(
            "ID of the RenderJob this worker is currently executing, if "
            "any. None when the worker is idle, offline, or draining."
        ),
    )


class SceneJob(BaseModel):
    """A single scene to be rendered — the smallest unit of rendering work.

    A RenderJob is composed of one or more of these. Fully specifies
    what should be generated (prompt, references, and a pinned seed)
    so that, given the same SceneJob, a render is reproducible. This
    module doesn't assign or validate the seed — it only requires that
    one is always present.
    """

    model_config = ConfigDict(validate_assignment=True)

    scene_id: str = Field(
        ..., min_length=1, description="Unique identifier for this scene."
    )
    scene_number: int = Field(
        ...,
        ge=1,
        description="1-indexed position of this scene within its parent RenderJob.",
    )
    workflow: str = Field(
        ...,
        min_length=1,
        description=(
            "Identifier of the render workflow/template this scene should "
            "use. An opaque string at this layer — this module has no "
            "knowledge of what a workflow contains or how it's executed."
        ),
    )
    duration_seconds: float = Field(
        ..., gt=0, description="Target duration of this scene, in seconds."
    )
    prompt: str = Field(
        ..., min_length=1, description="Positive generation prompt for this scene."
    )
    negative_prompt: str = Field(
        default="",
        description=(
            "Generation prompt describing what to avoid. Empty string "
            "when the scene has no specific exclusions."
        ),
    )
    references: list[str] = Field(
        default_factory=list,
        description="Paths to reference images for this scene, if any.",
    )
    seed: int = Field(
        ...,
        description=(
            "Pinned random seed for this scene's generation, for reproducibility."
        ),
    )
    status: RenderStatus = Field(
        default=RenderStatus.NEW, description="Current lifecycle stage of this scene."
    )


class RenderResult(BaseModel):
    """The output artifacts and provenance of a completed render.

    Every field is required: a RenderResult represents a render that
    finished successfully, and a partially-produced set of artifacts
    isn't a result yet — it's still an in-progress RenderJob with no
    result set.
    """

    model_config = ConfigDict(validate_assignment=True)

    video_path: str = Field(
        ..., min_length=1, description="Path to the final rendered video."
    )
    thumbnail_path: str = Field(
        ..., min_length=1, description="Path to the generated thumbnail image."
    )
    gif_preview_path: str = Field(
        ..., min_length=1, description="Path to the generated GIF preview."
    )
    render_time_seconds: float = Field(
        ...,
        ge=0,
        description=(
            "Wall-clock time the render took to produce these artifacts, in seconds."
        ),
    )
    workflow_version: str = Field(
        ...,
        min_length=1,
        description=(
            "Version of the render workflow that actually produced this "
            "result. May differ from RenderJob.workflow_version if the "
            "job was retried under a newer version after an earlier "
            "attempt failed."
        ),
    )
    seed: int = Field(
        ..., description="The random seed actually used to produce these artifacts."
    )


class RenderJob(BaseModel):
    """The primary rendering infrastructure object.

    One user-requested render, made up of one or more scenes, tracked
    from creation through to a final result. `progress` and `status`
    should be changed together, through `update_progress()`, rather
    than by assigning the attributes directly — see that method's
    docstring for why.
    """

    model_config = ConfigDict(validate_assignment=True)

    job_id: str = Field(
        ..., min_length=1, description="Unique identifier for this render job."
    )
    user_id: str = Field(
        ..., min_length=1, description="Identifier of the user who requested this job."
    )
    product: str = Field(
        ...,
        min_length=1,
        description=(
            'Which product this job was requested from, e.g. "chess2fight", '
            '"song2dance". A plain string, deliberately not an enum, so '
            "this shared infrastructure module never needs to change when "
            "a new product is added."
        ),
    )
    status: RenderStatus = Field(
        default=RenderStatus.NEW,
        description="Current lifecycle stage of the overall job.",
    )
    priority: RenderPriority = Field(
        ..., description="Scheduling priority requested for this job."
    )
    progress: int = Field(
        default=0,
        ge=0,
        le=100,
        description="Overall completion percentage of the job, 0-100.",
    )
    created_at: datetime = Field(
        default_factory=_utcnow,
        description="When this job record was created, in UTC. Set automatically.",
    )
    updated_at: datetime = Field(
        default_factory=_utcnow,
        description=(
            "When this job record was last modified, in UTC. Set "
            "automatically at creation, and refreshed by "
            "update_progress() on every subsequent change."
        ),
    )
    workflow_version: str = Field(
        ...,
        min_length=1,
        description="Version of the render workflow this job is pinned to.",
    )
    scene_jobs: list[SceneJob] = Field(
        default_factory=list,
        description="The individual scenes that make up this job, in order.",
    )
    result: RenderResult | None = Field(
        default=None,
        description=(
            "The job's output artifacts, once rendering has completed. "
            "None until then."
        ),
    )

    @model_validator(mode="before")
    @classmethod
    def _align_fresh_timestamps(cls, data: object) -> object:
        """Aligns `created_at`/`updated_at` for a freshly constructed job.

        If a caller constructs a RenderJob without providing either
        timestamp, `created_at` and `updated_at` should be exactly
        equal — this is a genuinely new record, not one that's been
        updated since creation. Each field's own `default_factory`
        would otherwise call `_utcnow()` independently and produce two
        timestamps a few microseconds apart.

        Has no effect when either timestamp is provided explicitly
        (e.g. reconstructing a RenderJob from persisted data, where
        they legitimately differ).
        """
        has_created = isinstance(data, dict) and "created_at" in data
        has_updated = isinstance(data, dict) and "updated_at" in data
        if isinstance(data, dict) and not has_created and not has_updated:
            timestamp = _utcnow()
            data = {**data, "created_at": timestamp, "updated_at": timestamp}
        return data

    def update_progress(
        self, progress: int, status: RenderStatus | None = None
    ) -> None:
        """Records new progress, refreshing `updated_at` in the same call.

        This is the intended way to mutate `progress` (and, when
        relevant, `status`) after construction: it guarantees
        `updated_at` is refreshed every time either one changes, so
        staleness-tracking can never silently fall out of sync with
        the fields it's meant to track. Callers that need to change
        `status` without a progress update (e.g. moving straight from
        QUEUED to FAILED) should still call this method, passing the
        job's current `progress` value unchanged.

        Args:
            progress: New completion percentage for the job, 0-100
                inclusive.
            status: New status to apply alongside the progress update.
                Left unchanged if omitted.

        Raises:
            ValueError: If `progress` is outside the 0-100 range.
        """
        if not 0 <= progress <= 100:
            raise ValueError(f"progress must be between 0 and 100, got {progress}")

        self.progress = progress
        if status is not None:
            self.status = status
        self.updated_at = _utcnow()