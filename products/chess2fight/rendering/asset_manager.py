"""AssetManager: owns the on-disk storage layout for rendered frames.

    storage/
        renders/
            {fight_id}/
                frame0001.png
                frame0002.png
                ...
                metadata.json

RenderPipeline (render_pipeline.py) generates frames by calling
ImageRouter — a cross-product, generic service that knows nothing
about "fights," "frames," or this storage layout. AssetManager is
where that separation is enforced: it copies whatever file
`ImageRouter` produced into this Chess2Fight-specific layout, and owns
the metadata format entirely. Nothing here calls ImageRouter or
generates an image itself — this module is pure storage/IO, no
generation logic.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from pydantic import BaseModel, Field

from core.config import settings


class FrameMetadata(BaseModel):
    """Everything recorded about one rendered frame.

    `generation_seed` records the actual seed the image provider
    reports using, when it reports one (via
    `ImageGenerationResult.metadata["seed"]` — a generic field any
    provider may populate; RenderPipeline reads it without importing
    anything provider-specific). `ComfyUIImageProvider` reports its
    real, actually-injected seed this way — including under Sprint 4
    Prompt 12's shared/derived visual-continuity seed policies, where
    it's no longer simply a hash of the prompt. For a provider that
    doesn't report a seed at all (e.g. `MockImageProvider`, whose
    metadata carries no "seed" key), this falls back to a deterministic
    value derived from the frame's prompt (`render_pipeline.py`'s
    `_derive_seed`) — recorded for traceability/reproducibility even
    without a real provider-reported value. An earlier version of this
    docstring said this value was always independently computed rather
    than threaded into generation, on the reasoning that the interface
    didn't support it — true before Prompt 12's seed_override existed,
    no longer the accurate description for ComfyUIImageProvider.
    """

    frame_number: int = Field(..., ge=1, description="1-indexed frame number within the fight.")
    prompt: str = Field(..., min_length=1, description="The image prompt used to generate this frame.")
    camera_angle: str = Field(..., min_length=1, description="Camera framing for this frame's shot.")
    camera_motion: str = Field(..., min_length=1, description="Camera movement for this frame's shot.")
    shot_id: str = Field(..., min_length=1, description="ID of the Shot this frame renders.")
    shot_type: str = Field(..., min_length=1, description="Narrative role of this frame's shot.")
    source_moves: list[str] = Field(
        default_factory=list, description="Chess move labels this frame dramatizes, if any."
    )
    timestamp: str = Field(..., min_length=1, description="ISO 8601 UTC timestamp of when this frame was rendered.")
    generation_seed: int = Field(
        ..., description="The actual seed the image provider reported using, when available; otherwise a "
        "deterministic value derived from this frame's prompt."
    )


class RenderManifest(BaseModel):
    """The complete metadata.json contents for one fight's render."""

    fight_id: str = Field(..., min_length=1, description="Which fight this render belongs to.")
    frame_count: int = Field(..., ge=1, description="Number of frames in this render — len(frames).")
    frames: list[FrameMetadata] = Field(..., min_length=1, description="Every frame's metadata, in frame_number order.")


class AssetManager:
    """Owns the `storage/renders/{fight_id}/` layout: frame file
    naming, copying generated images into place, and reading/writing
    `metadata.json`.
    """

    def __init__(self, storage_root: str | None = None) -> None:
        """Initializes the AssetManager.

        Args:
            storage_root: Root directory for all rendered output.
                Defaults to `settings.render_storage_root`.
        """
        self._storage_root = Path(storage_root if storage_root is not None else settings.render_storage_root)

    @property
    def storage_root(self) -> Path:
        """Sprint 4 Prompt 14 — the configured root directory, exposed
        publicly so a caller can derive paths relative to it (e.g. an
        isolated subdirectory outside `fight_directory()`'s own
        `renders/` convention) without reaching into a private
        attribute. Purely additive — no existing behavior changed."""
        return self._storage_root

    def fight_directory(self, fight_id: str) -> Path:
        """Returns (creating if needed) the storage directory for one
        fight's rendered frames: `{storage_root}/renders/{fight_id}/`."""
        path = self._storage_root / "renders" / fight_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    def frame_filename(self, frame_number: int) -> str:
        """Returns the standard filename for a given frame number,
        e.g. `frame0001.png` for frame_number=1."""
        return f"frame{frame_number:04d}.png"

    def frame_path(self, fight_id: str, frame_number: int) -> Path:
        """Returns the full path a given frame would be stored at,
        without creating anything."""
        return self._storage_root / "renders" / fight_id / self.frame_filename(frame_number)

    def save_frame(self, fight_id: str, frame_number: int, source_image_path: str) -> Path:
        """Copies a generated image into this fight's storage
        directory under the correct frame filename.

        Args:
            fight_id: Which fight this frame belongs to.
            frame_number: 1-indexed frame number.
            source_image_path: Path to the image ImageRouter already
                generated (e.g. under `settings.image_output_dir`) —
                this is copied, not moved, so the original generated
                file is left intact for the image provider's own use
                (e.g. its own dedup/caching by content hash).

        Returns:
            The path the frame was saved to within this fight's
            storage directory.
        """
        destination = self.fight_directory(fight_id) / self.frame_filename(frame_number)
        shutil.copyfile(source_image_path, destination)
        return destination

    def manifest_path(self, fight_id: str) -> Path:
        """Returns the path to a fight's metadata.json, without
        creating anything."""
        return self._storage_root / "renders" / fight_id / "metadata.json"

    def write_manifest(self, fight_id: str, manifest: RenderManifest) -> Path:
        """Writes a fight's complete metadata.json."""
        path = self.fight_directory(fight_id) / "metadata.json"
        path.write_text(manifest.model_dump_json(indent=2))
        return path

    def read_manifest(self, fight_id: str) -> RenderManifest:
        """Reads back a previously-written metadata.json.

        Raises:
            FileNotFoundError: If no manifest has been written for
                this fight_id yet.
        """
        return RenderManifest.model_validate_json(self.manifest_path(fight_id).read_text())
