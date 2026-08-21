"""ComfyUIAnimationProvider: a real image-to-video AnimationProvider
that communicates with a ComfyUI server's HTTP API to run the
experimentally-validated Wan 2.2 TI2V-5B workflow at
products/chess2fight/rendering/workflows/wan22_i2v_5b.json.

    AnimationInstruction -> ComfyUIAnimationProvider -> ComfyUI HTTP API -> AnimationResult

SPRINT 4 PROMPT 4 STATUS UPDATE — read this before trusting a claim
below: Prompt 3 built this class against no real workflow at all
(every node title was invented). Prompt 4 replaced every placeholder
with the *actual* node IDs from a real, experimentally-validated
workflow JSON — supplied, not fabricated — that was manually run on an
RTX 4090 (CUDA 12.8, Wan 2.2 TI2V-5B) and produced a genuine 640x352,
24fps, 49-frame (2.04s) MP4 with five distinct sampled-frame hashes,
confirming real generated motion. That resolves the *workflow
structure* question. It does NOT resolve whether this Python code
correctly drives that workflow end-to-end over HTTP — no ComfyUI
server exists in the environment this code was written in (see this
feature's engineering report for the environment audit, unchanged
since Prompt 3). Three confidence tiers apply here, and they're kept
distinct throughout:

1. ComfyUI's own HTTP API shape (`POST /upload/image`, `POST /prompt`,
   `GET /history/{prompt_id}`, `GET /view`) — stable, documented
   ComfyUI infrastructure. Confident.
2. This specific workflow's node IDs, input keys, and model filenames
   — read directly from the supplied wan22_i2v_5b.json, not guessed.
   Confident, for exactly what the file contains.
3. Wan22ImageToVideoLatent's *constraints* (frame-count alignment,
   dimension alignment) and which key SaveVideo's history output
   appears under — researched (web search against ComfyUI's public
   source/docs/community reports), not verified against a live
   instance. Each such claim is marked with its actual sourcing below,
   not overstated as "verified."

Until `COMFYUI_LIVE_TEST=1` succeeds against a real server, treat this
provider as "matches the validated workflow structure, HTTP logic
unverified" — a meaningfully stronger claim than Prompt 3's "entirely
unverified," but still short of "known to work."
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import subprocess
import time
import uuid
from pathlib import Path
from typing import Any

import httpx

from core.animation_router import AnimationInstruction, AnimationProvider, AnimationResult
from core.config import get_settings

settings = get_settings()
logger = logging.getLogger(__name__)


# --- Node IDs from the validated wan22_i2v_5b.json workflow -----------------
#
# Read directly from the supplied JSON, not inferred or guessed. A
# numeric ID is used directly (not a title lookup, as Prompt 3 used)
# because this is now a specific, versioned workflow artifact — see
# products/chess2fight/rendering/workflows/README.md. If node IDs
# aren't found at these keys, injection fails loudly (KeyError,
# caught and reported) rather than silently skipping, since — unlike
# Prompt 3's "maybe this generic template has this node" situation —
# these IDs are now expected to exist; their absence means the
# workflow file on disk doesn't match this provider's understanding
# of it, which is worth failing loudly over.
_NODE_LOAD_IMAGE = "56"
_NODE_POSITIVE_PROMPT = "6"
_NODE_NEGATIVE_PROMPT = "7"
_NODE_SAMPLER = "3"
_NODE_IMAGE_TO_VIDEO_LATENT = "55"
_NODE_CREATE_VIDEO = "57"
_NODE_SAVE_VIDEO = "58"

# The validated workflow's own proof-quality resolution — used as the
# default when AnimationInstruction doesn't specify one (matches
# AnimationInstruction's own schema default of 1024x1024 being
# overridden here, since 640x352 is what's actually been proven to
# work with this exact workflow, not a generic guess).
_DEFAULT_WIDTH = 640
_DEFAULT_HEIGHT = 352

# Wan22ImageToVideoLatent's dimension-alignment requirement: width and
# height must be a multiple of 16. SOURCED: docs.comfy.org's own
# Wan22ImageToVideoLatent reference page states this explicitly for
# this exact node ("The width and height parameters must be divisible
# by 16 for proper latent space dimensions") — confirmed, not
# inferred. 640 % 16 == 0 and 352 % 16 == 0, consistent with the
# validated workflow's own values.
_DIMENSION_ALIGNMENT = 16

# Wan22ImageToVideoLatent's frame-count alignment: `length` must
# satisfy (length - 1) % 4 == 0 (i.e. 1, 5, 9, ..., 49, ...).
# SOURCED WITH LESS CERTAINTY than the dimension rule above: this is
# not confirmed from Wan22ImageToVideoLatent's own source directly
# (unavailable to inspect in this environment). It's inferred from
# converging evidence: (a) the closely-related WanImageToVideo node's
# actual INPUT_TYPES in ComfyUI's source defines `length` with
# `"step": 4, "min": 1`; (b) multiple independent ComfyUI-Wiki/tutorial
# sources for Wan22ImageToVideoLatent specifically cite default/
# recommended lengths of 41, 81, and 121 frames — all satisfying this
# same pattern; (c) the validated workflow's own 49 frames also
# satisfies it (49 = 4*12 + 1). Three independent, converging sources
# is strong circumstantial evidence, not a single authoritative
# confirmation for this exact node — treat accordingly.
_LENGTH_ALIGNMENT = 4
_LENGTH_MINIMUM = 1

# ComfyUI's native SaveVideo node (comfy_extras/nodes_video.py — NOT
# the older VHS_VideoCombine or N-Nodes/JNodes custom nodes of the
# same name, confirmed by this workflow's node schema: video/
# filename_prefix/format/codec, matching SaveVideo's documented
# fields) reports its output under an as-yet-unconfirmed key in
# /history's response. UNSOURCED beyond reasonable candidates: video
# generation nodes have historically reported under "images", "gifs",
# or "videos" depending on node generation and object-type conventions
# — all three are checked. Genuinely unresolved: a ComfyUI forum post
# (forum.comfy.org/t/comfy-ui-api-automation/4251, Feb 2026, unanswered
# at time of writing) reports SaveVideo output sometimes not appearing
# in /history at all via polling, even though the file was generated
# successfully — a real, documented, unresolved community-reported
# risk this provider's polling approach cannot fully rule out without
# a live instance to test against. See this feature's engineering
# report, "Remaining blockers."
_VIDEO_OUTPUT_KEYS = ("images", "videos", "gifs")

_POLL_INTERVAL_SECONDS = 2.0

# Cap on raw ComfyUI payload text embedded in a returned error message
# — the task explicitly warns against leaking huge raw payloads into
# user-facing errors. Full, untruncated detail is still available via
# the logger.exception() call in generate_animation's outer handler
# and via ComfyUI's own server-side logs — this cap only bounds what
# AnimationResult.error_message itself carries.
_ERROR_MESSAGE_PAYLOAD_LIMIT = 500


def _truncate(text: str, limit: int = _ERROR_MESSAGE_PAYLOAD_LIMIT) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"... [truncated, {len(text)} chars total]"


class ComfyUIRequestError(Exception):
    """Internal to this module — never propagates past
    `generate_animation`, which catches it and returns a failed
    `AnimationResult` instead, matching `MockAnimationProvider`'s own
    established pattern of representing generation failure as data."""


def _normalize_dimension(value: int, alignment: int = _DIMENSION_ALIGNMENT) -> int:
    """Rounds a width/height to the nearest multiple of `alignment`
    (minimum one full `alignment` unit), per
    Wan22ImageToVideoLatent's documented divisible-by-16 requirement.
    "Round to nearest" (not down or up) was chosen as the least
    arbitrary of the undocumented options — the task doesn't specify
    a direction, and rounding to nearest minimizes distortion from
    what was actually requested.
    """
    normalized = round(value / alignment) * alignment
    return max(alignment, normalized)


def _duration_to_frame_count(duration_seconds: float, fps: int) -> int:
    """Converts a requested duration into a Wan-valid frame count,
    snapping to the nearest value satisfying (length - 1) % 4 == 0 —
    see `_LENGTH_ALIGNMENT`'s docstring above for exactly how
    confident that constraint is. Verified against the one known-good
    data point available: 2.0s at 24fps produces 49 frames here,
    exactly matching the validated workflow's own proven value.
    """
    raw_frame_count = max(_LENGTH_MINIMUM, round(duration_seconds * fps))
    steps = round((raw_frame_count - _LENGTH_MINIMUM) / _LENGTH_ALIGNMENT)
    return max(_LENGTH_MINIMUM, steps * _LENGTH_ALIGNMENT + _LENGTH_MINIMUM)


def _derive_seed(prompt: str) -> int:
    """A deterministic fallback seed derived from the prompt, used only
    when `AnimationInstruction.seed` is unset — keeps generation
    reproducible by default rather than silently random, matching the
    same reasoning `products/chess2fight/rendering/render_pipeline.py`'s
    own `_derive_seed` already applies elsewhere in this codebase (not
    imported directly — see this file's module docstring on `core/`
    not depending on `products/chess2fight/`)."""
    return int(hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:8], 16)


class ComfyUIAnimationProvider(AnimationProvider):
    """Communicates with a real ComfyUI server to run image-to-video
    generation via the validated Wan 2.2 TI2V-5B workflow.

    Self-contained like every `AnimationProvider`: no knowledge of
    `AnimationRouter`, `AnimationProviderRegistry`, or any other
    provider. Reads its configuration from the existing settings
    system (`settings.comfyui_base_url`, `.comfyui_timeout_seconds`,
    `.comfyui_workflow_path`, `.comfyui_default_fps`).

    Never assumes a shared filesystem with the ComfyUI server: the
    reference image is uploaded over HTTP (`/upload/image`) and the
    generated video is downloaded over HTTP (`/view`) — this provider
    works identically whether ComfyUI is on the same machine or a
    remote GPU worker.
    """

    def __init__(
        self,
        base_url: str | None = None,
        workflow_path: str | None = None,
        timeout_seconds: float | None = None,
        output_dir: str | None = None,
        default_fps: int | None = None,
    ) -> None:
        """Initializes the provider.

        Args:
            base_url: ComfyUI server URL. Defaults to `settings.comfyui_base_url`.
            workflow_path: Path to the ComfyUI API-format workflow JSON.
                Defaults to `settings.comfyui_workflow_path`.
            timeout_seconds: Maximum time to wait for one generation to
                complete. Defaults to `settings.comfyui_timeout_seconds`.
            output_dir: Where to save downloaded clips. Defaults to
                `settings.animation_output_dir` (shared with
                MockAnimationProvider — both providers' outputs live in
                the same configured location).
            default_fps: Used to convert `duration_seconds` into a
                frame count when `instruction.fps` is unset. Defaults
                to `settings.comfyui_default_fps`, now 24 — verified
                directly from the validated workflow's own node 57
                (CreateVideo) `fps` input, not a placeholder.
        """
        self._base_url = (base_url if base_url is not None else settings.comfyui_base_url).rstrip("/")
        self._workflow_path = workflow_path if workflow_path is not None else settings.comfyui_workflow_path
        self._timeout_seconds = (
            timeout_seconds if timeout_seconds is not None else settings.comfyui_timeout_seconds
        )
        self._default_fps = default_fps if default_fps is not None else settings.comfyui_default_fps
        self._output_dir = Path(output_dir if output_dir is not None else settings.animation_output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)

    async def generate_animation(self, instruction: AnimationInstruction) -> AnimationResult:
        try:
            workflow = self._load_workflow()
        except FileNotFoundError as exc:
            return self._failure(instruction, str(exc))

        if not Path(instruction.source_image_path).exists():
            return self._failure(
                instruction, f"Reference image not found: {instruction.source_image_path!r}."
            )

        prompt_id = None
        try:
            async with httpx.AsyncClient(timeout=self._timeout_seconds) as client:
                try:
                    uploaded_name = await self._upload_image(client, instruction.source_image_path)
                except httpx.HTTPError as exc:
                    return self._failure(instruction, f"Image upload to ComfyUI failed: {exc}")

                prepared_workflow = self._inject_parameters(workflow, instruction, uploaded_name)

                try:
                    prompt_id = await self._queue_prompt(client, prepared_workflow)
                except httpx.HTTPError as exc:
                    return self._failure(instruction, f"Workflow submission to ComfyUI failed: {exc}")
                except ComfyUIRequestError as exc:
                    return self._failure(instruction, f"ComfyUI rejected the workflow: {exc}")
                logger.info("ComfyUI: queued prompt_id=%s for shot %s.", prompt_id, instruction.shot_id)

                try:
                    history_entry = await self._wait_for_completion(client, prompt_id)
                except TimeoutError as exc:
                    return self._failure(instruction, f"Execution timeout: {exc}")
                except ComfyUIRequestError as exc:
                    return self._failure(instruction, f"Generation failed: {exc}")
                except httpx.HTTPError as exc:
                    return self._failure(instruction, f"Polling ComfyUI for completion failed: {exc}")

                try:
                    filename, subfolder, file_type = self._extract_output_reference(history_entry)
                except ComfyUIRequestError as exc:
                    return self._failure(instruction, f"History/output missing: {exc}")

                try:
                    video_bytes = await self._download_output(client, filename, subfolder, file_type)
                except httpx.HTTPError as exc:
                    return self._failure(instruction, f"Video download failed: {exc}")
        except httpx.HTTPError as exc:
            return self._failure(instruction, f"Could not reach ComfyUI at {self._base_url!r}: {exc}")
        except Exception as exc:  # noqa: BLE001 — any other failure must still become a result, never a crash
            logger.exception("Unexpected ComfyUIAnimationProvider failure for shot %s.", instruction.shot_id)
            return self._failure(instruction, f"Unexpected provider error: {exc}")

        # Unique per (shot_id, prompt_id) — avoids collisions between
        # concurrent or repeated jobs for the same shot (e.g. a retry).
        output_path = self._output_dir / f"comfyui_{instruction.shot_id}_{prompt_id}.mp4"
        output_path.write_bytes(video_bytes)

        try:
            probed = self._verify_video(output_path)
        except ComfyUIRequestError as exc:
            return self._failure(instruction, f"Invalid output video: {exc}")

        return AnimationResult(
            success=True,
            shot_id=instruction.shot_id,
            provider="ComfyUIAnimationProvider",
            video_path=str(output_path),
            duration_seconds=probed["duration_seconds"],
            width=probed["width"],
            height=probed["height"],
            fps=instruction.fps or self._default_fps,
            metadata={
                "prompt_id": prompt_id,
                "comfyui_base_url": self._base_url,
                "model": "wan2.2_ti2v_5B_fp16.safetensors",
                "vae": "wan2.2_vae.safetensors",
                "text_encoder": "umt5_xxl_fp8_e4m3fn_scaled.safetensors",
            },
        )

    def _failure(self, instruction: AnimationInstruction, message: str) -> AnimationResult:
        return AnimationResult(
            success=False, shot_id=instruction.shot_id, provider="ComfyUIAnimationProvider", error_message=message
        )

    # --- Workflow loading and parameter injection ---------------------------

    def _load_workflow(self) -> dict[str, Any]:
        """Loads the ComfyUI API-format workflow JSON from disk.

        Raises:
            FileNotFoundError: If no workflow file exists at the
                configured path.
        """
        path = Path(self._workflow_path)
        if not path.exists():
            raise FileNotFoundError(
                f"ComfyUI workflow file not found at {path!r}. Expected the validated "
                "wan22_i2v_5b.json — see products/chess2fight/rendering/workflows/README.md."
            )
        # Explicit encoding, not Path.read_text()'s platform default: this
        # workflow's negative prompt (node 7) is non-ASCII Chinese text —
        # Windows' default locale encoding (cp1252) cannot decode it and
        # would raise or silently corrupt it. Sprint 4 Prompt 7.1.
        return json.loads(path.read_text(encoding="utf-8"))

    def _inject_parameters(
        self, workflow: dict[str, Any], instruction: AnimationInstruction, uploaded_image_name: str
    ) -> dict[str, Any]:
        """Injects AnimationInstruction values into the loaded workflow
        graph, at the exact node IDs read from the validated
        wan22_i2v_5b.json — see the module-level `_NODE_*` constants.

        The negative prompt is deliberately NOT overwritten when
        `instruction.negative_prompt` is unset — the validated workflow
        ships with a real, substantial, tuned negative prompt (node 7);
        Prompt 3's version of this method unconditionally overwrote it
        with an empty string whenever the caller didn't supply one,
        which would have silently discarded that tuned content. This
        is a correction, not a stylistic change.

        `fps` is written to node 57 (CreateVideo), not just used to
        compute `frame_count` — Prompt 4/5's version computed fps but
        never actually wrote it to the node, which happened not to
        matter while every caller used the default (which already
        matches the workflow's own baked-in 24), but would silently
        mismatch frame_count against playback rate for any other fps.
        Fixed here, not just noted, per this task's "verify the
        current Wan mappings" instruction.
        """
        fps = instruction.fps or self._default_fps
        frame_count = _duration_to_frame_count(instruction.duration_seconds, fps)
        seed = instruction.seed if instruction.seed is not None else _derive_seed(instruction.prompt)
        width = _normalize_dimension(instruction.width)
        height = _normalize_dimension(instruction.height)

        prepared = json.loads(json.dumps(workflow))  # plain-JSON deep copy

        prepared[_NODE_LOAD_IMAGE]["inputs"]["image"] = uploaded_image_name
        prepared[_NODE_POSITIVE_PROMPT]["inputs"]["text"] = instruction.prompt
        if instruction.negative_prompt:
            prepared[_NODE_NEGATIVE_PROMPT]["inputs"]["text"] = instruction.negative_prompt
        prepared[_NODE_SAMPLER]["inputs"]["seed"] = seed
        prepared[_NODE_IMAGE_TO_VIDEO_LATENT]["inputs"]["width"] = width
        prepared[_NODE_IMAGE_TO_VIDEO_LATENT]["inputs"]["height"] = height
        prepared[_NODE_IMAGE_TO_VIDEO_LATENT]["inputs"]["length"] = frame_count
        prepared[_NODE_CREATE_VIDEO]["inputs"]["fps"] = fps

        return prepared

    # --- ComfyUI HTTP API calls -----------------------------------------------

    async def _upload_image(self, client: httpx.AsyncClient, image_path: str) -> str:
        image_bytes = Path(image_path).read_bytes()
        response = await client.post(
            f"{self._base_url}/upload/image",
            files={"image": (Path(image_path).name, image_bytes, "image/png")},
        )
        response.raise_for_status()
        return response.json()["name"]

    async def _queue_prompt(self, client: httpx.AsyncClient, workflow: dict[str, Any]) -> str:
        response = await client.post(
            f"{self._base_url}/prompt",
            json={"prompt": workflow, "client_id": uuid.uuid4().hex},
        )
        response.raise_for_status()
        data = response.json()
        if data.get("node_errors"):
            raise ComfyUIRequestError(_truncate(str(data["node_errors"])))
        return data["prompt_id"]

    async def _wait_for_completion(self, client: httpx.AsyncClient, prompt_id: str) -> dict[str, Any]:
        deadline = time.monotonic() + self._timeout_seconds
        while time.monotonic() < deadline:
            response = await client.get(f"{self._base_url}/history/{prompt_id}")
            response.raise_for_status()
            history = response.json()
            if prompt_id in history:
                entry = history[prompt_id]
                status = entry.get("status", {})
                if status.get("status_str") == "error":
                    raise ComfyUIRequestError(_truncate(str(status)))
                return entry
            await asyncio.sleep(_POLL_INTERVAL_SECONDS)
        raise TimeoutError(f"did not complete within {self._timeout_seconds}s (prompt_id={prompt_id}).")

    def _extract_output_reference(self, history_entry: dict[str, Any]) -> tuple[str, str, str]:
        """Resolves the generated video's filename/subfolder/type from
        ComfyUI's own history response — never guesses a filename (see
        `_VIDEO_OUTPUT_KEYS`'s module-level docstring for exactly how
        confident this key-matching is, including a known,
        unresolved community-reported risk with this exact node)."""
        outputs = history_entry.get("outputs", {})

        node_58_output = outputs.get(_NODE_SAVE_VIDEO, {})
        for key in _VIDEO_OUTPUT_KEYS:
            if key in node_58_output and node_58_output[key]:
                entry = node_58_output[key][0]
                return entry["filename"], entry.get("subfolder", ""), entry.get("type", "output")

        # Fall back to scanning every node's output, in case the
        # reporting node ID doesn't match _NODE_SAVE_VIDEO at runtime
        # for some reason (defensive, not expected).
        for node_output in outputs.values():
            for key in _VIDEO_OUTPUT_KEYS:
                if key in node_output and node_output[key]:
                    entry = node_output[key][0]
                    return entry["filename"], entry.get("subfolder", ""), entry.get("type", "output")

        raise ComfyUIRequestError(
            f"no video output found under node {_NODE_SAVE_VIDEO!r} or any other node, "
            f"checked keys {_VIDEO_OUTPUT_KEYS!r}."
        )

    async def _download_output(
        self, client: httpx.AsyncClient, filename: str, subfolder: str, file_type: str
    ) -> bytes:
        response = await client.get(
            f"{self._base_url}/view", params={"filename": filename, "subfolder": subfolder, "type": file_type}
        )
        response.raise_for_status()
        return response.content

    # --- Output verification -----------------------------------------------------

    def _verify_video(self, path: Path) -> dict[str, float]:
        """Validates the downloaded file is a real, playable video with
        non-zero duration and valid dimensions — a successful HTTP
        response is not sufficient on its own. Shells out to `ffprobe`
        directly (the same tool, invoked the same way, that
        VideoBuilder's own test suite already uses for verification —
        VideoBuilder itself has no reusable ffprobe-wrapping method to
        call instead, so this is a small, focused addition here, not a
        second abstraction competing with an existing one).

        Raises:
            ComfyUIRequestError: If the file is missing, empty, or
                ffprobe reports invalid duration/dimensions.
        """
        if not path.exists() or path.stat().st_size == 0:
            raise ComfyUIRequestError(f"downloaded file is missing or empty: {path}")

        try:
            probe = subprocess.run(
                [
                    "ffprobe", "-v", "error", "-print_format", "json",
                    "-show_format", "-show_streams", str(path),
                ],
                capture_output=True, text=True, timeout=30.0,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise ComfyUIRequestError(f"could not run ffprobe to verify output: {exc}") from exc

        if probe.returncode != 0:
            raise ComfyUIRequestError(f"ffprobe could not read downloaded file: {probe.stderr[-500:]}")

        try:
            data = json.loads(probe.stdout)
            duration = float(data["format"]["duration"])
            video_stream = next(s for s in data["streams"] if s["codec_type"] == "video")
            width, height = int(video_stream["width"]), int(video_stream["height"])
        except (KeyError, ValueError, StopIteration, json.JSONDecodeError) as exc:
            raise ComfyUIRequestError(f"downloaded file has no valid video stream: {exc}") from exc

        if not (duration > 0 and width > 0 and height > 0):
            raise ComfyUIRequestError(
                f"downloaded video has invalid properties: duration={duration}, width={width}, height={height}"
            )

        return {"duration_seconds": duration, "width": width, "height": height}
