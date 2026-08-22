"""ComfyUIAnimationProvider: a real AnimationProvider that communicates
with a ComfyUI server's HTTP API to run the experimentally-validated
Wan 2.2 TI2V-5B workflow — in either image-to-video (I2V, primary) or
text-to-video (T2V, secondary) mode.

    AnimationInstruction -> ComfyUIAnimationProvider -> ComfyUI HTTP API -> AnimationResult

SPRINT 4 PROMPT 8 STATUS UPDATE — read this before trusting a claim
below:

Prompt 4 established I2V-only support against one live-validated
workflow (640x352, 24fps, 49 frames). This task supplied a *newer*
live validation — a RunPod RTX 4090 run producing both a real T2V and
a real I2V clip — with different settings (832x480, 8fps, 17 frames,
8 steps). Both are genuine, live-validated data points from different
sessions; this backend now uses the more recent one as its default,
per that task's own instruction to preserve it "unless there is a
strong reason not to" — it is a supersession, not a guess reconciling
two conflicting claims.

A discrepancy worth recording plainly: the task supplying these
artifacts labeled one file "the I2V API-format workflow... the most
important artifact," and the other as merely useful for "understanding
the working node structure" of "the small Wan 2.2 test." Inspecting
the actual JSON structure (not the labels) showed the reverse: the
file called "the I2V workflow" has no `LoadImage` node and no
`start_image` wiring on `Wan22ImageToVideoLatent` at all — it is
structurally text-to-video. The file described as merely a small test
is the one with a real, wired `LoadImage` -> `start_image` connection
— structurally image-to-video. Both files share identical settings and
model filenames otherwise, consistent with being genuine T2V/I2V
variants of the same validated session, not unrelated or broken
artifacts. This module treats the verified graph structure as ground
truth over the accompanying labels: `wan22_i2v_5b.json` is built from
the file that actually has the image-conditioning wiring;
`wan22_t2v_5b.json` from the one that doesn't.

Three confidence tiers still apply, as in Prompt 4:

1. ComfyUI's own HTTP API shape (`POST /upload/image`, `POST /prompt`,
   `GET /history/{prompt_id}`, `GET /view`) — stable, documented
   ComfyUI infrastructure. Confident.
2. Both workflows' node IDs, input keys, and model filenames — read
   directly from the supplied JSON files, not guessed. Confident, for
   exactly what those files contain (see the structural note above for
   how that confidence was actually established, not just assumed from
   a filename).
3. Wan22ImageToVideoLatent's *constraints* (frame-count alignment,
   dimension alignment) and which key SaveVideo's history output
   appears under — researched (Prompt 4), not verified against a live
   instance. Unchanged by this task.

No ComfyUI server exists in the environment this code was written in
(see this feature's engineering report for the environment audit,
unchanged since Prompt 3). Until `COMFYUI_LIVE_TEST=1` succeeds against
a real server, treat this provider as "matches validated workflow
structure for both modes, HTTP logic unverified in this environment."
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

from core.animation_router import AnimationInstruction, AnimationProvider, AnimationResult, AnimationType
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

# T2V's node IDs happen to be identical to I2V's for everything except
# the image — both were exported from the same underlying graph
# structure with the image-conditioning branch removed for T2V (see
# module docstring). Kept as separate names, not aliased to the I2V
# constants above, so a future edit to one workflow's node numbering
# doesn't silently affect the other just because today's values match.
_T2V_NODE_POSITIVE_PROMPT = "6"
_T2V_NODE_NEGATIVE_PROMPT = "7"
_T2V_NODE_SAMPLER = "3"
_T2V_NODE_VIDEO_LATENT = "55"  # no start_image input on this node in the T2V workflow — never set
_T2V_NODE_CREATE_VIDEO = "57"
_T2V_NODE_SAVE_VIDEO = "58"

# The validated workflow's own proof-quality resolution — documented
# here as the reference values a caller should pass to match what's
# actually been proven to work with this exact workflow. NOT an active
# fallback inside _inject_i2v_parameters/_inject_t2v_parameters, which
# always use whatever instruction.width/.height the caller actually
# supplied (defaulting to AnimationInstruction's own generic 1024x1024
# schema default if the caller passed nothing) — deliberately, so this
# provider never silently substitutes a different resolution than what
# was asked for, matching the same reasoning already applied to
# ComfyUIImageProvider's own resolution handling (Prompt 6). A caller
# that wants the validated resolution should pass width=832, height=480
# explicitly. Sprint 4 Prompt 8's newer live validation superseded
# Prompt 4's 640x352 as the recorded reference values.
_DEFAULT_WIDTH = 832
_DEFAULT_HEIGHT = 480

# Wan22ImageToVideoLatent's dimension-alignment requirement: width and
# height must be a multiple of 16. SOURCED: docs.comfy.org's own
# Wan22ImageToVideoLatent reference page states this explicitly for
# this exact node ("The width and height parameters must be divisible
# by 16 for proper latent space dimensions") — confirmed, not
# inferred. 832 % 16 == 0 and 480 % 16 == 0, consistent with the
# validated workflow's own values (and with Prompt 4's earlier
# 640x352, which also satisfied this).
_DIMENSION_ALIGNMENT = 16

# Wan22ImageToVideoLatent's frame-count alignment: `length` must
# satisfy (length - 1) % 4 == 0 (i.e. 1, 5, 9, ..., 17, ..., 49, ...).
# SOURCED WITH LESS CERTAINTY than the dimension rule above: this is
# not confirmed from Wan22ImageToVideoLatent's own source directly
# (unavailable to inspect in this environment). It's inferred from
# converging evidence: (a) the closely-related WanImageToVideo node's
# actual INPUT_TYPES in ComfyUI's source defines `length` with
# `"step": 4, "min": 1`; (b) multiple independent ComfyUI-Wiki/tutorial
# sources for Wan22ImageToVideoLatent specifically cite default/
# recommended lengths of 41, 81, and 121 frames — all satisfying this
# same pattern; (c) Prompt 4's validated workflow used 49 frames
# (= 4*12 + 1); (d) Prompt 8's newer validated workflow uses 17 frames
# (= 4*4 + 1) — a second, independent live data point satisfying the
# same rule. Four independent, converging sources is strong
# circumstantial evidence, not a single authoritative confirmation for
# this exact node — treat accordingly.
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
    """Converts a requested duration into a Wan-valid frame count.

    The rounding rule, explicitly, in three steps:

    1. Compute the raw frame count a naive `duration * fps` would give,
       rounded to the nearest whole frame (never below
       `_LENGTH_MINIMUM`, currently 1).
    2. Wan's `Wan22ImageToVideoLatent.length` must satisfy
       `(length - 1) % _LENGTH_ALIGNMENT == 0` (i.e. 1, 5, 9, ..., 17,
       ..., 49, ...) — see `_LENGTH_ALIGNMENT`'s own module-level
       docstring for exactly how confident that constraint is and
       what it's sourced from. Step 1's raw count is snapped to the
       *nearest* such value (not floored or ceiled) — `round()` on
       `(raw_frame_count - _LENGTH_MINIMUM) / _LENGTH_ALIGNMENT`, i.e.
       the nearest whole number of alignment steps above the minimum.
    3. The snapped value is floored at `_LENGTH_MINIMUM` again, in
       case step 2's rounding went below it for a very short duration.

    This is deterministic: the same (duration_seconds, fps) pair
    always produces the same frame count, with no randomness or
    external state involved.

    Verified against two known-good data points: 2.0s at 24fps
    produces 49 frames (Sprint 4 Prompt 4's original RTX 4090 proof);
    2.0s at 8fps produces 17 frames (Sprint 4 Prompt 8's newer RunPod
    RTX 4090 proof) — both exactly matching their respective validated
    workflow's own proven value.

    Because frame count is snapped to the nearest valid value rather
    than computed exactly, the resulting clip's *actual* duration
    almost never exactly equals the requested `duration_seconds` — use
    `_frame_count_to_duration()` to compute what it actually will be.
    """
    raw_frame_count = max(_LENGTH_MINIMUM, round(duration_seconds * fps))
    steps = round((raw_frame_count - _LENGTH_MINIMUM) / _LENGTH_ALIGNMENT)
    return max(_LENGTH_MINIMUM, steps * _LENGTH_ALIGNMENT + _LENGTH_MINIMUM)


def _frame_count_to_duration(frame_count: int, fps: int) -> float:
    """The exact inverse arithmetic of `_duration_to_frame_count`'s
    first step — the effective duration a given (already Wan-valid or
    not) frame count actually produces at `fps`. Exposed so a caller
    can report "you asked for 2.0s; Wan will actually produce
    {_frame_count_to_duration(_duration_to_frame_count(2.0, fps), fps)}s"
    rather than silently letting the two values differ unremarked."""
    return frame_count / fps


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
        t2v_workflow_path: str | None = None,
        timeout_seconds: float | None = None,
        output_dir: str | None = None,
        default_fps: int | None = None,
    ) -> None:
        """Initializes the provider.

        Args:
            base_url: ComfyUI server URL. Defaults to `settings.comfyui_base_url`.
            workflow_path: Path to the I2V ComfyUI API-format workflow
                JSON. Defaults to `settings.comfyui_workflow_path`.
            t2v_workflow_path: Path to the T2V ComfyUI API-format
                workflow JSON — a genuinely different graph (no
                LoadImage/start_image at all), not the same file with a
                flag. Defaults to `settings.comfyui_t2v_workflow_path`.
                Only loaded when an instruction's `animation_type` is
                `TEXT_TO_VIDEO`.
            timeout_seconds: Maximum time to wait for one generation to
                complete. Defaults to `settings.comfyui_timeout_seconds`.
            output_dir: Where to save downloaded clips. Defaults to
                `settings.animation_output_dir` (shared with
                MockAnimationProvider — both providers' outputs live in
                the same configured location).
            default_fps: Used to convert `duration_seconds` into a
                frame count when `instruction.fps` is unset. Defaults
                to `settings.comfyui_default_fps`, now 8 — Sprint 4
                Prompt 8's newer live validation superseded Prompt 4's
                24 (see module docstring).
        """
        self._base_url = (base_url if base_url is not None else settings.comfyui_base_url).rstrip("/")
        self._workflow_path = workflow_path if workflow_path is not None else settings.comfyui_workflow_path
        self._t2v_workflow_path = (
            t2v_workflow_path if t2v_workflow_path is not None else settings.comfyui_t2v_workflow_path
        )
        self._timeout_seconds = (
            timeout_seconds if timeout_seconds is not None else settings.comfyui_timeout_seconds
        )
        self._default_fps = default_fps if default_fps is not None else settings.comfyui_default_fps
        self._output_dir = Path(output_dir if output_dir is not None else settings.animation_output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)

    async def generate_animation(self, instruction: AnimationInstruction) -> AnimationResult:
        is_t2v = instruction.animation_type == AnimationType.TEXT_TO_VIDEO

        try:
            workflow = self._load_workflow(self._t2v_workflow_path if is_t2v else self._workflow_path)
        except FileNotFoundError as exc:
            return self._failure(instruction, str(exc))

        if not is_t2v:
            # AnimationInstruction's own model validator already
            # guarantees source_image_path is set whenever
            # animation_type isn't TEXT_TO_VIDEO — this checks the file
            # actually exists on disk, which no schema validator can.
            if not Path(instruction.source_image_path).exists():
                return self._failure(
                    instruction, f"Reference image not found: {instruction.source_image_path!r}."
                )

        prompt_id = None
        try:
            async with httpx.AsyncClient(timeout=self._timeout_seconds) as client:
                uploaded_name = None
                if not is_t2v:
                    try:
                        uploaded_name = await self._upload_image(client, instruction.source_image_path)
                    except httpx.HTTPError as exc:
                        return self._failure(instruction, f"Image upload to ComfyUI failed: {exc}")

                prepared_workflow = (
                    self._inject_t2v_parameters(workflow, instruction)
                    if is_t2v
                    else self._inject_i2v_parameters(workflow, instruction, uploaded_name)
                )

                try:
                    prompt_id = await self._queue_prompt(client, prepared_workflow)
                except httpx.HTTPError as exc:
                    return self._failure(instruction, f"Workflow submission to ComfyUI failed: {exc}")
                except ComfyUIRequestError as exc:
                    return self._failure(instruction, f"ComfyUI rejected the workflow: {exc}")
                logger.info(
                    "ComfyUI: queued prompt_id=%s for shot %s (mode=%s).",
                    prompt_id, instruction.shot_id, "t2v" if is_t2v else "i2v",
                )

                try:
                    history_entry = await self._wait_for_completion(client, prompt_id)
                except TimeoutError as exc:
                    return self._failure(instruction, f"Execution timeout: {exc}")
                except ComfyUIRequestError as exc:
                    return self._failure(instruction, f"Generation failed: {exc}")
                except httpx.HTTPError as exc:
                    return self._failure(instruction, f"Polling ComfyUI for completion failed: {exc}")

                try:
                    filename, subfolder, file_type = self._extract_output_reference(
                        history_entry, _T2V_NODE_SAVE_VIDEO if is_t2v else _NODE_SAVE_VIDEO
                    )
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
                "mode": "t2v" if is_t2v else "i2v",
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

    def _load_workflow(self, path_str: str) -> dict[str, Any]:
        """Loads a ComfyUI API-format workflow JSON from disk.

        Args:
            path_str: Which workflow file to load — the caller decides
                between `self._workflow_path` (I2V) and
                `self._t2v_workflow_path` (T2V) based on the
                instruction's `animation_type`.

        Raises:
            FileNotFoundError: If no workflow file exists at the
                configured path.
        """
        path = Path(path_str)
        if not path.exists():
            raise FileNotFoundError(
                f"ComfyUI workflow file not found at {path!r}. Expected the validated "
                "wan22_i2v_5b.json or wan22_t2v_5b.json — see "
                "products/chess2fight/rendering/workflows/README.md."
            )
        # Explicit encoding, not Path.read_text()'s platform default: this
        # workflow's negative prompt (node 7) is non-ASCII Chinese text —
        # Windows' default locale encoding (cp1252) cannot decode it and
        # would raise or silently corrupt it. Sprint 4 Prompt 7.1.
        return json.loads(path.read_text(encoding="utf-8"))

    def _inject_i2v_parameters(
        self, workflow: dict[str, Any], instruction: AnimationInstruction, uploaded_image_name: str
    ) -> dict[str, Any]:
        """Injects AnimationInstruction values into a *copy* of the
        loaded I2V workflow graph, at the exact node IDs read from the
        validated wan22_i2v_5b.json — see the module-level `_NODE_*`
        constants. The original loaded dict is never mutated.

        The negative prompt is deliberately NOT overwritten when
        `instruction.negative_prompt` is unset — the validated workflow
        ships with a real, substantial, tuned negative prompt (node 7);
        an earlier version of this method unconditionally overwrote it
        with an empty string whenever the caller didn't supply one,
        which would have silently discarded that tuned content.

        `fps` is written to node 57 (CreateVideo), not just used to
        compute `frame_count` — an earlier version computed fps but
        never actually wrote it to the node, which happened not to
        matter while every caller used the default (which already
        matched the workflow's own baked-in value at the time), but
        would silently mismatch frame_count against playback rate for
        any other fps.
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

    def _inject_t2v_parameters(self, workflow: dict[str, Any], instruction: AnimationInstruction) -> dict[str, Any]:
        """Injects AnimationInstruction values into a *copy* of the
        loaded T2V workflow graph. The original loaded dict is never
        mutated.

        Never touches an image node — the T2V workflow (Sprint 4
        Prompt 8's `wan22_t2v_5b.json`) has no `LoadImage` node and no
        `start_image` input on `Wan22ImageToVideoLatent` at all (see
        module docstring); there is nothing to set. Everything else
        mirrors `_inject_i2v_parameters`'s reasoning (negative-prompt
        preservation, fps written to the CreateVideo node) exactly —
        this is a genuinely different graph, not different logic.
        """
        fps = instruction.fps or self._default_fps
        frame_count = _duration_to_frame_count(instruction.duration_seconds, fps)
        seed = instruction.seed if instruction.seed is not None else _derive_seed(instruction.prompt)
        width = _normalize_dimension(instruction.width)
        height = _normalize_dimension(instruction.height)

        prepared = json.loads(json.dumps(workflow))  # plain-JSON deep copy

        prepared[_T2V_NODE_POSITIVE_PROMPT]["inputs"]["text"] = instruction.prompt
        if instruction.negative_prompt:
            prepared[_T2V_NODE_NEGATIVE_PROMPT]["inputs"]["text"] = instruction.negative_prompt
        prepared[_T2V_NODE_SAMPLER]["inputs"]["seed"] = seed
        prepared[_T2V_NODE_VIDEO_LATENT]["inputs"]["width"] = width
        prepared[_T2V_NODE_VIDEO_LATENT]["inputs"]["height"] = height
        prepared[_T2V_NODE_VIDEO_LATENT]["inputs"]["length"] = frame_count
        prepared[_T2V_NODE_CREATE_VIDEO]["inputs"]["fps"] = fps

        return prepared

    # --- ComfyUI HTTP API calls -----------------------------------------------

    async def _upload_image(self, client: httpx.AsyncClient, image_path: str) -> str:
        """Uploads a local image to ComfyUI and returns the value its
        `LoadImage` node's `image` input should be set to.

        Defensive about the response shape, per this task's explicit
        instruction not to assume an output field exists merely
        because a mock used it: `name` is treated as required (a
        missing one is a genuine upload failure, surfaced as a
        KeyError caught by `generate_animation`'s outer handler); a
        missing `subfolder` is tolerated and treated as empty, since
        ComfyUI's own `/upload/image` omits it entirely for a plain
        root-level upload in some versions rather than returning `""`
        explicitly — both are valid, unremarkable responses.

        Uses ComfyUI's own `subfolder/filename` convention for
        `LoadImage.image` when a non-empty subfolder is returned
        (rather than silently discarding it and passing just the bare
        filename, which would point `LoadImage` at the wrong file if
        the upload landed in a subfolder) — this previously discarded
        the subfolder unconditionally; harmless while ComfyUI happens
        to return an empty subfolder for a plain upload, but wrong for
        any upload that doesn't.
        """
        image_bytes = Path(image_path).read_bytes()
        response = await client.post(
            f"{self._base_url}/upload/image",
            files={"image": (Path(image_path).name, image_bytes, "image/png")},
        )
        response.raise_for_status()
        data = response.json()
        filename = data["name"]
        subfolder = data.get("subfolder", "")
        return f"{subfolder}/{filename}" if subfolder else filename

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

    def _extract_output_reference(self, history_entry: dict[str, Any], save_video_node_id: str) -> tuple[str, str, str]:
        """Resolves the generated video's filename/subfolder/type from
        ComfyUI's own history response — never guesses a filename (see
        `_VIDEO_OUTPUT_KEYS`'s module-level docstring for exactly how
        confident this key-matching is, including a known,
        unresolved community-reported risk with this exact node).

        Args:
            history_entry: This job's entry from ComfyUI's /history response.
            save_video_node_id: Which node ID to check first —
                `_NODE_SAVE_VIDEO` for I2V or `_T2V_NODE_SAVE_VIDEO`
                for T2V (currently equal, but passed explicitly rather
                than hardcoded, since the two workflows are maintained
                as separate files that could diverge).
        """
        outputs = history_entry.get("outputs", {})

        primary_output = outputs.get(save_video_node_id, {})
        for key in _VIDEO_OUTPUT_KEYS:
            if key in primary_output and primary_output[key]:
                entry = primary_output[key][0]
                return entry["filename"], entry.get("subfolder", ""), entry.get("type", "output")

        # Fall back to scanning every node's output, in case the
        # reporting node ID doesn't match at runtime for some reason
        # (defensive, not expected).
        for node_output in outputs.values():
            for key in _VIDEO_OUTPUT_KEYS:
                if key in node_output and node_output[key]:
                    entry = node_output[key][0]
                    return entry["filename"], entry.get("subfolder", ""), entry.get("type", "output")

        raise ComfyUIRequestError(
            f"no video output found under node {save_video_node_id!r} or any other node, "
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
