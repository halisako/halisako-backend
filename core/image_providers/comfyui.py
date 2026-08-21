"""ComfyUIImageProvider: a real text-to-image ImageProvider that
communicates with a ComfyUI server's HTTP API to run a validated FLUX.2
Klein 4B (distilled) workflow.

    prompt -> ComfyUIImageProvider -> ComfyUI HTTP API -> FLUX.2 Klein -> ImageGenerationResult

SPRINT 4 PROMPT 6 STATUS UPDATE — this supersedes Prompt 5's provisional
implementation. Read this before trusting anything below:

Prompt 5 had no supplied workflow and no live ComfyUI/GPU access, so it
targeted nodes by `_meta.title` (a real, working FLUX workflow just
needed matching titles) and flagged real, unresolved uncertainty —
FLUX.1 vs FLUX.2, single vs. dual CLIP loader, the exact sampler chain.

Prompt 6 supplies the actual, experimentally-validated API-format
workflow (`workflows/flux2_klein_t2i_4b.json`), successfully executed
on a real RTX 4090, producing a real 1280x704 Halisako battle keyframe
that was then fed directly into the validated Wan 2.2 provider and
produced real motion — the full FLUX -> Wan handoff has been proven
end to end, not just each half independently. That resolves every
Prompt 5 uncertainty:

- FLUX.2 [klein] 4B **distilled** variant, confirmed (not FLUX.1, not
  the "base" non-distilled branch — the supplied file's CFG=1/steps=4
  matches the distilled branch specifically; the UI-export companion
  file at workflows/flux2_klein_t2i_4b.ui_export.json contains both
  branches as subgraphs, confirming this).
- Single `CLIPLoader` (not `DualCLIPLoader`), confirmed — Prompt 5's
  own research had found conflicting sources on this; the real file
  settles it.
- The modular `RandomNoise` + `KSamplerSelect` + `Flux2Scheduler` +
  `CFGGuider` + `SamplerCustomAdvanced` chain, confirmed — not a plain
  `KSampler`.
- No real negative-prompt text exists in this workflow at all — CFG=1
  (the distilled branch) uses `ConditioningZeroOut` on the positive
  conditioning instead of a second text encode. There is nothing to
  "preserve" for negative prompts here, unlike Wan.

Per this task's explicit instruction, node lookup is now BY EXACT
VALIDATED NODE ID, not title — title-based `_PARAMETER_NODE_MAP`
lookup is gone. Some real ComfyUI node IDs contain colons (e.g.
"77:84", from a collapsed subgraph) — these are treated as plain
string dict keys throughout; nothing here ever assumes an integer ID.

A missing expected node is now a hard error (`ComfyUIImageRequestError`),
not a skip-and-warn: Prompt 5's title lookup expected an unverified
workflow that might legitimately lack a node; this workflow is a known,
versioned, validated artifact, so a missing node means something is
actually wrong (a hand-edited or corrupted file), not "no real workflow
exists yet."

HTTP request/response handling remains written independently from
`core/animation_providers/comfyui.py` — same reasoning as Prompt 5:
similar shape, deliberately not shared, to avoid risking either
provider's own verified test suite for a deduplication neither
provider's correctness depends on.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
import uuid
from pathlib import Path
from typing import Any

import httpx
from PIL import Image

from core.config import get_settings
from core.exceptions import ImageProviderError
from core.image_router import ImageGenerationResult, ImageProvider

settings = get_settings()
logger = logging.getLogger(__name__)


# --- Exact, validated node IDs — see module docstring ------------------------
#
# From products/chess2fight/rendering/workflows/flux2_klein_t2i_4b.json,
# the API-format export of a workflow experimentally run and confirmed
# to produce a real image on an RTX 4090. Node IDs are exactly as
# ComfyUI assigned them — including the colon-containing ones from a
# collapsed subgraph ("77:84" etc.) — never reconstructed or guessed.
_NODE_PROMPT = "76"  # PrimitiveStringMultiline, inputs.value
_NODE_WIDTH = "77:84"  # PrimitiveInt (title "Width"), inputs.value
_NODE_HEIGHT = "77:85"  # PrimitiveInt (title "Height"), inputs.value
_NODE_SEED = "77:86"  # RandomNoise, inputs.noise_seed
_NODE_OUTPUT = "78"  # SaveImage

# Model loader nodes (77:87 UNETLoader/flux-2-klein-4b.safetensors,
# 77:88 CLIPLoader/qwen_3_4b.safetensors, 77:89 VAELoader/
# flux2-vae.safetensors) and workflow constants (CFG=1 at 77:90,
# steps=4 at 77:93, sampler=euler at 77:80) are deliberately never
# touched by this provider — validated workflow configuration, not
# per-request parameters. Per this task's own instruction: "Do not
# expose arbitrary knobs merely because they exist."

# FLUX.2 [klein] VAE-encodes/decodes at a spatial compression factor
# requiring dimensions divisible by 16 — same figure independently
# confirmed for Wan22ImageToVideoLatent (Prompt 4) and consistent with
# this workflow's own validated 1280x704 (both divisible by 16).
_DIMENSION_ALIGNMENT = 16

_IMAGE_OUTPUT_KEYS = ("images", "gifs", "videos")  # "images" is what SaveImage actually reports under

_POLL_INTERVAL_SECONDS = 2.0
_ERROR_MESSAGE_PAYLOAD_LIMIT = 500


def _truncate(text: str, limit: int = _ERROR_MESSAGE_PAYLOAD_LIMIT) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"... [truncated, {len(text)} chars total]"


class ComfyUIImageRequestError(Exception):
    """Internal to this module — never propagates past
    `generate_image`, which catches it and raises `ImageProviderError`
    instead, matching the existing `ImageProvider` contract: unlike
    `AnimationResult`, `ImageGenerationResult` has no success/
    error_message fields — failure is always an exception here, never
    data, exactly as `MockImageProvider`'s own contract already
    establishes."""


def _normalize_dimension(value: int, alignment: int = _DIMENSION_ALIGNMENT) -> int:
    """Rounds a width/height to the nearest multiple of `alignment`
    (minimum one full unit) — see `_DIMENSION_ALIGNMENT`'s docstring."""
    normalized = round(value / alignment) * alignment
    return max(alignment, normalized)


def _derive_seed(prompt: str) -> int:
    """A deterministic seed derived from the prompt — keeps generation
    reproducible by default, matching the same convention already
    established in `core/animation_providers/comfyui.py` and
    `products/chess2fight/rendering/render_pipeline.py` (not imported
    directly — this module intentionally doesn't depend on either)."""
    return int(hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:8], 16)


class ComfyUIImageProvider(ImageProvider):
    """Communicates with a real ComfyUI server to run the validated
    FLUX.2 Klein 4B (distilled) text-to-image workflow.

    Self-contained like every `ImageProvider`: no knowledge of
    `ImageRouter`, `ImageProviderRegistry`, or any other provider. Must
    not know about PGN, chess, BattleDirector, FastAPI, or ShotTimeline
    internals — it receives only a prompt and dimensions, exactly the
    same `(prompt, width, height)` contract `MockImageProvider` already
    implements.
    """

    def __init__(
        self,
        base_url: str | None = None,
        workflow_path: str | None = None,
        timeout_seconds: float | None = None,
        output_dir: str | None = None,
    ) -> None:
        """Initializes the provider.

        Args:
            base_url: ComfyUI server URL. Defaults to
                `settings.comfyui_base_url` — shared with
                ComfyUIAnimationProvider, per this task's explicit
                instruction not to duplicate this setting.
            workflow_path: Path to the ComfyUI API-format FLUX workflow
                JSON. Defaults to `settings.comfyui_image_workflow_path`
                (`workflows/flux2_klein_t2i_4b.json`).
            timeout_seconds: Maximum time to wait for one generation to
                complete. Defaults to `settings.comfyui_timeout_seconds`
                — also shared with the animation provider.
            output_dir: Where to save generated images. Defaults to
                `settings.image_output_dir` (shared with
                MockImageProvider — both providers' outputs live in the
                same configured location).
        """
        self._base_url = (base_url if base_url is not None else settings.comfyui_base_url).rstrip("/")
        self._workflow_path = workflow_path if workflow_path is not None else settings.comfyui_image_workflow_path
        self._timeout_seconds = (
            timeout_seconds if timeout_seconds is not None else settings.comfyui_timeout_seconds
        )
        self._output_dir = Path(output_dir if output_dir is not None else settings.image_output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)

    async def generate_image(self, prompt: str, width: int = 1024, height: int = 1024) -> ImageGenerationResult:
        start = time.monotonic()

        try:
            workflow = self._load_workflow()
        except (FileNotFoundError, json.JSONDecodeError) as exc:
            raise ImageProviderError(str(exc)) from exc

        try:
            prepared_workflow = self._inject_parameters(workflow, prompt, width, height)
        except ComfyUIImageRequestError as exc:
            raise ImageProviderError(f"Invalid workflow mapping: {exc}") from exc

        try:
            async with httpx.AsyncClient(timeout=self._timeout_seconds) as client:
                try:
                    prompt_id = await self._queue_prompt(client, prepared_workflow)
                except httpx.HTTPError as exc:
                    raise ImageProviderError(f"Workflow submission to ComfyUI failed: {exc}") from exc
                except ComfyUIImageRequestError as exc:
                    raise ImageProviderError(f"ComfyUI rejected the workflow: {exc}") from exc
                logger.info("ComfyUI: queued image prompt_id=%s.", prompt_id)

                try:
                    history_entry = await self._wait_for_completion(client, prompt_id)
                except TimeoutError as exc:
                    raise ImageProviderError(f"Execution timeout: {exc}") from exc
                except ComfyUIImageRequestError as exc:
                    raise ImageProviderError(f"Generation failed: {exc}") from exc

                try:
                    filename, subfolder, file_type = self._extract_output_reference(history_entry)
                except ComfyUIImageRequestError as exc:
                    raise ImageProviderError(f"History/output missing: {exc}") from exc

                try:
                    image_bytes = await self._download_output(client, filename, subfolder, file_type)
                except httpx.HTTPError as exc:
                    raise ImageProviderError(f"Image download failed: {exc}") from exc
        except httpx.HTTPError as exc:
            raise ImageProviderError(f"Could not reach ComfyUI at {self._base_url!r}: {exc}") from exc

        # Saved into Halisako's own local rendering storage — the rest
        # of the pipeline (RenderPipeline, AnimationPipeline) only ever
        # deals with local Halisako paths, never a ComfyUI-only
        # reference; see this task's Task 5 handoff verification.
        output_path = self._output_dir / f"comfyui_flux_{_derive_seed(prompt)}_{prompt_id}.png"
        output_path.write_bytes(image_bytes)

        try:
            actual_width, actual_height = self._verify_image(output_path)
        except ComfyUIImageRequestError as exc:
            raise ImageProviderError(f"Invalid output image: {exc}") from exc

        elapsed = time.monotonic() - start
        return ImageGenerationResult(
            image_path=str(output_path),
            provider="ComfyUIImageProvider",
            prompt=prompt,
            width=actual_width,
            height=actual_height,
            generation_time_seconds=elapsed,
            metadata={
                "prompt_id": prompt_id,
                "comfyui_base_url": self._base_url,
                "seed": _derive_seed(prompt),
                "model": "flux-2-klein-4b.safetensors (distilled) — experimentally validated, Sprint 4 Prompt 6",
            },
        )

    # --- Workflow loading and parameter injection ---------------------------

    def _load_workflow(self) -> dict[str, Any]:
        """Loads the ComfyUI API-format FLUX workflow JSON from disk.

        Raises:
            FileNotFoundError: If no workflow file exists at the
                configured path.
            json.JSONDecodeError: If the file exists but isn't valid JSON.
        """
        path = Path(self._workflow_path)
        if not path.exists():
            raise FileNotFoundError(
                f"ComfyUI FLUX workflow file not found at {path!r}. Expected the validated "
                "flux2_klein_t2i_4b.json (Sprint 4 Prompt 6) to already be in the repository — "
                "has settings.comfyui_image_workflow_path been changed?"
            )
        return json.loads(path.read_text(encoding="utf-8"))

    def _inject_parameters(self, workflow: dict[str, Any], prompt: str, width: int, height: int) -> dict[str, Any]:
        """Injects prompt/seed/dimensions into a *copy* of the loaded
        workflow graph, targeting exact validated node IDs — the
        original loaded dict is never mutated, so it can be safely
        reused across requests.

        Negative prompt is never touched: neither does this provider's
        interface accept one, nor does the validated distilled-branch
        workflow have a real negative-prompt text node to preserve or
        override — see module docstring.

        Raises:
            ComfyUIImageRequestError: If an expected node ID is missing
                from the loaded workflow — a hard error now, not a
                skip-and-warn, since these are known-valid IDs from a
                verified file (see module docstring on why this
                differs from Prompt 5's title-lookup behavior).
        """
        seed = _derive_seed(prompt)
        norm_width = _normalize_dimension(width)
        norm_height = _normalize_dimension(height)

        prepared = json.loads(json.dumps(workflow))  # plain-JSON deep copy

        self._set_node_input(prepared, _NODE_PROMPT, "value", prompt)
        self._set_node_input(prepared, _NODE_WIDTH, "value", norm_width)
        self._set_node_input(prepared, _NODE_HEIGHT, "value", norm_height)
        self._set_node_input(prepared, _NODE_SEED, "noise_seed", seed)

        return prepared

    def _set_node_input(self, workflow: dict[str, Any], node_id: str, input_key: str, value: Any) -> None:
        """Sets one input on one node, addressed by its exact (string —
        possibly colon-containing) ID.

        Raises:
            ComfyUIImageRequestError: If `node_id` isn't present in
                `workflow` at all.
        """
        node = workflow.get(node_id)
        if node is None:
            raise ComfyUIImageRequestError(
                f"expected node {node_id!r} not found in the loaded workflow "
                f"(has flux2_klein_t2i_4b.json been hand-edited?)."
            )
        node["inputs"][input_key] = value

    # --- ComfyUI HTTP API calls -----------------------------------------------

    async def _queue_prompt(self, client: httpx.AsyncClient, workflow: dict[str, Any]) -> str:
        response = await client.post(
            f"{self._base_url}/prompt",
            json={"prompt": workflow, "client_id": uuid.uuid4().hex},
        )
        response.raise_for_status()
        data = response.json()
        if data.get("node_errors"):
            raise ComfyUIImageRequestError(_truncate(str(data["node_errors"])))
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
                    raise ComfyUIImageRequestError(_truncate(str(status)))
                return entry
            await asyncio.sleep(_POLL_INTERVAL_SECONDS)
        raise TimeoutError(f"did not complete within {self._timeout_seconds}s (prompt_id={prompt_id}).")

    def _extract_output_reference(self, history_entry: dict[str, Any]) -> tuple[str, str, str]:
        """Resolves the generated image's filename/subfolder/type from
        ComfyUI's own history response — never guesses a filename.

        Checks the known, validated output node (`_NODE_OUTPUT`, "78",
        SaveImage) first; falls back to scanning every node's outputs
        for an image-shaped result, in case ComfyUI renumbers nodes on
        a workflow re-export.
        """
        outputs = history_entry.get("outputs", {})

        primary = outputs.get(_NODE_OUTPUT)
        if primary:
            for key in _IMAGE_OUTPUT_KEYS:
                if key in primary and primary[key]:
                    entry = primary[key][0]
                    return entry["filename"], entry.get("subfolder", ""), entry.get("type", "output")

        for node_output in outputs.values():
            for key in _IMAGE_OUTPUT_KEYS:
                if key in node_output and node_output[key]:
                    entry = node_output[key][0]
                    return entry["filename"], entry.get("subfolder", ""), entry.get("type", "output")

        raise ComfyUIImageRequestError(
            f"no image output found under node {_NODE_OUTPUT!r} or any other node, "
            f"checked keys {_IMAGE_OUTPUT_KEYS!r}."
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

    def _verify_image(self, path: Path) -> tuple[int, int]:
        """Validates the downloaded file is a real, decodable image
        with valid, non-mock-like dimensions — a successful HTTP
        response is not sufficient on its own. Uses Pillow directly
        (already a hard dependency of this backend, via
        MockImageProvider/MockAnimationProvider) rather than ffprobe —
        this is an image, not a video, so ffprobe would be the wrong
        tool, not a reused one.

        Raises:
            ComfyUIImageRequestError: If the file is missing, empty,
                or cannot be decoded as a valid image.
        """
        if not path.exists() or path.stat().st_size == 0:
            raise ComfyUIImageRequestError(f"downloaded file is missing or empty: {path}")

        try:
            with Image.open(path) as image:
                image.verify()
            with Image.open(path) as image:
                width, height = image.size
        except Exception as exc:
            raise ComfyUIImageRequestError(f"downloaded file is not a valid image: {exc}") from exc

        if not (width > 0 and height > 0):
            raise ComfyUIImageRequestError(f"downloaded image has invalid dimensions: {width}x{height}")

        return width, height
