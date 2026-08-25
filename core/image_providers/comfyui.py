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
from typing import Any, Callable

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

# Sprint 4 Prompt 13 — reference-conditioned workflow only
# (flux2_klein_reference_4b.json). Every other node ID above is
# shared between both workflow files by construction — the reference
# workflow is an additive extension of the T2I one, not a parallel
# reconstruction (see that file's own README).
_REF_NODE_LOAD_IMAGE = "ref:1"  # LoadImage, inputs.image

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
        seed_override: Callable[[str], int] | None = None,
        reference_workflow_path: str | None = None,
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
            seed_override: Sprint 4 Prompt 12 — an optional function
                from prompt text to a seed, replacing the default
                `_derive_seed(prompt)` when given. Deliberately a
                *callable*, not a fixed int: this lets one provider
                instance serve either a "shared" visual-continuity
                experiment (the callable ignores its argument and
                always returns the same fight-level base seed) or a
                "derived" one (the callable combines the base seed with
                each shot's own prompt, still varying per shot) without
                needing a different provider instance per shot — the
                generic `ImageProvider.generate_image(prompt, width,
                height)` interface itself is completely unchanged; this
                is a `ComfyUIImageProvider`-specific constructor option,
                not a new abstract-method parameter. Only ever set by
                an acceptance/experiment caller (see
                `products/chess2fight/rendering/visual_continuity.py`)
                — production `FightVideoPipeline` never sets this,
                leaving `None` (the existing per-prompt-hash behavior)
                as its unchanged default.
            reference_workflow_path: Sprint 4 Prompt 13 — path to the
                reference-conditioned/image-edit workflow JSON, used
                only by `generate_reference_conditioned_image` (never
                by `generate_image`, the generic T2I method). Defaults
                to `settings.comfyui_reference_workflow_path`. See
                `workflows/README-reference-conditioning.md` for this
                workflow file's own provenance and limits — unlike the
                T2I workflow, it has not yet been proven on real
                hardware.
        """
        self._base_url = (base_url if base_url is not None else settings.comfyui_base_url).rstrip("/")
        self._workflow_path = workflow_path if workflow_path is not None else settings.comfyui_image_workflow_path
        self._timeout_seconds = (
            timeout_seconds if timeout_seconds is not None else settings.comfyui_timeout_seconds
        )
        self._output_dir = Path(output_dir if output_dir is not None else settings.image_output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._seed_override = seed_override
        self._reference_workflow_path = (
            reference_workflow_path if reference_workflow_path is not None
            else settings.comfyui_reference_workflow_path
        )

    def _resolve_seed(self, prompt: str) -> int:
        """The one seed-resolution rule this provider uses. Sprint 4
        Prompt 12.1: called exactly ONCE per `generate_image()`
        invocation (from that method itself, at the top) — the
        resulting value is then threaded through as a parameter to
        every seed-bearing use site (`_inject_parameters`, the local
        output filename, and the returned `metadata["seed"]`), rather
        than each site calling this method separately. A prior version
        of this docstring said calling this method from multiple sites
        was itself sufficient to keep them in agreement — true only
        because the Prompt 12 shared/derived seed_override callables
        happen to be pure; nothing about accepting an arbitrary
        `Callable[[str], int]` guarantees that in general, so resolving
        once and reusing the value is the actual guarantee now, not an
        assumption about caller purity.
        """
        return self._seed_override(prompt) if self._seed_override is not None else _derive_seed(prompt)

    async def generate_image(self, prompt: str, width: int = 1024, height: int = 1024) -> ImageGenerationResult:
        start = time.monotonic()

        # Sprint 4 Prompt 12.1: resolved exactly ONCE per generate_image()
        # call, then threaded through to every seed-bearing use site below
        # (workflow injection, the local filename, and the returned
        # metadata) — never re-resolved. The Prompt 12 shared/derived
        # seed_override callables happen to be pure (same prompt always
        # yields the same value), so calling _resolve_seed() multiple
        # times previously produced the same number in practice — but
        # ComfyUIImageProvider's constructor accepts an arbitrary
        # Callable[[str], int], and nothing guarantees every future
        # override stays pure. Resolving once and reusing the exact value
        # is correct regardless of what the override does internally.
        seed = self._resolve_seed(prompt)

        try:
            workflow = self._load_workflow()
        except (FileNotFoundError, json.JSONDecodeError) as exc:
            raise ImageProviderError(str(exc)) from exc

        try:
            prepared_workflow = self._inject_parameters(workflow, prompt, width, height, seed)
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
        # Sprint 4 Prompt 12.1: uses the same `seed` resolved once above
        # — previously called _derive_seed(prompt) directly here,
        # bypassing any seed_override entirely, so a shared-seed
        # generation's local filename could claim an unrelated
        # per-prompt hash instead of the seed ComfyUI actually used.
        output_path = self._output_dir / f"comfyui_flux_{seed}_{prompt_id}.png"
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
                "seed": seed,
                "model": "flux-2-klein-4b.safetensors (distilled) — experimentally validated, Sprint 4 Prompt 6",
            },
        )

    # --- Sprint 4 Prompt 13: reference-conditioned generation ------------------
    #
    # A ComfyUIImageProvider-specific capability, not part of the
    # generic ImageProvider interface (generate_image's signature
    # above is completely untouched) — per this task's own explicit
    # instruction not to overload the generic interface with
    # reference-conditioning arguments. Only ever called directly by
    # an acceptance/experiment caller that already knows it's holding
    # a concrete ComfyUIImageProvider (see
    # products/chess2fight/rendering/reference_continuity_acceptance.py)
    # — RenderPipeline/ImageRouter never call this, and never need to
    # know it exists, keeping RenderPipeline itself provider-agnostic
    # exactly as before.

    async def generate_reference_conditioned_image(
        self, prompt: str, reference_image_path: str, width: int = 1024, height: int = 1024,
    ) -> ImageGenerationResult:
        """Generates an image conditioned on both `prompt` and a
        reference image — the fight's canonical visual anchor, in
        Halisako's own usage — via the reference-conditioned workflow
        (`workflows/flux2_klein_reference_4b.json` by default; see
        that file's own README before trusting its exact graph shape).

        Mirrors `generate_image`'s own structure closely (same seed
        resolution, same HTTP call sequence, same error wrapping) —
        the only genuinely new step is uploading `reference_image_path`
        before submission, reusing the same `/upload/image` endpoint
        and subfolder-combining convention already proven in
        `core/animation_providers/comfyui.py`'s own `_upload_image`
        (mirrored here, not imported — these two provider modules stay
        deliberately independent; see this module's own docstring).

        Args:
            prompt: The reference-edit prompt — expected to explicitly
                distinguish what to preserve from the reference image
                vs. what to change (see
                `products/chess2fight/rendering/reference_continuity_acceptance.py`'s
                own prompt-composition contract) — this method itself
                has no opinion on prompt content, exactly like
                `generate_image`.
            reference_image_path: Local path to the reference image.
            width: Output width, in pixels.
            height: Output height, in pixels.

        Raises:
            ImageProviderError: On any failure — reference image
                missing/unreadable, workflow file missing, ComfyUI
                unreachable, generation failure, timeout, or an
                invalid output — same failure contract as
                `generate_image`. Never falls back to plain
                text-to-image on any failure here: a caller that wants
                that fallback must implement it explicitly, since
                silently substituting T2I would invalidate whatever
                reference-conditioning claim depends on this call
                actually having happened.
        """
        start = time.monotonic()

        if not Path(reference_image_path).exists():
            raise ImageProviderError(f"Reference image not found: {reference_image_path!r}.")

        seed = self._resolve_seed(prompt)

        try:
            workflow = self._load_workflow(self._reference_workflow_path)
        except (FileNotFoundError, json.JSONDecodeError) as exc:
            raise ImageProviderError(str(exc)) from exc

        try:
            async with httpx.AsyncClient(timeout=self._timeout_seconds) as client:
                try:
                    uploaded_reference_name = await self._upload_reference_image(client, reference_image_path)
                except httpx.HTTPError as exc:
                    raise ImageProviderError(f"Reference image upload to ComfyUI failed: {exc}") from exc

                try:
                    prepared_workflow = self._inject_reference_parameters(
                        workflow, prompt, width, height, seed, uploaded_reference_name,
                    )
                except ComfyUIImageRequestError as exc:
                    raise ImageProviderError(f"Invalid workflow mapping: {exc}") from exc

                try:
                    prompt_id = await self._queue_prompt(client, prepared_workflow)
                except httpx.HTTPError as exc:
                    raise ImageProviderError(f"Workflow submission to ComfyUI failed: {exc}") from exc
                except ComfyUIImageRequestError as exc:
                    raise ImageProviderError(f"ComfyUI rejected the workflow: {exc}") from exc
                logger.info("ComfyUI: queued reference-conditioned prompt_id=%s.", prompt_id)

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

        output_path = self._output_dir / f"comfyui_flux_reference_{seed}_{prompt_id}.png"
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
                "seed": seed,
                "model": "flux-2-klein-4b.safetensors (distilled) — same distilled stack as T2I, "
                "reference-conditioned workflow not yet live-validated, Sprint 4 Prompt 13",
                "reference_image_path": reference_image_path,
                "uploaded_reference_name": uploaded_reference_name,
            },
        )

    async def _upload_reference_image(self, client: httpx.AsyncClient, image_path: str) -> str:
        """Uploads a local image to ComfyUI and returns the value the
        reference workflow's `LoadImage` node's `image` input should
        be set to.

        Mirrors `core/animation_providers/comfyui.py`'s own
        `_upload_image` exactly — same endpoint, same defensive
        handling of a possibly-missing `subfolder` field, same
        `subfolder/filename` convention when one is returned
        non-empty. Not imported from there (these two provider
        modules stay deliberately independent — see this module's own
        docstring) — mirrored, so this method's own correctness
        doesn't depend on the other module's.
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

    # --- Workflow loading and parameter injection ---------------------------

    def _load_workflow(self, path_override: str | None = None) -> dict[str, Any]:
        """Loads a ComfyUI API-format workflow JSON from disk.

        Args:
            path_override: Sprint 4 Prompt 13 — loads this path instead
                of `self._workflow_path` when given (used by
                `generate_reference_conditioned_image` to load
                `self._reference_workflow_path` instead). `None`
                (the default) preserves `generate_image`'s existing
                call site exactly.

        Raises:
            FileNotFoundError: If no workflow file exists at the
                configured path.
            json.JSONDecodeError: If the file exists but isn't valid JSON.
        """
        path = Path(path_override if path_override is not None else self._workflow_path)
        if not path.exists():
            raise FileNotFoundError(
                f"ComfyUI workflow file not found at {path!r}. Expected the validated "
                "flux2_klein_t2i_4b.json (Sprint 4 Prompt 6) or "
                "flux2_klein_reference_4b.json (Sprint 4 Prompt 13) to already be in the "
                "repository — has a *_workflow_path setting been changed?"
            )
        # Explicit encoding, not Path.read_text()'s platform default —
        # general Windows portability, matching the same fix applied to
        # core/animation_providers/comfyui.py (where it's load-bearing:
        # that workflow's negative prompt is non-ASCII). This file's own
        # prompt text is currently ASCII, but pins the same explicit
        # behavior rather than relying on the platform default staying
        # compatible. Sprint 4 Prompt 7.1.
        return json.loads(path.read_text(encoding="utf-8"))

    def _inject_parameters(
        self, workflow: dict[str, Any], prompt: str, width: int, height: int, seed: int,
    ) -> dict[str, Any]:
        """Injects prompt/seed/dimensions into a *copy* of the loaded
        workflow graph, targeting exact validated node IDs — the
        original loaded dict is never mutated, so it can be safely
        reused across requests.

        Negative prompt is never touched: neither does this provider's
        interface accept one, nor does the validated distilled-branch
        workflow have a real negative-prompt text node to preserve or
        override — see module docstring.

        Args:
            seed: The already-resolved seed to inject — Sprint 4
                Prompt 12.1: this method no longer resolves its own
                seed via `_resolve_seed(prompt)`; the caller
                (`generate_image`) resolves it exactly once and passes
                the same value here and to every other seed-bearing
                use site, so a stateful seed_override (however unlikely
                today) can never cause the injected workflow seed and
                the reported result seed to disagree.

        Raises:
            ComfyUIImageRequestError: If an expected node ID is missing
                from the loaded workflow — a hard error now, not a
                skip-and-warn, since these are known-valid IDs from a
                verified file (see module docstring on why this
                differs from Prompt 5's title-lookup behavior).
        """
        norm_width = _normalize_dimension(width)
        norm_height = _normalize_dimension(height)

        prepared = json.loads(json.dumps(workflow))  # plain-JSON deep copy

        self._set_node_input(prepared, _NODE_PROMPT, "value", prompt)
        self._set_node_input(prepared, _NODE_WIDTH, "value", norm_width)
        self._set_node_input(prepared, _NODE_HEIGHT, "value", norm_height)
        self._set_node_input(prepared, _NODE_SEED, "noise_seed", seed)

        return prepared

    def _inject_reference_parameters(
        self, workflow: dict[str, Any], prompt: str, width: int, height: int, seed: int, uploaded_reference_name: str,
    ) -> dict[str, Any]:
        """Sprint 4 Prompt 13 — same structure as `_inject_parameters`,
        for the reference-conditioned workflow: every node ID it
        touches (`_NODE_PROMPT`, `_NODE_WIDTH`, `_NODE_HEIGHT`,
        `_NODE_SEED`) is identical between both workflow files by
        construction (see `flux2_klein_reference_4b.json`'s own
        README), plus the one new node this workflow adds
        (`_REF_NODE_LOAD_IMAGE`).

        Raises:
            ComfyUIImageRequestError: If an expected node ID is missing
                — same meaning as `_inject_parameters`, but here it's a
                genuinely open possibility this file's exact graph
                shape doesn't match what this method assumes, not just
                a hand-edit/corruption signal (see the workflow's own
                README on why).
        """
        norm_width = _normalize_dimension(width)
        norm_height = _normalize_dimension(height)

        prepared = json.loads(json.dumps(workflow))  # plain-JSON deep copy

        self._set_node_input(prepared, _NODE_PROMPT, "value", prompt)
        self._set_node_input(prepared, _NODE_WIDTH, "value", norm_width)
        self._set_node_input(prepared, _NODE_HEIGHT, "value", norm_height)
        self._set_node_input(prepared, _NODE_SEED, "noise_seed", seed)
        self._set_node_input(prepared, _REF_NODE_LOAD_IMAGE, "image", uploaded_reference_name)

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
