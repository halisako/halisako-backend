"""Shared preflight checking for real (comfyui) Chess2Fight acceptance
runs — Sprint 4 Prompt 11.

Extracted, not rewritten, from `scripts/render_single_shot.py`'s
`_preflight_check()` (Sprint 4 Prompt 10/10.1) — that function's logic
is unchanged here, character for character, only relocated so
`scripts/render_multi_shot_acceptance.py` can reuse it instead of
copy-pasting it, per this task's explicit instruction to "reuse and
generalize the Prompt 10.1 fail-fast behavior rather than
copy/pasting it." `render_single_shot.py` itself was updated to import
from here instead of defining its own copy — its behavior is
unchanged (verified by its own still-passing test suite), only where
the code lives changed.

Deliberately shot-count-agnostic: this function only depends on which
providers/workflows/models are configured, never on how many shots an
acceptance run intends to select — so it needed zero changes to
generalize from single-shot to multi-shot use.

Lives here (an acceptance-layer module, alongside
`single_shot_acceptance.py`) rather than inside `RenderPipeline`,
`AnimationPipeline`, or the providers themselves, which stay
ComfyUI-agnostic — this is CLI/acceptance-harness concern, not
production pipeline logic.
"""

from __future__ import annotations

import shutil
import uuid
from pathlib import Path

import httpx


def check_output_writability(paths: list[str]) -> list[str]:
    """Confirms each given directory is genuinely writable, by
    actually creating it (if missing) and writing then deleting a
    small temporary probe file — more reliable across different
    filesystems/permission models than `os.access()` alone, which can
    give false positives/negatives on some setups. Returns a list of
    problem messages (empty means everything checked out); a confirmed
    unwritable path is meant to be treated as a hard preflight
    problem by the caller, blocking before the first generation call.

    Never leaves a permanent artifact behind — the probe file is
    always removed, including when the check itself fails partway
    (e.g. the directory can be created but not written to).

    Sprint 4 Prompt 11.1: added after finding Prompt 11's own stated
    preflight requirement ("output directory is writable") was never
    actually implemented — a confirmed gap between what was documented
    as checked and what the code actually checked.
    """
    problems: list[str] = []
    for path_str in paths:
        path = Path(path_str)
        try:
            path.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            problems.append(f"Cannot create output directory {path}: {exc}")
            continue

        probe_path = path / f".halisako_writability_probe_{uuid.uuid4().hex}"
        try:
            probe_path.write_text("probe")
        except OSError as exc:
            problems.append(f"Output directory {path} is not writable: {exc}")
        finally:
            probe_path.unlink(missing_ok=True)

    return problems


async def preflight_check(settings) -> tuple[list[str], list[str]]:
    """Checks obvious prerequisites for a REAL (comfyui) acceptance run
    before the first expensive generation call.

    Returns (problems, warnings): `problems` are hard failures the
    caller should treat as blocking (exit before generation);
    `warnings` are printed but don't block. For the /object_info
    model-visibility check below, the two are deliberately distinct
    (Sprint 4 Prompt 10.1): if /object_info for a node type is fetched
    and parsed successfully and a required model genuinely isn't in
    the list ComfyUI reports, that's a confirmed, actionable problem —
    hard-blocked. If /object_info itself can't be fetched, or its
    response can't be reliably parsed (its exact shape hasn't been
    independently verified against a live server), that's genuine
    uncertainty, not a confirmed absence — a warning, not a block. A
    no-op (returns ([], [])) for a mock run — there's nothing real to
    check.

    Runs ONCE per acceptance run (called once by each CLI, not
    per-shot) — the checks here (reachability, workflow files, model
    visibility) don't vary by which or how many shots are selected, so
    repeating them per shot would only add redundant network calls.

    Per this task's explicit constraints: never downloads models,
    never silently falls back to mock, and lives here (a CLI-layer
    concern) rather than inside RenderPipeline/AnimationPipeline/the
    providers themselves — those stay ComfyUI-agnostic.
    """
    using_comfyui = settings.image_provider == "comfyui" or settings.animation_provider == "comfyui"
    if not using_comfyui:
        return [], []

    problems: list[str] = []
    warnings: list[str] = []

    if shutil.which("ffprobe") is None:
        problems.append(
            "ffprobe not found on PATH — ComfyUIAnimationProvider verifies every downloaded "
            "video with it after generation completes."
        )
    if shutil.which("ffmpeg") is None:
        problems.append(
            "ffmpeg not found on PATH — required by MockImageProvider/MockAnimationProvider "
            "(used for whichever side of this run isn't routed through comfyui) and by VideoBuilder."
        )

    if settings.image_provider == "comfyui":
        flux_workflow = Path(settings.comfyui_image_workflow_path)
        if not flux_workflow.exists():
            problems.append(f"FLUX workflow file not found: {flux_workflow}")
    if settings.animation_provider == "comfyui":
        wan_workflow = Path(settings.comfyui_workflow_path)
        if not wan_workflow.exists():
            problems.append(f"Wan I2V workflow file not found: {wan_workflow}")

    base_url = settings.comfyui_base_url.rstrip("/")
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f"{base_url}/system_stats")
            response.raise_for_status()
    except httpx.HTTPError as exc:
        problems.append(f"ComfyUI not reachable at {base_url}: {exc}")
        return problems, warnings  # nothing further to check without a reachable server

    required_models: dict[str, list[str]] = {"UNETLoader": [], "CLIPLoader": [], "VAELoader": []}
    if settings.image_provider == "comfyui":
        required_models["UNETLoader"].append("flux-2-klein-4b.safetensors")
        required_models["CLIPLoader"].append("qwen_3_4b.safetensors")
        required_models["VAELoader"].append("flux2-vae.safetensors")
    if settings.animation_provider == "comfyui":
        required_models["UNETLoader"].append("wan2.2_ti2v_5B_fp16.safetensors")
        required_models["CLIPLoader"].append("umt5_xxl_fp8_e4m3fn_scaled.safetensors")
        required_models["VAELoader"].append("wan2.2_vae.safetensors")

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            for node_type, expected_filenames in required_models.items():
                if not expected_filenames:
                    continue
                response = await client.get(f"{base_url}/object_info/{node_type}")
                response.raise_for_status()
                info = response.json()

                # Distinguish "structure looks unexpected, can't reliably
                # tell" (warning) from "structure looks right, model
                # genuinely isn't in the list" (hard problem) — chained
                # .get(..., {}) calls would silently degrade an
                # unexpected shape into an empty set, which looks
                # identical to "confirmed no models available" and
                # would wrongly hard-block on a parsing quirk rather
                # than a genuine missing model. Sprint 4 Prompt 10.1.
                if node_type not in info:
                    warnings.append(
                        f"/object_info/{node_type} response had no {node_type!r} key — could not verify "
                        "model visibility for this node; confirm required models are installed manually."
                    )
                    continue

                # ComfyUI's /object_info/<node> nests combo (dropdown)
                # values under input.required.<field>[0] as a list.
                input_spec = info[node_type].get("input", {}).get("required", {})
                available: set[str] | None = None
                for field_spec in input_spec.values():
                    if isinstance(field_spec, list) and field_spec and isinstance(field_spec[0], list):
                        available = (available or set()) | set(field_spec[0])

                if available is None:
                    warnings.append(
                        f"/object_info/{node_type} response had no recognizable combo (dropdown) field — "
                        "could not verify model visibility for this node; confirm required models are installed manually."
                    )
                    continue

                for expected in expected_filenames:
                    if expected not in available:
                        # Confirmed: the response parsed as expected,
                        # and this exact filename genuinely isn't in
                        # the list ComfyUI reports — a hard, actionable
                        # signal the generation would fail, not a mere
                        # "couldn't determine" warning. A prior version
                        # of this check routed this to `warnings`,
                        # which let a paid acceptance run proceed
                        # straight into a doomed generation.
                        problems.append(
                            f"Model {expected!r} not found via /object_info/{node_type} — "
                            "confirm it's installed in the correct ComfyUI models directory."
                        )
    except (httpx.HTTPError, ValueError, KeyError, AttributeError) as exc:
        warnings.append(
            f"Could not verify model visibility via /object_info ({exc}) — proceeding without this "
            "check; confirm required models are installed manually."
        )

    return problems, warnings
