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

import json
import shutil
import uuid
from pathlib import Path

import httpx


def validate_reference_workflow_topology(workflow_path: str) -> list[str]:
    """Sprint 4 Prompt 13.1 — validates the reference-conditioned
    workflow JSON contains the required conditioning topology, purely
    by reading and parsing the file: no network calls, safe to run in
    ordinary tests.

    Checks:
        - a VAEEncode node exists (encodes the reference image);
        - a "positive" ReferenceLatent exists, fed by the positive
          text conditioning;
        - a "negative" ReferenceLatent exists, fed by
          ConditioningZeroOut's output;
        - CFGGuider.positive points at the positive ReferenceLatent;
        - CFGGuider.negative points at the negative ReferenceLatent.

    Returns a list of problem messages — empty means the topology
    looks correct. Never raises on a missing/malformed file; a
    problem describing that is returned instead, so this function's
    return value alone is always sufficient for a preflight caller to
    act on.
    """
    problems: list[str] = []
    path = Path(workflow_path)
    if not path.exists():
        return [f"Reference workflow file not found: {path}"]

    try:
        workflow = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return [f"Reference workflow file could not be read/parsed: {exc}"]

    vae_encode_nodes = [node_id for node_id, node in workflow.items() if node.get("class_type") == "VAEEncode"]
    if not vae_encode_nodes:
        problems.append("Reference workflow has no VAEEncode node — the reference image would never be encoded.")

    reference_latent_nodes = {
        node_id: node for node_id, node in workflow.items() if node.get("class_type") == "ReferenceLatent"
    }
    if len(reference_latent_nodes) < 2:
        problems.append(
            f"Reference workflow has {len(reference_latent_nodes)} ReferenceLatent node(s), expected 2 "
            "(one for the positive branch, one for the negative branch) — matching the official "
            "Flux.2 Klein distilled image-edit topology."
        )

    guider_nodes = [node for node in workflow.values() if node.get("class_type") == "CFGGuider"]
    if not guider_nodes:
        problems.append("Reference workflow has no CFGGuider node.")
        return problems  # nothing further to check without a guider to trace from

    guider = guider_nodes[0]["inputs"]
    positive_ref = guider.get("positive")
    negative_ref = guider.get("negative")

    def _points_at_a_reference_latent(link) -> bool:
        return isinstance(link, list) and len(link) == 2 and link[0] in reference_latent_nodes

    if not _points_at_a_reference_latent(positive_ref):
        problems.append(
            f"CFGGuider.positive ({positive_ref!r}) does not point at a ReferenceLatent node — "
            "the positive branch would not be reference-conditioned."
        )
    if not _points_at_a_reference_latent(negative_ref):
        problems.append(
            f"CFGGuider.negative ({negative_ref!r}) does not point at a ReferenceLatent node — "
            "the negative branch would not be reference-conditioned, contradicting the official topology."
        )

    # Confirm both ReferenceLatent nodes are fed the same latent (the
    # single canonical anchor) — not two different/unrelated latents.
    if len(reference_latent_nodes) >= 2:
        latent_sources = {tuple(node["inputs"].get("latent", [])) for node in reference_latent_nodes.values()}
        if len(latent_sources) > 1:
            problems.append(
                f"The {len(reference_latent_nodes)} ReferenceLatent nodes reference different latent sources "
                f"({latent_sources}) — both should encode the SAME anchor image."
            )

    return problems


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


async def preflight_check(settings, check_reference_workflow: bool = False) -> tuple[list[str], list[str]]:
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

    Args:
        check_reference_workflow: Sprint 4 Prompt 13 — when True, also
            verifies `settings.comfyui_reference_workflow_path`
            exists (the models it needs are the same distilled 4B
            stack the FLUX check above already covers, so no separate
            model-visibility check is needed for it). Defaults to
            False, preserving this function's exact prior behavior for
            its two existing callers (render_single_shot.py,
            render_multi_shot_acceptance.py), neither of which uses
            reference conditioning.

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
        if check_reference_workflow:
            reference_workflow = Path(settings.comfyui_reference_workflow_path)
            if not reference_workflow.exists():
                problems.append(f"Reference-conditioned FLUX workflow file not found: {reference_workflow}")
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

    # Sprint 4 Prompt 13.1 — strict reference-capability preflight,
    # only for a reference-conditioned acceptance run
    # (check_reference_workflow=True). Deliberately does NOT reuse the
    # model-visibility try/except above, and deliberately does NOT
    # follow that check's own "uncertain parsing -> warning" policy —
    # per this task's explicit instruction, inability to confirm
    # ReferenceLatent/VAEEncode here is always a hard problem: if the
    # node type genuinely isn't available, the reference job cannot
    # possibly succeed, and generating the paid T2I anchor first would
    # waste money for nothing. This is a stricter policy specific to
    # this one check, not a change to Prompt 10/11's existing warning
    # policy for the model-filename-list check above.
    if check_reference_workflow and settings.image_provider == "comfyui":
        topology_problems = validate_reference_workflow_topology(settings.comfyui_reference_workflow_path)
        problems.extend(topology_problems)

        for required_node_type in ("ReferenceLatent", "VAEEncode"):
            try:
                async with httpx.AsyncClient(timeout=10.0) as client:
                    response = await client.get(f"{base_url}/object_info/{required_node_type}")
                    response.raise_for_status()
                    info = response.json()
            except (httpx.HTTPError, ValueError) as exc:
                problems.append(
                    f"Could not confirm the {required_node_type!r} node is available on this ComfyUI "
                    f"installation ({exc}) — required for reference-conditioned generation. Update ComfyUI "
                    "or confirm this node is registered before spending GPU time on the anchor."
                )
                continue
            if required_node_type not in info:
                problems.append(
                    f"{required_node_type!r} node not found via /object_info/{required_node_type} on this "
                    "ComfyUI installation — required for reference-conditioned generation. This node may be "
                    "missing from an outdated ComfyUI install; update before running this experiment."
                )

    return problems, warnings


async def check_reference_method_node_availability(settings, candidate_methods: list[str]) -> list[str]:
    """Sprint 4 Prompt 16 — a dedicated, additional preflight check for
    the reference-latent method sweep experiment: confirms
    `FluxKontextMultiReferenceLatentMethod` is available on the live
    ComfyUI installation, and — when the `/object_info` response
    exposes the node's combo choices — that every value in
    `candidate_methods` is actually a supported choice.

    Kept separate from `preflight_check` (not another
    `check_*_workflow`-style flag on that function) since this check
    is specific to this one experiment's own new node, not a general
    reference-conditioning concern every reference-conditioned caller
    needs — `preflight_check(settings, check_reference_workflow=True)`
    remains the right call for the base checks (workflow files,
    ReferenceLatent, VAEEncode, model visibility), and this function is
    called in addition, only by this experiment's own CLI.

    Same strictness policy as the Prompt 13.1 ReferenceLatent/VAEEncode
    check this mirrors: any failure to confirm here is always a hard
    problem, never a warning — an unsupported or missing method choice
    means the paid job cannot possibly succeed as configured.

    Args:
        candidate_methods: The `reference_latents_method` values this
            run intends to submit (e.g. `["offset", "uxo/uno",
            "index_timestep_zero"]`) — checked against the live node's
            own advertised choices, when available.

    Returns:
        A list of problem strings — empty means the node (and, where
        checkable, every candidate value) is confirmed available.
    """
    problems: list[str] = []
    base_url = settings.comfyui_base_url.rstrip("/")
    node_type = "FluxKontextMultiReferenceLatentMethod"

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(f"{base_url}/object_info/{node_type}")
            response.raise_for_status()
            info = response.json()
    except (httpx.HTTPError, ValueError) as exc:
        return [
            f"Could not confirm the {node_type!r} node is available on this ComfyUI installation "
            f"({exc}) — required for the reference-latent method sweep. Update ComfyUI or confirm this "
            "node is registered before spending GPU time on any candidate."
        ]

    if node_type not in info:
        return [
            f"{node_type!r} node not found via /object_info/{node_type} on this ComfyUI installation — "
            "required for the reference-latent method sweep. This node may be missing from an outdated "
            "ComfyUI install; update before running this experiment."
        ]

    # When the response exposes the combo's own advertised choices,
    # verify every candidate is actually one of them. Structure follows
    # the same /object_info combo-input shape the model-visibility
    # check above already parses (a [choices_list, {}] pair) — if this
    # specific response doesn't expose it the same way, this check is
    # skipped rather than guessed at, same "don't overclaim uncertain
    # parsing" caution as the rest of this module.
    try:
        method_input_spec = info[node_type]["input"]["required"]["reference_latents_method"]
        available_choices = method_input_spec[0]
        if isinstance(available_choices, list):
            for candidate in candidate_methods:
                if candidate not in available_choices:
                    problems.append(
                        f"Candidate method {candidate!r} is not among this ComfyUI installation's own "
                        f"advertised choices for {node_type!r} ({available_choices}) — refusing to submit "
                        "an unsupported candidate."
                    )
    except (KeyError, IndexError, TypeError):
        pass  # combo shape not parseable this way — proceed without this specific sub-check

    return problems
