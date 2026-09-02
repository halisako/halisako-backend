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
        - CFGGuider.positive traces back to the positive ReferenceLatent;
        - CFGGuider.negative traces back to the negative ReferenceLatent.

    Sprint 4 Prompt 18: CFGGuider.positive/.negative now trace through
    an intermediate `FluxKontextMultiReferenceLatentMethod` node
    (Sprint 4 Prompt 16's own conditioning-method node, promoted into
    the production reference workflow this prompt) rather than
    requiring a direct link to ReferenceLatent — every reference
    workflow file in this codebase (production and all three Prompt 16
    calibration variants) now has this one-hop-further topology.
    Tracing through the method node (when present) rather than only
    accepting a direct ReferenceLatent link keeps this function correct
    for the current, real files, while still correctly failing a
    hypothetical file where the chain is broken some other way (e.g. a
    method node present but not actually fed by a ReferenceLatent, or
    a positive/negative input pointing at neither kind of node at all).

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

    method_nodes = {
        node_id: node for node_id, node in workflow.items()
        if node.get("class_type") == "FluxKontextMultiReferenceLatentMethod"
    }

    guider_nodes = [node for node in workflow.values() if node.get("class_type") == "CFGGuider"]
    if not guider_nodes:
        problems.append("Reference workflow has no CFGGuider node.")
        return problems  # nothing further to check without a guider to trace from

    guider = guider_nodes[0]["inputs"]
    positive_ref = guider.get("positive")
    negative_ref = guider.get("negative")

    def _traces_back_to_a_reference_latent(link) -> bool:
        """True if `link` points directly at a ReferenceLatent node, or
        at a FluxKontextMultiReferenceLatentMethod node that is itself
        fed by one — the two topologies every current reference
        workflow file in this codebase actually uses."""
        if not (isinstance(link, list) and len(link) == 2):
            return False
        source_node_id = link[0]
        if source_node_id in reference_latent_nodes:
            return True
        if source_node_id in method_nodes:
            upstream = method_nodes[source_node_id]["inputs"].get("conditioning")
            return isinstance(upstream, list) and len(upstream) == 2 and upstream[0] in reference_latent_nodes
        return False

    if not _traces_back_to_a_reference_latent(positive_ref):
        problems.append(
            f"CFGGuider.positive ({positive_ref!r}) does not trace back to a ReferenceLatent node (directly "
            "or through a FluxKontextMultiReferenceLatentMethod node) — the positive branch would not be "
            "reference-conditioned."
        )
    if not _traces_back_to_a_reference_latent(negative_ref):
        problems.append(
            f"CFGGuider.negative ({negative_ref!r}) does not trace back to a ReferenceLatent node (directly "
            "or through a FluxKontextMultiReferenceLatentMethod node) — the negative branch would not be "
            "reference-conditioned, contradicting the official topology."
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

        # Sprint 4 Prompt 18: the production reference workflow now
        # always routes through FluxKontextMultiReferenceLatentMethod
        # (promoted from Sprint 4 Prompt 16 calibration), using
        # "offset" specifically — required in addition to
        # ReferenceLatent/VAEEncode above, not instead of them. Reuses
        # check_reference_method_node_availability directly (that
        # function's own docstring covers why it's a hard failure, not
        # a warning, on the same strictness policy as the checks
        # above) rather than duplicating its HTTP logic.
        problems.extend(await check_reference_method_node_availability(settings, ["offset"]))

    return problems, warnings


def _extract_combo_options(method_input_spec) -> list[str] | None:
    """Sprint 4 Prompt 18.1 — extracts a ComfyUI `/object_info` combo
    input's advertised choices, supporting both known representations:

    1. The real, live ComfyUI representation (confirmed directly on an
       RTX 4090 RunPod instance, independent of this codebase's own
       assumptions):
       `["COMBO", {"advanced": ..., "multiselect": ..., "options": [...]}]`
       — a type-tag string followed by a config dict whose own
       `"options"` key holds the actual choice list.

    2. The legacy/test representation this codebase's own tests
       previously assumed exclusively (retained for compatibility,
       since some `/object_info` combo inputs elsewhere in this
       codebase's own existing tests already use this shape):
       `[[...], {}]` — the choice list directly as the first element.

    Root cause this fixes: the prior parsing logic
    (`method_input_spec[0]`) only ever handled representation 2. On
    the real representation 1, `method_input_spec[0]` evaluates to the
    string `"COMBO"` — not a list — so the prior `isinstance(...,
    list)` guard silently evaluated False and the entire capability
    check was skipped, with no problem reported, even when the
    candidate method genuinely wasn't supported. Confirmed by direct
    reproduction against the exact live schema before this fix existed.

    Returns:
        The list of advertised choices, or `None` if `method_input_spec`
        doesn't match either known shape — callers must treat `None`
        as "choices could not be reliably determined," not as "no
        problem," per this task's own explicit "do not silently skip
        the check when parsing fails" requirement.
    """
    if not (isinstance(method_input_spec, list) and len(method_input_spec) >= 1):
        return None
    first = method_input_spec[0]
    if isinstance(first, list):
        # Representation 2 (legacy/test): the choices are the first element directly.
        return first
    if isinstance(first, str) and first == "COMBO" and len(method_input_spec) >= 2:
        # Representation 1 (real, live ComfyUI): a type tag followed by a config dict.
        config = method_input_spec[1]
        if isinstance(config, dict) and isinstance(config.get("options"), list):
            return config["options"]
    return None


async def check_reference_method_node_availability(settings, candidate_methods: list[str]) -> list[str]:
    """Sprint 4 Prompt 16 — a dedicated, additional preflight check for
    the reference-latent method sweep experiment: confirms
    `FluxKontextMultiReferenceLatentMethod` is available on the live
    ComfyUI installation, and that every value in `candidate_methods`
    is actually one of the node's own advertised choices.

    Sprint 4 Prompt 18: `preflight_check(settings, check_reference_workflow=True)`
    now also calls this function directly, with `["offset"]` — since
    the production reference workflow always requires that specific
    capability. This function remains independently callable (not
    folded entirely into `preflight_check`) because calibration
    callers (Sprint 4 Prompt 16's own CLI) need to check a caller-
    supplied candidate list — "offset", "uxo/uno", or
    "index_timestep_zero" — that `preflight_check` itself has no
    reason to know about.

    Sprint 4 Prompt 18.1: now uses `_extract_combo_options` (see that
    function's own docstring for the real-vs-legacy schema distinction
    this fixes) and, per this task's own explicit requirement, treats
    an unparseable/missing choices shape as a hard failure rather than
    silently skipping the capability check — this is a strict,
    paid-GPU preflight; a check that can't confirm what it claims to
    confirm must fail, not pass by default.

    Same strictness policy as the Prompt 13.1 ReferenceLatent/VAEEncode
    check this mirrors: any failure to confirm here is always a hard
    problem, never a warning — an unsupported, missing, or
    undeterminable method choice means the paid job cannot be trusted
    to succeed as configured.

    Args:
        candidate_methods: The `reference_latents_method` values this
            run intends to submit (e.g. `["offset", "uxo/uno",
            "index_timestep_zero"]`) — checked against the live node's
            own advertised choices.

    Returns:
        A list of problem strings — empty means the node is available
        and every candidate is confirmed among its own advertised
        choices.
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

    try:
        method_input_spec = info[node_type]["input"]["required"]["reference_latents_method"]
    except (KeyError, TypeError):
        return [
            f"{node_type!r} node is present, but its /object_info response does not expose a "
            "'reference_latents_method' input at all — cannot confirm any candidate method is supported. "
            "Refusing to proceed with an unverifiable capability."
        ]

    available_choices = _extract_combo_options(method_input_spec)
    if available_choices is None:
        return [
            f"{node_type!r} node's 'reference_latents_method' input was present but its advertised "
            f"choices could not be reliably parsed ({method_input_spec!r}) — refusing to proceed without "
            "confirming candidate support. This may indicate a ComfyUI schema change; update this "
            "codebase's own parsing if so."
        ]

    for candidate in candidate_methods:
        if candidate not in available_choices:
            problems.append(
                f"Candidate method {candidate!r} is not among this ComfyUI installation's own "
                f"advertised choices for {node_type!r} ({available_choices}) — refusing to submit "
                "an unsupported candidate."
            )

    return problems
