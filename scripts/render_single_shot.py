#!/usr/bin/env python3
"""Single-shot acceptance runner — Sprint 4 Prompt 7.

Drives exactly one real cinematic shot through the real Chess2Fight
rendering architecture (RenderPipeline, AnimationPipeline, and
whichever ImageRouter/AnimationRouter provider is currently
configured), without rendering the full 8-shot fight. See
products/chess2fight/rendering/single_shot_acceptance.py for the
underlying implementation and design notes.

Usage:
    # Dry run — no ComfyUI/network calls, uses the bundled sample PGN.
    python scripts/render_single_shot.py --sample --dry-run

    # Real run with the currently configured providers (mock by default).
    python scripts/render_single_shot.py --sample --shot-index 4

    # A specific PGN file.
    python scripts/render_single_shot.py --pgn-file my_game.pgn --shot-index 0

To exercise the real, externally-validated FLUX/Wan path instead of
mock, set the provider settings first — see
products/chess2fight/rendering/workflows/VALIDATED-SETTINGS.md for the
exact environment variables and one documented command.

This script contains no model downloading, ComfyUI installation, or
GPU provisioning — it only calls the existing provider abstractions;
whichever provider is configured is what runs.

Sprint 4 Prompt 10: real acceptance runs (image_provider=comfyui and/or
animation_provider=comfyui) are preflighted before the first expensive
generation call — ComfyUI reachability, local ffmpeg/ffprobe, configured
workflow files, and (best-effort) whether the required model filenames
are visible via ComfyUI's own /object_info API. See _preflight_check()
below for exactly what's checked and why. Mock runs skip this entirely
— there's nothing real to check. Use --skip-preflight to bypass (e.g.
if a check here has a false positive) — never used automatically.
"""

from __future__ import annotations

import argparse
import asyncio
import shutil
import sys
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.ai_router import get_ai_provider  # noqa: E402
from core.config import get_settings  # noqa: E402
from products.chess2fight.rendering.single_shot_acceptance import (  # noqa: E402
    ShotIndexOutOfRangeError,
    SingleShotAcceptanceRunner,
    SingleShotPlan,
)
from products.chess2fight.schemas import BattleMode, BattlePreferences  # noqa: E402

SAMPLE_PGN_PATH = Path(__file__).resolve().parent.parent / (
    "products/chess2fight/rendering/fixtures/sample_acceptance_game.pgn"
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    pgn_source = parser.add_mutually_exclusive_group()
    pgn_source.add_argument("--pgn", type=str, help="PGN text directly on the command line.")
    pgn_source.add_argument("--pgn-file", type=str, help="Path to a .pgn file.")
    pgn_source.add_argument(
        "--sample", action="store_true",
        help=f"Use the bundled sample game ({SAMPLE_PGN_PATH.name}).",
    )
    parser.add_argument("--shot-index", type=int, default=0, help="Which shot to select (0-indexed). Default: 0.")
    parser.add_argument("--style", type=str, default="anime", help="Visual/narrative style. Default: anime.")
    parser.add_argument(
        "--battle-mode", type=str, default="duel", choices=[m.value for m in BattleMode],
        help="Battle mode. Default: duel.",
    )
    parser.add_argument(
        "--animation-width", type=int, default=None,
        help="Output width for the ANIMATION step only (never the FLUX reference image — "
        "RenderPipeline.render() is never passed this value at all). Defaults to "
        "settings.comfyui_animation_default_width (832, the validated Wan resolution) when "
        "omitted. Sprint 4 Prompt 10: renamed from --width, and changed to default to None "
        "instead of a literal 1280, after finding the old literal default silently overrode "
        "the validated Wan policy on every run that didn't explicitly pass a value.",
    )
    parser.add_argument(
        "--animation-height", type=int, default=None,
        help="Output height for the ANIMATION step only. Defaults to "
        "settings.comfyui_animation_default_height (480) when omitted.",
    )
    parser.add_argument(
        "--fps", type=int, default=None,
        help="FPS for Wan frame-count calculation. Defaults to settings.comfyui_default_fps (currently 8).",
    )
    parser.add_argument(
        "--max-animation-seconds", type=float, default=None,
        help="Sprint 4 Prompt 7.1: acceptance-only cap on the animation duration, for a low-cost GPU "
        "smoke test — e.g. --max-animation-seconds 2 produces a real 17-frame clip at 8fps (the "
        "current validated Wan baseline) instead of "
        "the shot's full real duration. Never changes the real cinematic shot duration itself, only "
        "the duration passed to the animation step. Default: no cap (full real shot duration is used).",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Run orchestration and shot selection only — print the plan, make no ComfyUI/network calls.",
    )
    parser.add_argument(
        "--skip-preflight", action="store_true",
        help="Skip the pre-generation preflight checks (ComfyUI reachability, ffmpeg/ffprobe, workflow "
        "files, model visibility) that otherwise run automatically for a real (comfyui) acceptance run. "
        "Never enabled automatically — only for bypassing a false-positive check.",
    )
    return parser.parse_args()


def _resolve_pgn(args: argparse.Namespace) -> str:
    if args.pgn:
        return args.pgn
    if args.pgn_file:
        return Path(args.pgn_file).read_text(encoding="utf-8")
    if args.sample:
        return SAMPLE_PGN_PATH.read_text(encoding="utf-8")
    print("ERROR: one of --pgn, --pgn-file, or --sample is required.", file=sys.stderr)
    sys.exit(1)


async def _preflight_check(settings) -> tuple[list[str], list[str]]:
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


def _print_plan_summary(plan: SingleShotPlan) -> None:
    print("=== Single-shot acceptance plan ===")
    print(f"shot index:          {plan.shot_index} of {plan.total_shots_in_timeline}")
    print(f"shot type:           {plan.shot.shot_type.value}")
    print(f"original shot duration:       {plan.shot.duration_seconds:.2f}s")
    if plan.max_animation_seconds is not None:
        print(f"effective animation duration: {plan.effective_animation_duration_seconds:.2f}s (capped via --max-animation-seconds {plan.max_animation_seconds:.2f})")
    else:
        print(f"effective animation duration: {plan.effective_animation_duration_seconds:.2f}s (no cap requested — using the full real shot duration)")
    print(f"image prompt:        {plan.shot.image_prompt[:200]}{'...' if len(plan.shot.image_prompt) > 200 else ''}")
    print(f"image provider:      {plan.image_provider}")
    print(f"animation provider:  {plan.animation_provider}")
    print(f"comfyui base url:    {plan.comfyui_base_url}")
    print(f"flux workflow:       {plan.comfyui_image_workflow_path}")
    print(f"wan workflow:        {plan.comfyui_animation_workflow_path}")
    print(f"wan frame count:     {plan.calculated_wan_frame_count} frames @ {plan.fps}fps")


async def _main() -> int:
    args = _parse_args()
    pgn = _resolve_pgn(args)

    settings = get_settings()
    ai_provider = get_ai_provider()
    runner = SingleShotAcceptanceRunner(ai_provider)

    preferences = BattlePreferences(battle_mode=BattleMode(args.battle_mode), style=args.style)

    try:
        plan = await runner.prepare(
            pgn, preferences, shot_index=args.shot_index, fps=args.fps,
            max_animation_seconds=args.max_animation_seconds,
        )
    except ShotIndexOutOfRangeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # PGN parsing, analysis, etc.
        print(f"ERROR during orchestration: {exc}", file=sys.stderr)
        return 1

    _print_plan_summary(plan)

    if args.dry_run:
        print("\nDry run complete — no ComfyUI/network calls were made.")
        return 0

    if not args.skip_preflight:
        problems, preflight_warnings = await _preflight_check(settings)
        for warning in preflight_warnings:
            print(f"WARNING: {warning}", file=sys.stderr)
        if problems:
            print("\nPreflight check failed — refusing to start generation:", file=sys.stderr)
            for problem in problems:
                print(f"  - {problem}", file=sys.stderr)
            return 1

    print(f"\nRendering via image_provider={settings.image_provider!r}, animation_provider={settings.animation_provider!r}...")
    try:
        result = await runner.execute(plan, width=args.animation_width, height=args.animation_height)
    except Exception as exc:
        print(f"ERROR during rendering: {exc}", file=sys.stderr)
        return 1

    print(f"\nimage path: {result.image_path}")
    print(f"video path: {result.video_path}")
    print(f"video duration: {result.video_duration_seconds:.2f}s")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(_main()))
