#!/usr/bin/env python3
"""Capped multi-shot acceptance runner — Sprint 4 Prompt 11.

Drives a capped range of real cinematic shots (default: shots 0-2,
i.e. the first 3) through the real Chess2Fight rendering architecture
(RenderPipeline, AnimationPipeline, VideoBuilder, and whichever
ImageRouter/AnimationRouter provider is currently configured), ending
in one real locally-concatenated MP4 — without rendering the full
8-shot fight. See
products/chess2fight/rendering/multi_shot_acceptance.py for the
underlying implementation and design notes; this generalizes
scripts/render_single_shot.py's proven pattern (real end-to-end GPU
success, Sprint 4 Prompt 10) to N shots.

Usage:
    # Dry run — no ComfyUI/network calls, uses the bundled sample PGN.
    python scripts/render_multi_shot_acceptance.py --sample --dry-run

    # Real run with the currently configured providers (mock by default).
    # Safely defaults to a ~2s animation cap per shot — see below.
    python scripts/render_multi_shot_acceptance.py --sample --start-shot-index 0 --shot-count 3

To exercise the real, externally-validated FLUX/Wan path instead of
mock, set the provider settings first — see
products/chess2fight/rendering/workflows/VALIDATED-SETTINGS.md.

This script contains no model downloading, ComfyUI installation, or
GPU provisioning — it only calls the existing provider abstractions;
whichever provider is configured is what runs. Reuses the exact same
preflight logic as render_single_shot.py (see
products/chess2fight/rendering/acceptance_preflight.py) rather than
duplicating it — real acceptance runs (image_provider=comfyui and/or
animation_provider=comfyui) are preflighted ONCE, before the first
generation call, not redundantly per shot. Use --skip-preflight to
bypass (e.g. if a check here has a false positive) — never used
automatically.

shot_count defaults to, and is capped at, 3 unless
--allow-more-than-cap is passed explicitly — a deliberate friction
point (this task's own cost-control requirement), not a convenience
default, so this script can't accidentally render the entire 8-shot
timeline.

Same principle for animation duration (Sprint 4 Prompt 11.1):
--max-animation-seconds defaults to 2.0, not None — a normal
invocation with no duration flag at all resolves every selected shot
to the safe ~17-frame/8fps baseline, never each shot's full real
cinematic duration (~7.75-8.91s each in the bundled sample game — over
3x the intended paid cost across 3 shots). Genuinely uncapped
generation requires the explicit, high-friction
--allow-uncapped-duration flag; omitting --max-animation-seconds alone
is never enough to mean "uncapped."
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.ai_router import get_ai_provider  # noqa: E402
from core.config import get_settings  # noqa: E402
from core.image_providers.comfyui import ComfyUIImageProvider  # noqa: E402
from core.image_router import ImageProviderRegistry, ImageRouter, MockImageProvider  # noqa: E402
from products.chess2fight.rendering.acceptance_preflight import check_output_writability, preflight_check  # noqa: E402
from products.chess2fight.rendering.multi_shot_acceptance import (  # noqa: E402
    MultiShotAcceptanceRunner,
    MultiShotPlan,
    ShotCountExceedsAcceptanceCapError,
    ShotRangeOutOfRangeError,
)
from products.chess2fight.rendering.render_pipeline import RenderPipeline  # noqa: E402
from products.chess2fight.rendering.visual_continuity import (  # noqa: E402
    VisualSeedPolicy,
    build_seed_override,
    derive_fight_base_visual_seed,
)
from products.chess2fight.schemas import BattleMode, BattlePreferences  # noqa: E402

SAMPLE_PGN_PATH = Path(__file__).resolve().parent.parent / (
    "products/chess2fight/rendering/fixtures/sample_acceptance_game.pgn"
)

DEFAULT_MANIFEST_PATH = "multi_shot_acceptance_manifest.json"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    pgn_source = parser.add_mutually_exclusive_group()
    pgn_source.add_argument("--pgn", type=str, help="PGN text directly on the command line.")
    pgn_source.add_argument("--pgn-file", type=str, help="Path to a .pgn file.")
    pgn_source.add_argument(
        "--sample", action="store_true", help=f"Use the bundled sample game ({SAMPLE_PGN_PATH.name})."
    )
    parser.add_argument(
        "--start-shot-index", type=int, default=0, help="First timeline shot to select (0-indexed). Default: 0.",
    )
    parser.add_argument(
        "--shot-count", type=int, default=3,
        help="How many consecutive shots to select. Default and safety cap: 3 — see --allow-more-than-cap.",
    )
    parser.add_argument(
        "--allow-more-than-cap", action="store_true",
        help="Required to request --shot-count above the default safety cap of 3. Never implied by any "
        "other flag — a deliberate override for future development, not the first paid milestone.",
    )
    parser.add_argument("--style", type=str, default="anime", help="Visual/narrative style. Default: anime.")
    parser.add_argument(
        "--battle-mode", type=str, default="duel", choices=[m.value for m in BattleMode],
        help="Battle mode. Default: duel.",
    )
    parser.add_argument(
        "--animation-width", type=int, default=None,
        help="Output width for every animated clip. Defaults to settings.comfyui_animation_default_width "
        "(832, the validated Wan resolution) when omitted. Never affects the FLUX reference images, which "
        "use their own independent policy (settings.comfyui_image_default_width, 1280).",
    )
    parser.add_argument(
        "--animation-height", type=int, default=None,
        help="Output height for every animated clip. Defaults to settings.comfyui_animation_default_height (480).",
    )
    parser.add_argument(
        "--fps", type=int, default=None, help="FPS for Wan frame-count calculation. Defaults to settings.comfyui_default_fps (currently 8).",
    )
    parser.add_argument(
        "--max-animation-seconds", type=float, default=2.0,
        help="Acceptance-only cap applied to every selected shot's animation duration. Default: 2.0 — "
        "produces real ~17-frame clips at 8fps (the current validated Wan baseline) per shot. Sprint 4 "
        "Prompt 11.1: this used to default to None (uncapped), letting a normal invocation with no flags "
        "accidentally generate all 3 shots at their full real cinematic durations (~25s of paid Wan "
        "generation instead of ~6.375s) — fixed to default safely. To run genuinely uncapped, pass "
        "--allow-uncapped-duration explicitly; this flag alone is never enough by itself.",
    )
    parser.add_argument(
        "--allow-uncapped-duration", action="store_true",
        help="Required, in addition to omitting --max-animation-seconds, to run with NO animation "
        "duration cap at all — each selected shot's full real cinematic duration is used instead of the "
        "safe ~2s default. A deliberate, high-friction opt-out, never implied by any other flag or by "
        "simply omitting --max-animation-seconds.",
    )
    parser.add_argument(
        "--manifest-path", type=str, default=DEFAULT_MANIFEST_PATH,
        help=f"Where to write the machine-readable acceptance manifest after a successful run. "
        f"Default: {DEFAULT_MANIFEST_PATH}.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Run orchestration and shot selection only — print the plan, make no ComfyUI/network calls, "
        "write no manifest.",
    )
    parser.add_argument(
        "--skip-preflight", action="store_true",
        help="Skip the pre-generation preflight checks. Never enabled automatically.",
    )
    parser.add_argument(
        "--visual-seed-policy", type=str, default="default", choices=[p.value for p in VisualSeedPolicy],
        help="Sprint 4 Prompt 12 — how each selected shot's FLUX seed is resolved. 'default' "
        "(unchanged pre-Prompt-12 behavior): each shot's FLUX seed is independently derived from its "
        "own prompt text. 'shared': every selected shot gets the identical fight-level base seed — a "
        "controlled experiment, not a guarantee of improved visual identity consistency. 'derived': "
        "every shot's seed is deterministically derived from the base seed combined with that shot's "
        "own prompt (varies per shot, unlike 'shared', but reproducible from the base seed). Wan's "
        "own seed is never affected by this flag, under any policy.",
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


def _print_plan_summary(
    plan: MultiShotPlan,
    resolved_image_width: int,
    resolved_image_height: int,
    resolved_animation_width: int,
    resolved_animation_height: int,
) -> None:
    print("=== Multi-shot acceptance plan ===")
    print(f"selected shot indexes: {plan.selected_shot_indices} (of {plan.total_shots_in_timeline} total)")

    # Sprint 4 Prompt 12: stable, fight-level visual descriptors —
    # shown once here, not per shot, since they're identical across
    # every selected shot by construction (compose_scene() builds
    # exactly one SceneContinuity per fight; see
    # prompt_generator.py's own _stable_continuity_block docstring for
    # the direct verification of this).
    scene = plan.shots[0].scene
    print("--- stable fighter visual descriptors (identical across every selected shot) ---")
    print(f"  white: {scene.white_fighter.hair}, {scene.white_fighter.facial_features}, "
          f"wearing {scene.white_fighter.clothing} and {scene.white_fighter.armor}, "
          f"wielding a {scene.white_fighter.weapon}")
    print(f"  black: {scene.black_fighter.hair}, {scene.black_fighter.facial_features}, "
          f"wearing {scene.black_fighter.clothing} and {scene.black_fighter.armor}, "
          f"wielding a {scene.black_fighter.weapon}")
    print(f"stable arena descriptor: {scene.arena.layout}, {scene.arena.time_of_day}, {scene.arena.weather}")
    print(f"stable art style: {scene.cinematic_art_style}")

    for i, (shot, orig_dur, eff_dur, frames) in enumerate(
        zip(plan.shots, [s.duration_seconds for s in plan.shots],
            plan.effective_animation_durations_seconds, plan.calculated_wan_frame_counts, strict=True)
    ):
        idx = plan.selected_shot_indices[i]
        abbreviated_prompt = shot.image_prompt[:100] + ("..." if len(shot.image_prompt) > 100 else "")
        print(f"  shot[{idx}] type={shot.shot_type.value!r} original={orig_dur:.2f}s effective={eff_dur:.2f}s frames={frames}")
        print(f"    action:  {shot.description}")
        print(f"    camera:  {shot.camera_angle.value}, {shot.camera_motion.value}")
        print(f"    flux seed: {plan.resolved_flux_seeds[i]}   wan seed: {plan.resolved_wan_seeds[i]}")
        print(f"    prompt: {abbreviated_prompt}")
    print(f"image provider:      {plan.image_provider}")
    print(f"animation provider:  {plan.animation_provider}")
    print(f"comfyui base url:    {plan.comfyui_base_url}")
    print(f"flux workflow:       {plan.comfyui_image_workflow_path}")
    print(f"wan workflow:        {plan.comfyui_animation_workflow_path}")
    # Sprint 4 Prompt 11.1: both resolutions shown explicitly and
    # separately — FLUX's reference-image policy and Wan's
    # animation policy are independently resolved and must never be
    # conflated, even for display purposes. Neither couples to the
    # generic ImageRouter (1024x1024) or AnimationInstruction
    # (1024x1024) defaults — both remain untouched; these are the
    # Chess2Fight-specific policies layered on top, same values
    # execute() itself will actually resolve and use.
    print(f"FLUX image resolution:      {resolved_image_width}x{resolved_image_height}")
    print(f"Wan animation resolution:   {resolved_animation_width}x{resolved_animation_height}")
    print(f"resolved fps:        {plan.fps}")
    # Sprint 4 Prompt 12: visual seed policy — the fight's base seed is
    # None under 'default' (no fight-level seed is computed or used at
    # all under that policy; each shot's FLUX seed is independently
    # derived from its own prompt text, unchanged from pre-Prompt-12
    # behavior). Wan's own seed is always _derive_seed(shot.image_prompt)
    # regardless of policy — shown per shot above for complete evidence,
    # not because it's ever affected by visual_seed_policy.
    print(f"visual seed policy:  {plan.visual_seed_policy}")
    print(f"fight base visual seed: {plan.fight_base_visual_seed}")
    print(f"expected ComfyUI job count: {plan.expected_comfyui_job_count} "
          f"({plan.shot_count} FLUX + {plan.shot_count} Wan)")
    print(f"expected assembled duration: ~{plan.expected_assembled_duration_seconds:.3f}s")


def _write_manifest(manifest_path: str, plan: MultiShotPlan, result) -> None:
    """Writes acceptance/debug evidence — never a substitute for the
    real domain objects (PromptedShot, AnimationResult, etc.), and
    never requires ComfyUI-specific fields in any generic model; every
    field here is sourced from the plan/result's own already-generic
    Pydantic fields."""
    scene = plan.shots[0].scene  # identical across every selected shot — see prompt_generator.py
    manifest = {
        "fight_id": plan.fight_id,
        "selected_shot_indices": plan.selected_shot_indices,
        # Sprint 4 Prompt 12: canonical, fight-level visual continuity
        # evidence — recorded once (not per shot) since it's identical
        # across every selected shot by construction, not because it
        # was arbitrarily deduplicated for this manifest.
        "canonical_visuals": {
            "white_fighter": scene.white_fighter.model_dump(),
            "black_fighter": scene.black_fighter.model_dump(),
            "arena": scene.arena.model_dump(),
            "cinematic_art_style": scene.cinematic_art_style,
        },
        "visual_seed_policy": plan.visual_seed_policy,
        "fight_base_visual_seed": plan.fight_base_visual_seed,
        "shots": [
            {
                "timeline_index": idx,
                "shot_id": shot.shot_id,
                "shot_type": shot.shot_type.value,
                "original_duration_seconds": shot.duration_seconds,
                "effective_animation_duration_seconds": eff_dur,
                "resolved_wan_frame_count": frames,
                # Sprint 4 Prompt 12.1: distinguished explicitly.
                # planned_flux_seed comes from plan.resolved_flux_seeds
                # (computed during prepare(), zero ComfyUI calls).
                # actual_flux_seed comes from result.actual_flux_seeds
                # — RenderPipeline's own real, provider-reported
                # RenderedFrame.metadata.generation_seed, never
                # re-derived from the prompt here. execute() itself
                # already verifies these agree (raising
                # SeedEvidenceMismatchError before any Wan call
                # otherwise, under a non-default policy) — both are
                # still recorded, so a successful manifest is itself
                # the evidence that they did.
                "planned_flux_seed": planned_flux_seed,
                "actual_flux_seed": actual_flux_seed,
                "wan_seed": wan_seed,
                "image_prompt": shot.image_prompt,
                "flux_keyframe_path": image_path,
                "animation_clip_path": video_path,
            }
            for idx, shot, eff_dur, frames, planned_flux_seed, actual_flux_seed, wan_seed, image_path, video_path in zip(
                plan.selected_shot_indices, plan.shots, plan.effective_animation_durations_seconds,
                plan.calculated_wan_frame_counts, plan.resolved_flux_seeds, result.actual_flux_seeds,
                plan.resolved_wan_seeds, result.image_paths, result.video_paths, strict=True,
            )
        ],
        "resolved_image_width": result.resolved_image_width,
        "resolved_image_height": result.resolved_image_height,
        "resolved_animation_width": result.resolved_animation_width,
        "resolved_animation_height": result.resolved_animation_height,
        "fps": plan.fps,
        "final_video_path": result.final_video_path,
        "expected_assembled_duration_seconds": plan.expected_assembled_duration_seconds,
        "actual_final_duration_seconds": result.final_video_duration_seconds,
        "image_provider": plan.image_provider,
        "animation_provider": plan.animation_provider,
    }
    Path(manifest_path).write_text(json.dumps(manifest, indent=2), encoding="utf-8")


async def _main() -> int:
    args = _parse_args()
    pgn = _resolve_pgn(args)

    settings = get_settings()
    ai_provider = get_ai_provider()

    # Sprint 4 Prompt 12: only construct a custom RenderPipeline (with
    # a seed_override-configured ComfyUIImageProvider registered as
    # "comfyui") when a non-default visual seed policy is actually
    # requested — the default policy leaves the runner's own default
    # RenderPipeline() completely untouched, exactly matching
    # pre-Prompt-12 behavior. base_visual_seed is computed here (for
    # constructing the provider) and, separately, again inside
    # runner.prepare() below (for the plan's own display fields) —
    # both calls are the same pure, deterministic function of the same
    # (pgn, style, battle_mode) inputs, so they cannot drift apart.
    visual_seed_policy = VisualSeedPolicy(args.visual_seed_policy)
    if visual_seed_policy != VisualSeedPolicy.DEFAULT:
        base_visual_seed = derive_fight_base_visual_seed(pgn, args.style, args.battle_mode)
        seed_override = build_seed_override(visual_seed_policy, base_visual_seed)
        image_registry = ImageProviderRegistry()
        image_registry.register("mock", MockImageProvider)
        image_registry.register("comfyui", lambda: ComfyUIImageProvider(seed_override=seed_override))
        render_pipeline = RenderPipeline(image_router=ImageRouter(registry=image_registry))
        runner = MultiShotAcceptanceRunner(ai_provider, render_pipeline=render_pipeline)
    else:
        runner = MultiShotAcceptanceRunner(ai_provider)

    preferences = BattlePreferences(battle_mode=BattleMode(args.battle_mode), style=args.style)

    # Sprint 4 Prompt 11.1: --allow-uncapped-duration always wins,
    # resolving to None (no cap) outright — not "whatever
    # --max-animation-seconds happens to be." argparse can't reliably
    # distinguish "the user explicitly typed --max-animation-seconds
    # 2.0" from "they didn't pass it and got the 2.0 default", so
    # trying to detect a contradictory explicit combination would be
    # unreliable; this simpler, unambiguous rule is deliberate, not an
    # oversight.
    resolved_max_animation_seconds = None if args.allow_uncapped_duration else args.max_animation_seconds

    try:
        plan = await runner.prepare(
            pgn, preferences, start_shot_index=args.start_shot_index, shot_count=args.shot_count,
            fps=args.fps, max_animation_seconds=resolved_max_animation_seconds,
            allow_exceeding_default_cap=args.allow_more_than_cap,
            visual_seed_policy=visual_seed_policy,
        )
    except (ShotRangeOutOfRangeError, ShotCountExceedsAcceptanceCapError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # PGN parsing, analysis, etc.
        print(f"ERROR during orchestration: {exc}", file=sys.stderr)
        return 1

    resolved_image_width = settings.comfyui_image_default_width
    resolved_image_height = settings.comfyui_image_default_height
    resolved_animation_width = (
        args.animation_width if args.animation_width is not None else settings.comfyui_animation_default_width
    )
    resolved_animation_height = (
        args.animation_height if args.animation_height is not None else settings.comfyui_animation_default_height
    )
    _print_plan_summary(plan, resolved_image_width, resolved_image_height, resolved_animation_width, resolved_animation_height)

    if args.dry_run:
        print("\nDry run complete — no ComfyUI/network calls were made, no manifest written.")
        return 0

    if not args.skip_preflight:
        # Writability first — cheapest check (no network calls), and
        # generic (not ComfyUI-specific, unlike preflight_check()
        # below, which is a no-op for a mock run). Sprint 4 Prompt
        # 11.1: this requirement was part of Prompt 11's own stated
        # preflight scope but was never actually implemented.
        manifest_parent = str(Path(args.manifest_path).parent) or "."
        output_paths_to_check = [settings.render_storage_root, settings.animation_output_dir, manifest_parent]
        writability_problems = check_output_writability(output_paths_to_check)
        if writability_problems:
            print("\nPreflight check failed — refusing to start generation:", file=sys.stderr)
            for problem in writability_problems:
                print(f"  - {problem}", file=sys.stderr)
            return 1

        problems, preflight_warnings = await preflight_check(settings)
        for warning in preflight_warnings:
            print(f"WARNING: {warning}", file=sys.stderr)
        if problems:
            print("\nPreflight check failed — refusing to start generation:", file=sys.stderr)
            for problem in problems:
                print(f"  - {problem}", file=sys.stderr)
            return 1

    print(
        f"\nRendering {plan.shot_count} shots via image_provider={settings.image_provider!r}, "
        f"animation_provider={settings.animation_provider!r}..."
    )
    try:
        result = await runner.execute(plan, width=args.animation_width, height=args.animation_height)
    except Exception as exc:
        print(f"\nERROR during rendering: {exc}", file=sys.stderr)
        print(
            "\nShot index <-> shot_id mapping, for cross-referencing the error above against any "
            "artifacts already on disk:",
            file=sys.stderr,
        )
        for idx, shot in zip(plan.selected_shot_indices, plan.shots, strict=True):
            print(f"  shot[{idx}]: shot_id={shot.shot_id!r}", file=sys.stderr)
        print(
            "\nAny shot whose generation completed before the failure remains on disk under "
            "storage/renders/ and the configured animation output directory — nothing is deleted "
            "or replaced by a fake artifact on failure.",
            file=sys.stderr,
        )
        return 1

    print(f"\nimage paths:  {result.image_paths}")
    print(f"video paths:  {result.video_paths}")
    print(f"final video:  {result.final_video_path}")
    print(f"final duration: {result.final_video_duration_seconds:.3f}s")

    _write_manifest(args.manifest_path, plan, result)
    print(f"manifest written: {args.manifest_path}")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(_main()))
