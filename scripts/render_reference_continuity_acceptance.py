#!/usr/bin/env python3
"""Reference-conditioned three-shot continuity experiment — Sprint 4 Prompt 13.

Drives the smallest architecture-faithful reference-conditioned
experiment: shot 0 is a normal FLUX text-to-image generation, becoming
the fight's canonical visual anchor; shots 1 and 2 are each
independently reference-conditioned against that SAME anchor (never
chained shot0->shot1->shot2). All three keyframes are then animated
via Wan and concatenated into one final MP4 — exactly the same
generation-count contract as Prompt 11/12
(3 FLUX + 3 Wan = 6 ComfyUI jobs, 1 local concatenation), just with 2
of the 3 FLUX jobs reference-conditioned instead of independent.

See products/chess2fight/rendering/reference_continuity_acceptance.py
for the underlying implementation and design notes, and
products/chess2fight/rendering/workflows/README-reference-conditioning.md
for the new workflow file's own provenance — read that before trusting
its exact graph shape the way the T2I/Wan workflows can be trusted.

Usage:
    # Dry run — no ComfyUI/network calls, uses the bundled sample PGN.
    python scripts/render_reference_continuity_acceptance.py --sample --dry-run

    # Real run with the currently configured providers (mock by default,
    # which — because MockImageProvider has no reference-conditioning
    # capability at all — will correctly fail preflight/execution rather
    # than silently doing something meaningless).
    python scripts/render_reference_continuity_acceptance.py --sample

To exercise the real, reference-conditioned path, set
IMAGE_PROVIDER=comfyui, ANIMATION_PROVIDER=comfyui, and
COMFYUI_BASE_URL first.

visual_seed_policy defaults to "shared" here (unlike
render_multi_shot_acceptance.py's own "default") — matching
ReferenceContinuityAcceptanceRunner.prepare()'s own default, since
this experiment's whole premise builds on the already-proven
shared-seed result.

--max-animation-seconds defaults to 2.0, not None — same Sprint 4
Prompt 11.1 safety reasoning as the plain multi-shot CLI: a normal
invocation with no duration flag must never accidentally generate
uncapped Wan animation. --allow-uncapped-duration is the same
high-friction, explicit opt-out.

shot_count defaults to, and is capped at, 3 unless --allow-more-than-cap
is passed explicitly — same Prompt 11 cost-control reasoning.
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
from products.chess2fight.rendering.acceptance_preflight import check_output_writability, preflight_check  # noqa: E402
from products.chess2fight.rendering.multi_shot_acceptance import (  # noqa: E402
    MultiShotPlan,
    ShotCountExceedsAcceptanceCapError,
    ShotRangeOutOfRangeError,
)
from products.chess2fight.rendering.reference_continuity_acceptance import (  # noqa: E402
    ReferenceAnchorInvalidError,
    ReferenceContinuityAcceptanceRunner,
)
from products.chess2fight.rendering.visual_continuity import VisualSeedPolicy, build_seed_override  # noqa: E402
from products.chess2fight.schemas import BattleMode, BattlePreferences  # noqa: E402

SAMPLE_PGN_PATH = Path(__file__).resolve().parent.parent / (
    "products/chess2fight/rendering/fixtures/sample_acceptance_game.pgn"
)

DEFAULT_MANIFEST_PATH = "reference_continuity_acceptance_manifest.json"


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
        help="How many consecutive shots to select. Default and safety cap: 3 — see --allow-more-than-cap. "
        "Shot 0 of the selection is always the T2I anchor; every other selected shot is reference-conditioned "
        "against it.",
    )
    parser.add_argument(
        "--allow-more-than-cap", action="store_true",
        help="Required to request --shot-count above the default safety cap of 3.",
    )
    parser.add_argument("--style", type=str, default="anime", help="Visual/narrative style. Default: anime.")
    parser.add_argument(
        "--battle-mode", type=str, default="duel", choices=[m.value for m in BattleMode],
        help="Battle mode. Default: duel.",
    )
    parser.add_argument(
        "--animation-width", type=int, default=None,
        help="Output width for every animated clip. Defaults to settings.comfyui_animation_default_width (832).",
    )
    parser.add_argument(
        "--animation-height", type=int, default=None,
        help="Output height for every animated clip. Defaults to settings.comfyui_animation_default_height (480).",
    )
    parser.add_argument(
        "--fps", type=int, default=None, help="FPS for Wan frame-count calculation. Defaults to settings.comfyui_default_fps (8).",
    )
    parser.add_argument(
        "--max-animation-seconds", type=float, default=2.0,
        help="Acceptance-only cap applied to every selected shot's animation duration. Default: 2.0 (safe). "
        "To run genuinely uncapped, pass --allow-uncapped-duration explicitly.",
    )
    parser.add_argument(
        "--allow-uncapped-duration", action="store_true",
        help="Required, in addition to omitting --max-animation-seconds, to run with no animation duration cap.",
    )
    parser.add_argument(
        "--visual-seed-policy", type=str, default="shared", choices=[p.value for p in VisualSeedPolicy],
        help="FLUX seed policy across all selected shots (anchor and reference-conditioned alike). "
        "Default: shared — matching this experiment's own premise (building on the already-proven "
        "shared-seed result). 'default' uses independent per-prompt seeds; 'derived' uses deterministic, "
        "per-shot-varying seeds from the same fight-level base.",
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
        "--skip-preflight", action="store_true", help="Skip the pre-generation preflight checks.",
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
    print("=== Reference continuity acceptance plan ===")
    print(f"selected shot indexes: {plan.selected_shot_indices} (of {plan.total_shots_in_timeline} total)")

    scene = plan.shots[0].scene
    print("--- stable fighter visual descriptors (identical across every selected shot) ---")
    print(f"  white: {scene.white_fighter.hair}, wearing {scene.white_fighter.clothing} and "
          f"{scene.white_fighter.armor}, wielding a {scene.white_fighter.weapon}")
    print(f"  black: {scene.black_fighter.hair}, wearing {scene.black_fighter.clothing} and "
          f"{scene.black_fighter.armor}, wielding a {scene.black_fighter.weapon}")
    print(f"stable arena descriptor: {scene.arena.layout}, {scene.arena.time_of_day}, {scene.arena.weather}")

    for i, (shot, eff_dur, frames) in enumerate(
        zip(plan.shots, plan.effective_animation_durations_seconds, plan.calculated_wan_frame_counts, strict=True)
    ):
        idx = plan.selected_shot_indices[i]
        mode = "T2I / anchor" if i == 0 else "reference-conditioned"
        reference = "(none — this shot IS the anchor)" if i == 0 else "shot 0's canonical anchor (same anchor for every reference-conditioned shot)"
        print(f"  shot[{idx}] generation mode = {mode}")
        print(f"    reference = {reference}")
        print(f"    effective={eff_dur:.2f}s frames={frames}")
        print(f"    flux seed: {plan.resolved_flux_seeds[i]}   wan seed: {plan.resolved_wan_seeds[i]}")

    print(f"image provider:      {plan.image_provider}")
    print(f"animation provider:  {plan.animation_provider}")
    print(f"comfyui base url:    {plan.comfyui_base_url}")
    print(f"t2i workflow:        {plan.comfyui_image_workflow_path}")
    print(f"reference workflow:  {get_settings().comfyui_reference_workflow_path}")
    print(f"wan workflow:        {plan.comfyui_animation_workflow_path}")
    print(f"FLUX image resolution:      {resolved_image_width}x{resolved_image_height}")
    print(f"Wan animation resolution:   {resolved_animation_width}x{resolved_animation_height}")
    print(f"resolved fps:        {plan.fps}")
    print(f"visual seed policy:  {plan.visual_seed_policy}")
    print(f"fight base visual seed: {plan.fight_base_visual_seed}")
    print(f"expected ComfyUI job count: {plan.expected_comfyui_job_count} "
          f"(1 T2I anchor + {plan.shot_count - 1} reference-conditioned FLUX + {plan.shot_count} Wan)")
    print(f"expected assembled duration: ~{plan.expected_assembled_duration_seconds:.3f}s")


def _write_manifest(manifest_path: str, plan: MultiShotPlan, result) -> None:
    """Sprint 4 Prompt 13: extends the Prompt 12 manifest shape with
    visual_generation_mode and canonical_anchor evidence. For shots
    1/2, reference_anchor_path is sourced from
    result.reference_anchor_paths — the runner's own recorded value
    from the actual generation call, never re-derived or assumed here
    — so a manifest showing both paths equal is evidence the same
    anchor was genuinely used, not a copy-pasted expectation."""
    scene = plan.shots[0].scene
    manifest = {
        "fight_id": plan.fight_id,
        "selected_shot_indices": plan.selected_shot_indices,
        "visual_generation_mode": "reference_conditioned",
        "canonical_anchor": {
            "shot_index": result.anchor.source_shot_index,
            "path": result.anchor.image_path,
            "actual_seed": result.anchor.generation_seed,
            "provenance": result.anchor.provenance,
        },
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
                "generation_mode": mode,
                "reference_anchor_path": reference_anchor_path,
                "planned_flux_seed": planned_seed,
                "actual_flux_seed": actual_seed,
                "wan_seed": wan_seed,
                "keyframe_path": image_path,
                "clip_path": video_path,
            }
            for idx, shot, mode, reference_anchor_path, planned_seed, actual_seed, wan_seed, image_path, video_path in zip(
                plan.selected_shot_indices, plan.shots, result.generation_modes, result.reference_anchor_paths,
                plan.resolved_flux_seeds, result.actual_flux_seeds, plan.resolved_wan_seeds,
                result.image_paths, result.video_paths, strict=True,
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
    runner = ReferenceContinuityAcceptanceRunner(ai_provider)

    preferences = BattlePreferences(battle_mode=BattleMode(args.battle_mode), style=args.style)
    resolved_max_animation_seconds = None if args.allow_uncapped_duration else args.max_animation_seconds
    visual_seed_policy = VisualSeedPolicy(args.visual_seed_policy)

    try:
        plan = await runner.prepare(
            pgn, preferences, start_shot_index=args.start_shot_index, shot_count=args.shot_count,
            fps=args.fps, max_animation_seconds=resolved_max_animation_seconds,
            visual_seed_policy=visual_seed_policy, allow_exceeding_default_cap=args.allow_more_than_cap,
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

    # Reference-conditioning has no mock equivalent at all — unlike
    # ordinary T2I/animation, where a mock run is a legitimate way to
    # exercise the pipeline without ComfyUI. Checked explicitly, early,
    # and clearly here rather than letting a mock-configured run reach
    # generation and fail with a confusing "provider not registered" or
    # network error instead.
    #
    # Sprint 4 Prompt 13.1: BOTH image_provider and animation_provider
    # must be comfyui, not just image_provider — the acceptance
    # contract is 3 real FLUX + 3 real Wan = 6 real ComfyUI jobs; a run
    # with real FLUX but MockAnimationProvider would produce a video
    # but must never be reported as satisfying that contract.
    misconfigured_providers = []
    if settings.image_provider != "comfyui":
        misconfigured_providers.append(f"image_provider is {settings.image_provider!r}")
    if settings.animation_provider != "comfyui":
        misconfigured_providers.append(f"animation_provider is {settings.animation_provider!r}")
    if misconfigured_providers:
        print(
            f"\nERROR: {' and '.join(misconfigured_providers)}, not 'comfyui'. Reference-conditioned "
            "generation has no mock equivalent, and this experiment's acceptance contract requires all "
            "6 jobs (3 FLUX + 3 Wan) to be real — set IMAGE_PROVIDER=comfyui, ANIMATION_PROVIDER=comfyui, "
            "and COMFYUI_BASE_URL before running this script for real.",
            file=sys.stderr,
        )
        return 1

    if not args.skip_preflight:
        manifest_parent = str(Path(args.manifest_path).parent) or "."
        writability_problems = check_output_writability(
            [settings.render_storage_root, settings.animation_output_dir, settings.image_output_dir, manifest_parent]
        )
        if writability_problems:
            print("\nPreflight check failed — refusing to start generation:", file=sys.stderr)
            for problem in writability_problems:
                print(f"  - {problem}", file=sys.stderr)
            return 1

        problems, preflight_warnings = await preflight_check(settings, check_reference_workflow=True)
        for warning in preflight_warnings:
            print(f"WARNING: {warning}", file=sys.stderr)
        if problems:
            print("\nPreflight check failed — refusing to start generation:", file=sys.stderr)
            for problem in problems:
                print(f"  - {problem}", file=sys.stderr)
            return 1

    # Both the anchor (via render_pipeline) and the reference-conditioned
    # shots (via reference_provider) must resolve the SAME seed policy —
    # constructed here, not left to two independently-resolved values
    # that could drift. base_visual_seed is computed identically to how
    # runner.prepare() itself computed plan.fight_base_visual_seed (same
    # pure, deterministic function of the same inputs) — cannot diverge.
    from products.chess2fight.rendering.visual_continuity import derive_fight_base_visual_seed

    base_visual_seed = derive_fight_base_visual_seed(pgn, args.style, args.battle_mode)
    seed_override = build_seed_override(visual_seed_policy, base_visual_seed)

    reference_provider = ComfyUIImageProvider(seed_override=seed_override)

    if seed_override is not None:
        from core.image_router import ImageProviderRegistry, ImageRouter
        from products.chess2fight.rendering.asset_manager import AssetManager
        from products.chess2fight.rendering.render_pipeline import RenderPipeline

        image_registry = ImageProviderRegistry()
        image_registry.register("comfyui", lambda: ComfyUIImageProvider(seed_override=seed_override))
        render_pipeline = RenderPipeline(
            image_router=ImageRouter(registry=image_registry), asset_manager=AssetManager()
        )
        runner = ReferenceContinuityAcceptanceRunner(ai_provider, render_pipeline=render_pipeline)

    print(
        f"\nGenerating anchor + {plan.shot_count - 1} reference-conditioned shots via "
        f"image_provider={settings.image_provider!r}, animation_provider={settings.animation_provider!r}..."
    )
    try:
        result = await runner.execute(
            plan, reference_provider, width=args.animation_width, height=args.animation_height,
        )
    except ReferenceAnchorInvalidError as exc:
        print(f"\nERROR: canonical anchor invalid — {exc}", file=sys.stderr)
        print("No reference-conditioned generation was attempted for shots 1/2.", file=sys.stderr)
        return 1
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
            "\nAny shot whose generation completed before the failure remains on disk — nothing is "
            "deleted or replaced by a fake artifact on failure. This includes the anchor image itself, "
            "if the anchor succeeded but a later reference-conditioned call failed.",
            file=sys.stderr,
        )
        return 1

    print(f"\nanchor:       {result.anchor.image_path} (seed {result.anchor.generation_seed})")
    print(f"generation modes: {result.generation_modes}")
    print(f"keyframe paths:   {result.image_paths}")
    print(f"video paths:      {result.video_paths}")
    print(f"final video:      {result.final_video_path}")
    print(f"final duration:   {result.final_video_duration_seconds:.3f}s")

    _write_manifest(args.manifest_path, plan, result)
    print(f"manifest written: {args.manifest_path}")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(_main()))
