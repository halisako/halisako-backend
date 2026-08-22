#!/usr/bin/env python3
"""One-command REAL ComfyUI I2V smoke test — Sprint 4 Prompt 9.

Drives the actual, production ComfyUIAnimationProvider directly —
bypassing Chess2Fight's PGN/cinematic pipeline entirely — against a
real ComfyUI server, using a manually-supplied reference image and
prompt. This is the harness for the next paid GPU session: the
smallest possible command that exercises the REAL programmatic path —
image upload, prompt submission, completion polling, MP4 download and
verification — with no PGN, no chess analysis, no Chess2Fight
orchestration involved at all. For testing the full Chess2Fight path
(PGN through one real animated shot), see scripts/render_single_shot.py
instead — this script is deliberately narrower, isolating the ComfyUI
provider itself as the thing being proven.

Usage:
    python scripts/comfyui_single_shot_smoke.py \\
        --base-url http://<comfyui-host>:8188 \\
        --image path/to/reference.png \\
        --prompt "Cinematic battle animation, character attacks with a sword..."

Calls ComfyUIAnimationProvider.generate_animation() directly — not
through AnimationRouter — since this script always knows exactly
which provider it wants to exercise; the router's job (resolving
whichever provider `settings.animation_provider` currently names)
adds nothing here and would need a private-attribute workaround to
force it to a specific instance regardless of global configuration.
Still uses the real, existing AnimationInstruction/AnimationProvider
contracts unchanged — nothing here reimplements ComfyUI's HTTP
mechanics separately.

On failure, exits non-zero with the real error message. Never writes
a fake/placeholder MP4 — a failed generation produces no output file.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.animation_providers.comfyui import (  # noqa: E402
    ComfyUIAnimationProvider,
    _duration_to_frame_count,
    _frame_count_to_duration,
)
from core.animation_router import AnimationInstruction  # noqa: E402
from core.config import get_settings  # noqa: E402

# The validated baseline (Sprint 4 Prompt 8/9): 17 frames is the
# proven, live-tested smoke-test length. Used only to pick a default
# --duration-seconds when the caller doesn't specify one.
_VALIDATED_BASELINE_FRAME_COUNT = 17


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--base-url", type=str, default=None,
        help="ComfyUI server URL, e.g. http://127.0.0.1:8188. Defaults to settings.comfyui_base_url.",
    )
    parser.add_argument("--image", type=str, required=True, help="Path to the local reference image (I2V source).")
    parser.add_argument("--prompt", type=str, required=True, help="Positive motion prompt.")
    parser.add_argument(
        "--negative-prompt", type=str, default=None,
        help="Negative prompt. If omitted, the workflow's own tuned negative prompt is left untouched "
        "(never blanked) — matching AnimationInstruction's established negative-prompt-preservation behavior.",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Where to save the downloaded MP4. Defaults to settings.animation_output_dir.",
    )
    parser.add_argument(
        "--width", type=int, default=None,
        help="Output width. Defaults to the validated Wan baseline (settings.comfyui_animation_default_width, 832).",
    )
    parser.add_argument(
        "--height", type=int, default=None,
        help="Output height. Defaults to settings.comfyui_animation_default_height (480).",
    )
    parser.add_argument(
        "--duration-seconds", type=float, default=None,
        help="Requested animation duration, in seconds. Defaults to the validated baseline "
        f"({_VALIDATED_BASELINE_FRAME_COUNT} frames at the resolved fps).",
    )
    parser.add_argument(
        "--fps", type=int, default=None, help="Output frame rate. Defaults to settings.comfyui_default_fps (8).",
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Reproducibility seed. Defaults to a value deterministically derived from --prompt.",
    )
    parser.add_argument(
        "--timeout-seconds", type=float, default=None,
        help="Max time to wait for completion. Defaults to settings.comfyui_timeout_seconds.",
    )
    parser.add_argument(
        "--shot-id", type=str, default="smoke_test_shot",
        help="Identifier for this run, used in the output filename. Default: smoke_test_shot.",
    )
    return parser.parse_args()


async def _main() -> int:
    args = _parse_args()
    settings = get_settings()

    image_path = Path(args.image)
    if not image_path.exists():
        print(f"ERROR: image not found: {image_path}", file=sys.stderr)
        return 1

    resolved_fps = args.fps if args.fps is not None else settings.comfyui_default_fps
    resolved_width = args.width if args.width is not None else settings.comfyui_animation_default_width
    resolved_height = args.height if args.height is not None else settings.comfyui_animation_default_height

    if args.duration_seconds is not None:
        resolved_duration = args.duration_seconds
    else:
        resolved_duration = _frame_count_to_duration(_VALIDATED_BASELINE_FRAME_COUNT, resolved_fps)

    resolved_frame_count = _duration_to_frame_count(resolved_duration, resolved_fps)
    effective_duration = _frame_count_to_duration(resolved_frame_count, resolved_fps)

    provider = ComfyUIAnimationProvider(
        base_url=args.base_url,
        timeout_seconds=args.timeout_seconds,
        output_dir=args.output_dir,
        default_fps=resolved_fps,
    )

    # Best-effort read of the workflow's own baked-in sampler settings,
    # purely for the pre-submission summary print below — a missing or
    # unreadable workflow file is a real failure the actual
    # generate_animation() call further down will also hit and report
    # properly, so this is display-only, not validation.
    steps = cfg = sampler_name = "(unknown — could not read workflow)"
    try:
        workflow = provider._load_workflow(provider._workflow_path)
        steps = workflow["3"]["inputs"]["steps"]
        cfg = workflow["3"]["inputs"]["cfg"]
        sampler_name = workflow["3"]["inputs"]["sampler_name"]
    except Exception:
        pass

    print("=== ComfyUI I2V single-shot smoke test ===")
    print(f"base_url:            {provider._base_url}")
    print(f"workflow:            {provider._workflow_path}")
    print(f"image:               {image_path}")
    print(f"prompt:              {args.prompt}")
    print(f"negative_prompt:     {args.negative_prompt or '(none supplied — workflow default preserved)'}")
    print(f"resolved width:      {resolved_width}")
    print(f"resolved height:     {resolved_height}")
    print(f"resolved fps:        {resolved_fps}")
    print(f"steps (from workflow): {steps}")
    print(f"cfg (from workflow):   {cfg}")
    print(f"sampler (from workflow): {sampler_name}")
    print(f"requested duration:  {resolved_duration:.3f}s")
    print(f"resolved frame count: {resolved_frame_count} (effective duration: {effective_duration:.3f}s)")
    print()

    instruction = AnimationInstruction(
        shot_id=args.shot_id,
        source_image_path=str(image_path),
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        duration_seconds=resolved_duration,
        camera_motion="static",
        subject_motion="smoke test motion",
        width=resolved_width,
        height=resolved_height,
        fps=resolved_fps,
        seed=args.seed,
    )

    print(f"Submitting to ComfyUI at {provider._base_url}...")
    start = time.monotonic()
    result = await provider.generate_animation(instruction)
    elapsed = time.monotonic() - start

    print(f"elapsed:             {elapsed:.1f}s")
    if result.metadata.get("prompt_id"):
        print(f"prompt_id:           {result.metadata['prompt_id']}")

    if not result.success:
        print(f"\nFAILED: {result.error_message}", file=sys.stderr)
        return 1

    print("\nSUCCESS")
    print(f"video_path:          {result.video_path}")
    print(f"duration_seconds:    {result.duration_seconds}")
    print(f"resolution:          {result.width}x{result.height}")
    print(f"fps:                 {result.fps}")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(_main()))
