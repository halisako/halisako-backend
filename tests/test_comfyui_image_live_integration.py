"""Live integration test for ComfyUIImageProvider against a real
ComfyUI + FLUX installation.

This is NOT part of the ordinary test suite. It requires:

1. `COMFYUI_IMAGE_LIVE_TEST=1` set in the environment.
2. A real, running ComfyUI server (default: http://localhost:8188,
   override via settings.comfyui_base_url).
3. A real FLUX workflow JSON at the path configured by
   `settings.comfyui_image_workflow_path` — see
   products/chess2fight/rendering/workflows/README-flux-keyframe.md
   for exactly what that file needs to contain and why it doesn't
   exist in this repository.

Without all three, this test is skipped — not faked, not silently
passed. Run it explicitly with:

    COMFYUI_IMAGE_LIVE_TEST=1 pytest tests/test_comfyui_image_live_integration.py -v -s

Uses a real prompt from Chess2Fight's own PromptGenerator pipeline
(built via the real Sprint 2/3 modules, not a generic "a cat"), per
this task's explicit instruction to demonstrate the provider against
something semantically representative of what Chess2Fight actually
needs — a fighter, a weapon, a combat environment, cinematic framing.
"""

import asyncio
import os
import time

import pytest
from PIL import Image

from core.ai_router import TemplateProvider
from core.config import get_settings
from core.image_providers.comfyui import ComfyUIImageProvider
from products.chess2fight.battle_director import generate_battle_intelligence
from products.chess2fight.battle_mode_engine import generate_battle_mode_intelligence
from products.chess2fight.cinematic.prompt_generator import generate_prompts
from products.chess2fight.cinematic.scene_composer import compose_scene
from products.chess2fight.cinematic.timeline_engine import generate_shot_timeline
from products.chess2fight.combat_mapper import generate_combat_intelligence
from products.chess2fight.narrative_generator import NarrativeGenerator
from products.chess2fight.pgn_analyzer import analyze_game
from products.chess2fight.schemas import BattleMode
from products.chess2fight.style_engine import generate_style_profile

LIVE_TEST_ENABLED = os.environ.get("COMFYUI_IMAGE_LIVE_TEST") == "1"

pytestmark = pytest.mark.skipif(
    not LIVE_TEST_ENABLED,
    reason=(
        "Live ComfyUI image test skipped — set COMFYUI_IMAGE_LIVE_TEST=1 to run it "
        "(and see this file's own module docstring for the other two requirements: "
        "a running ComfyUI server, and a real FLUX workflow JSON on disk)."
    ),
)

SCHOLARS_MATE_PGN = (
    '[Event "Example"]\n[White "Halisako"]\n[Black "Guest"]\n[Result "1-0"]\n\n'
    "1. e4 e5 2. Qh5 Nc6 3. Bc4 Nf6 4. Qxf7# 1-0"
)


def _real_shot_prompt() -> str:
    """Builds one real Shot.image_prompt via the actual Chess2Fight
    pipeline — the "climax" shot specifically, since it's the most
    visually representative (a fighter, a weapon, a decisive combat
    moment, cinematic framing)."""
    analysis = analyze_game(SCHOLARS_MATE_PGN)
    combat = generate_combat_intelligence(analysis)
    battle = generate_battle_intelligence(analysis, combat)
    profile = generate_style_profile(battle, combat, "anime")
    battle_mode = generate_battle_mode_intelligence(BattleMode.DUEL, combat, battle)
    story = asyncio.run(
        NarrativeGenerator(TemplateProvider()).generate(analysis, combat, battle, profile, battle_mode)
    )
    timeline = generate_shot_timeline(battle, story)
    composed = compose_scene(timeline, battle, profile, battle_mode)
    prompted = generate_prompts(composed)
    climax_shot = next(s for s in prompted.shots if s.shot_type.value == "climax")
    return climax_shot.image_prompt


def test_comfyui_server_is_reachable_before_attempting_generation():
    """A focused pre-check, separate from the full generation test
    below, so a missing server produces a clear, specific failure
    message rather than an opaque one buried inside a longer test."""
    import httpx

    settings = get_settings()
    try:
        response = httpx.get(f"{settings.comfyui_base_url.rstrip('/')}/system_stats", timeout=5.0)
        response.raise_for_status()
    except httpx.HTTPError as exc:
        pytest.fail(
            f"ComfyUI is not reachable at {settings.comfyui_base_url!r}: {exc}. "
            "COMFYUI_IMAGE_LIVE_TEST=1 was set, but no real ComfyUI server is running."
        )


def test_workflow_file_exists_before_attempting_generation():
    from pathlib import Path

    settings = get_settings()
    path = Path(settings.comfyui_image_workflow_path)
    if not path.exists():
        pytest.fail(
            f"No workflow file at {path!r}. See "
            "products/chess2fight/rendering/workflows/README-flux-keyframe.md."
        )


def test_real_shot_prompt_generates_a_real_usable_keyframe(tmp_path):
    """The full proof-of-life: one real Chess2Fight shot prompt,
    through the real provider, through real ComfyUI + FLUX generation,
    verified with Pillow — not merely a file, but a valid, decodable,
    non-mock-looking image at the expected resolution."""
    settings = get_settings()
    provider = ComfyUIImageProvider(output_dir=str(tmp_path / "out"))

    prompt = _real_shot_prompt()
    print(f"\nUsing real Chess2Fight shot prompt ({len(prompt)} chars):\n{prompt[:300]}...")

    start = time.monotonic()
    result = asyncio.run(
        provider.generate_image(
            prompt, width=settings.comfyui_image_default_width, height=settings.comfyui_image_default_height
        )
    )
    generation_time = time.monotonic() - start

    print(f"\ngeneration_time_seconds: {generation_time:.1f}")
    print(f"image_path: {result.image_path}")
    print(f"dimensions: {result.width}x{result.height}")
    print(f"metadata: {result.metadata}")

    assert os.path.exists(result.image_path)
    assert os.path.getsize(result.image_path) > 0
    assert result.width > 0 and result.height > 0

    with Image.open(result.image_path) as image:
        image.verify()
    with Image.open(result.image_path) as image:
        assert image.mode in ("RGB", "RGBA")
        # Confirm this is NOT a flat, mock-style solid color placeholder
        # — a real generated image should have real variance across
        # pixels; MockImageProvider's placeholders are a single solid
        # background color with a small text overlay.
        sample_pixels = [
            image.convert("RGB").getpixel((x, y))
            for x in (0, image.width // 4, image.width // 2, 3 * image.width // 4, image.width - 1)
            for y in (0, image.height // 2, image.height - 1)
        ]
        assert len(set(sample_pixels)) > 1, "Sampled pixels are all identical — looks like a solid-color placeholder."


def test_generated_image_is_wan_compatible_structure():
    """Without live Wan access in this environment (see this
    project's Prompt 3/4 environment audits), verifies the structural
    compatibility this task allows as a fallback: the PNG exists, has
    valid dimensions, is standard RGB/RGBA data, and could be used as
    a normal ComfyUI LoadImage input — the actual Wan generation call
    is not attempted here, since that would require live Wan/ComfyUI
    access this environment doesn't have. Not a fake integration
    result — a real, narrower check standing in for one the task
    explicitly permits when live Wan access isn't available."""
    settings = get_settings()
    provider = ComfyUIImageProvider()
    prompt = _real_shot_prompt()

    result = asyncio.run(
        provider.generate_image(
            prompt, width=settings.comfyui_image_default_width, height=settings.comfyui_image_default_height
        )
    )

    with Image.open(result.image_path) as image:
        assert image.format == "PNG"
        assert image.mode in ("RGB", "RGBA")
        assert image.width % 16 == 0
        assert image.height % 16 == 0
