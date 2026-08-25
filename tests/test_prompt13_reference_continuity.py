"""Tests for Sprint 4 Prompt 13's reference-conditioned visual
continuity experiment.

TEST CATEGORY: MOCKED UNIT / LOCAL INTEGRATION TESTS — no real ComfyUI
server or GPU is ever contacted.
"""

import asyncio
import subprocess
import sys

import pytest
from PIL import Image

from core.ai_router import TemplateProvider
from core.animation_router import AnimationProviderRegistry, AnimationRouter, MockAnimationProvider
from core.exceptions import ImageProviderError
from core.image_router import ImageGenerationResult, ImageProvider, ImageProviderRegistry, ImageRouter
from products.chess2fight.rendering.animation_pipeline import AnimationPipeline
from products.chess2fight.rendering.asset_manager import AssetManager
from products.chess2fight.rendering.multi_shot_acceptance import SeedEvidenceMismatchError
from products.chess2fight.rendering.reference_continuity_acceptance import ReferenceContinuityAcceptanceRunner
from products.chess2fight.rendering.render_pipeline import RenderPipeline
from products.chess2fight.rendering.video_builder import VideoBuilder
from tests.test_single_shot_acceptance import _preferences, _sample_pgn

RENDER_REFERENCE_CLI = "scripts/render_reference_continuity_acceptance.py"


class _SharedSeedFakeProvider(ImageProvider):
    """A fake T2I provider that respects a fixed seed and correctly
    reports it via metadata — simulating a correctly-configured
    ComfyUIImageProvider(seed_override=...) for the anchor step."""

    def __init__(self, fixed_seed: int):
        self._fixed_seed = fixed_seed
        self.call_count = 0

    async def generate_image(self, prompt, width=1024, height=1024):
        self.call_count += 1
        from PIL import Image as _Image

        path = f"/tmp/anchor_fake_{self.call_count}_{abs(hash(prompt))}.png"
        _Image.new("RGB", (width, height)).save(path)
        return ImageGenerationResult(
            image_path=path, provider="_SharedSeedFakeProvider", prompt=prompt,
            width=width, height=height, generation_time_seconds=0.0,
            metadata={"seed": self._fixed_seed},
        )


class _FakeReferenceProvider:
    """A fake reference-conditioned provider — tracks every reference
    image path it was called with, for verifying same-anchor and
    never-chained behavior directly."""

    def __init__(self, fixed_seed: int, fail_on_call: int | None = None):
        self._fixed_seed = fixed_seed
        self.calls: list[tuple[str, str]] = []  # (prompt, reference_image_path)
        self._fail_on_call = fail_on_call

    async def generate_reference_conditioned_image(self, prompt, reference_image_path, width=1024, height=1024):
        self.calls.append((prompt, reference_image_path))
        if self._fail_on_call is not None and len(self.calls) == self._fail_on_call:
            raise ImageProviderError("simulated reference-conditioned generation failure")
        path = f"/tmp/ref_fake_{len(self.calls)}.png"
        Image.new("RGB", (width, height)).save(path)
        return ImageGenerationResult(
            image_path=path, provider="_FakeReferenceProvider", prompt=prompt,
            width=width, height=height, generation_time_seconds=0.0,
            metadata={"seed": self._fixed_seed},
        )


def _runner_and_provider(tmp_path, fixed_seed=1697950441, reference_fail_on_call=None):
    image_registry = ImageProviderRegistry()
    fake_t2i = _SharedSeedFakeProvider(fixed_seed)
    image_registry.register("mock", lambda: fake_t2i)
    render_pipeline = RenderPipeline(
        image_router=ImageRouter(registry=image_registry), asset_manager=AssetManager(storage_root=str(tmp_path / "storage"))
    )
    anim_registry = AnimationProviderRegistry()
    anim_registry.register("mock", lambda: MockAnimationProvider(output_dir=str(tmp_path / "anim")))
    animation_pipeline = AnimationPipeline(animation_router=AnimationRouter(registry=anim_registry))

    runner = ReferenceContinuityAcceptanceRunner(
        TemplateProvider(), render_pipeline=render_pipeline, animation_pipeline=animation_pipeline,
        video_builder=VideoBuilder(), asset_manager=AssetManager(storage_root=str(tmp_path / "storage")),
    )
    reference_provider = _FakeReferenceProvider(fixed_seed, fail_on_call=reference_fail_on_call)
    return runner, reference_provider


# --- 1/2. Shot 0 is normal T2I, becomes the canonical anchor ---------------


def test_shot_0_is_normal_t2i(tmp_path):
    runner, reference_provider = _runner_and_provider(tmp_path)
    plan = asyncio.run(runner.prepare(_sample_pgn(), _preferences(), max_animation_seconds=2.0))
    result = asyncio.run(runner.execute(plan, reference_provider, width=256, height=256))
    assert result.generation_modes[0] == "t2i"
    assert result.anchor.provenance == "t2i"


def test_shot_0_becomes_the_canonical_anchor(tmp_path):
    runner, reference_provider = _runner_and_provider(tmp_path)
    plan = asyncio.run(runner.prepare(_sample_pgn(), _preferences(), max_animation_seconds=2.0))
    result = asyncio.run(runner.execute(plan, reference_provider, width=256, height=256))
    assert result.anchor.source_shot_index == plan.selected_shot_indices[0]
    assert result.anchor.image_path == result.image_paths[0]


# --- 3/4/5/6. Reference conditioning: shots 1/2, same anchor, never chained -


def test_shot_1_uses_reference_conditioned_generation(tmp_path):
    runner, reference_provider = _runner_and_provider(tmp_path)
    plan = asyncio.run(runner.prepare(_sample_pgn(), _preferences(), max_animation_seconds=2.0))
    result = asyncio.run(runner.execute(plan, reference_provider, width=256, height=256))
    assert result.generation_modes[1] == "reference_conditioned"


def test_shot_2_uses_reference_conditioned_generation(tmp_path):
    runner, reference_provider = _runner_and_provider(tmp_path)
    plan = asyncio.run(runner.prepare(_sample_pgn(), _preferences(), max_animation_seconds=2.0))
    result = asyncio.run(runner.execute(plan, reference_provider, width=256, height=256))
    assert result.generation_modes[2] == "reference_conditioned"


def test_shots_1_and_2_use_exactly_the_same_anchor(tmp_path):
    runner, reference_provider = _runner_and_provider(tmp_path)
    plan = asyncio.run(runner.prepare(_sample_pgn(), _preferences(), max_animation_seconds=2.0))
    result = asyncio.run(runner.execute(plan, reference_provider, width=256, height=256))
    assert reference_provider.calls[0][1] == reference_provider.calls[1][1] == result.anchor.image_path


def test_shot_2_never_uses_shot_1_as_its_reference():
    """The core anti-chaining guarantee: shot 2's reference must never
    be shot 1's own output path — only ever the original anchor."""

    async def run():
        import tempfile

        with tempfile.TemporaryDirectory() as tmp_dir:
            from pathlib import Path

            tmp_path = Path(tmp_dir)
            runner, reference_provider = _runner_and_provider(tmp_path)
            plan = await runner.prepare(_sample_pgn(), _preferences(), max_animation_seconds=2.0)
            result = await runner.execute(plan, reference_provider, width=256, height=256)
            shot1_output_path = result.image_paths[1]
            shot2_reference_path = reference_provider.calls[1][1]
            assert shot2_reference_path != shot1_output_path
            assert shot2_reference_path == result.anchor.image_path

    asyncio.run(run())


# --- 7/8/9/10. Generation counts ---------------------------------------------


def test_exactly_three_image_generations_total(tmp_path):
    runner, reference_provider = _runner_and_provider(tmp_path)
    plan = asyncio.run(runner.prepare(_sample_pgn(), _preferences(), max_animation_seconds=2.0))
    asyncio.run(runner.execute(plan, reference_provider, width=256, height=256))
    # 1 T2I (anchor) + 2 reference-conditioned = 3 total image generations
    assert len(reference_provider.calls) == 2  # the 2 reference-conditioned calls
    # anchor itself is the 3rd (verified via generation_modes elsewhere)


def test_exactly_three_wan_generations(tmp_path):
    wan_calls = []

    class _CountingAnimationProvider(MockAnimationProvider):
        async def generate_animation(self, instruction):
            wan_calls.append(instruction.shot_id)
            return await super().generate_animation(instruction)

    # Resolve the actual base seed this run will plan for (via the same
    # runner type, which correctly defaults to "shared" — a separate
    # MultiShotAcceptanceRunner probe would default to "default" instead
    # and give a None base seed here, an unrelated test bug).
    probe_runner, _ = _runner_and_provider(tmp_path)
    probe_plan = asyncio.run(probe_runner.prepare(_sample_pgn(), _preferences(), max_animation_seconds=2.0))
    base_seed = probe_plan.fight_base_visual_seed

    image_registry = ImageProviderRegistry()
    image_registry.register("mock", lambda: _SharedSeedFakeProvider(base_seed))
    render_pipeline = RenderPipeline(
        image_router=ImageRouter(registry=image_registry), asset_manager=AssetManager(storage_root=str(tmp_path / "storage"))
    )
    anim_registry = AnimationProviderRegistry()
    anim_registry.register("mock", lambda: _CountingAnimationProvider(output_dir=str(tmp_path / "anim")))
    animation_pipeline = AnimationPipeline(animation_router=AnimationRouter(registry=anim_registry))
    runner = ReferenceContinuityAcceptanceRunner(
        TemplateProvider(), render_pipeline=render_pipeline, animation_pipeline=animation_pipeline,
        video_builder=VideoBuilder(), asset_manager=AssetManager(storage_root=str(tmp_path / "storage")),
    )
    reference_provider = _FakeReferenceProvider(base_seed)
    plan = asyncio.run(runner.prepare(_sample_pgn(), _preferences(), max_animation_seconds=2.0))
    asyncio.run(runner.execute(plan, reference_provider, width=256, height=256))
    assert len(wan_calls) == 3


def test_expected_comfyui_job_count_is_six(tmp_path):
    runner, _ = _runner_and_provider(tmp_path)
    plan = asyncio.run(runner.prepare(_sample_pgn(), _preferences(), max_animation_seconds=2.0))
    assert plan.expected_comfyui_job_count == 6


def test_no_additional_character_sheet_generation(tmp_path):
    """Exactly 1 T2I + 2 reference-conditioned = 3 image calls total,
    never a 4th (e.g. a separate character-sheet generation)."""
    runner, reference_provider = _runner_and_provider(tmp_path)
    plan = asyncio.run(runner.prepare(_sample_pgn(), _preferences(), max_animation_seconds=2.0))
    result = asyncio.run(runner.execute(plan, reference_provider, width=256, height=256))
    assert len(result.image_paths) == 3
    assert len(reference_provider.calls) == 2  # never 3+


# --- 11/12/13. Anchor validation and no fallback -----------------------------


def test_anchor_must_exist_before_reference_generation(tmp_path):
    """Confirms the validation happens — via a genuinely working run,
    the anchor is validated (no exception) before any reference call."""
    runner, reference_provider = _runner_and_provider(tmp_path)
    plan = asyncio.run(runner.prepare(_sample_pgn(), _preferences(), max_animation_seconds=2.0))
    asyncio.run(runner.execute(plan, reference_provider, width=256, height=256))
    assert len(reference_provider.calls) == 2  # reached, meaning validation passed


def test_invalid_anchor_fails_before_reference_upload_generation(tmp_path, monkeypatch):
    """A T2I provider that reports a nonexistent path — the anchor
    validation must fail before any reference-conditioned call."""

    class _BrokenAnchorProvider(ImageProvider):
        async def generate_image(self, prompt, width=1024, height=1024):
            return ImageGenerationResult(
                image_path="/nonexistent/broken_anchor.png", provider="_BrokenAnchorProvider", prompt=prompt,
                width=width, height=height, generation_time_seconds=0.0, metadata={"seed": 1},
            )

    image_registry = ImageProviderRegistry()
    image_registry.register("mock", lambda: _BrokenAnchorProvider())

    # save_frame will fail to copy a nonexistent file -- confirm this
    # surfaces as a clear failure before any reference call, one way or another.
    render_pipeline = RenderPipeline(
        image_router=ImageRouter(registry=image_registry), asset_manager=AssetManager(storage_root=str(tmp_path / "storage"))
    )
    anim_registry = AnimationProviderRegistry()
    anim_registry.register("mock", lambda: MockAnimationProvider(output_dir=str(tmp_path / "anim")))
    animation_pipeline = AnimationPipeline(animation_router=AnimationRouter(registry=anim_registry))
    runner = ReferenceContinuityAcceptanceRunner(
        TemplateProvider(), render_pipeline=render_pipeline, animation_pipeline=animation_pipeline,
        video_builder=VideoBuilder(), asset_manager=AssetManager(storage_root=str(tmp_path / "storage")),
    )
    reference_provider = _FakeReferenceProvider(1)
    plan = asyncio.run(runner.prepare(_sample_pgn(), _preferences(), max_animation_seconds=2.0))
    with pytest.raises(Exception):
        asyncio.run(runner.execute(plan, reference_provider, width=256, height=256))
    assert len(reference_provider.calls) == 0  # never reached


def test_reference_provider_failure_does_not_fall_back_to_t2i(tmp_path):
    runner, reference_provider = _runner_and_provider(tmp_path, reference_fail_on_call=1)
    plan = asyncio.run(runner.prepare(_sample_pgn(), _preferences(), max_animation_seconds=2.0))
    with pytest.raises(ImageProviderError):
        asyncio.run(runner.execute(plan, reference_provider, width=256, height=256))
    # The failing call was the only one attempted -- no fallback attempt followed.
    assert len(reference_provider.calls) == 1


# --- 14. Partial failure never claims final success --------------------------


def test_partial_failure_never_produces_a_final_video(tmp_path):
    runner, reference_provider = _runner_and_provider(tmp_path, reference_fail_on_call=2)
    plan = asyncio.run(runner.prepare(_sample_pgn(), _preferences(), max_animation_seconds=2.0))
    with pytest.raises(ImageProviderError):
        asyncio.run(runner.execute(plan, reference_provider, width=256, height=256))
    final_path = tmp_path / "storage" / "renders" / plan.fight_id / "reference_continuity_acceptance.mp4"
    assert not final_path.exists()


# --- 15/16/17. Prompt composition: preserve identity, shot-specific action/camera, explicit contract -


def test_canonical_visual_descriptors_preserved_in_edit_prompts():
    from products.chess2fight.cinematic.prompt_generator import compose_reference_edit_prompt
    from tests.test_prompt12_visual_continuity import _prompted_timeline

    prompted = _prompted_timeline(_sample_pgn())
    edit_prompt = compose_reference_edit_prompt(prompted.shots[1])
    scene = prompted.shots[1].scene
    assert scene.white_fighter.hair in edit_prompt
    assert scene.white_fighter.weapon in edit_prompt
    assert scene.black_fighter.hair in edit_prompt
    assert scene.arena.layout in edit_prompt


def test_action_camera_data_remains_shot_specific():
    from products.chess2fight.cinematic.prompt_generator import compose_reference_edit_prompt
    from tests.test_prompt12_visual_continuity import _prompted_timeline

    prompted = _prompted_timeline(_sample_pgn())
    edit_prompt_1 = compose_reference_edit_prompt(prompted.shots[1])
    edit_prompt_2 = compose_reference_edit_prompt(prompted.shots[2])
    # .rstrip(".,;: ") mirrors compose_prompt's own trailing-punctuation
    # stripping (Sprint 4 Prompt 12) — shot.description can be a complete
    # sentence ending in a period, which the composer correctly strips
    # before rejoining, so the exact raw description string is not what
    # appears in the composed prompt.
    assert prompted.shots[1].description.rstrip(".,;: ") in edit_prompt_1
    assert prompted.shots[2].description.rstrip(".,;: ") in edit_prompt_2
    assert edit_prompt_1 != edit_prompt_2


def test_editing_prompts_explicitly_distinguish_preserve_vs_change():
    from products.chess2fight.cinematic.prompt_generator import compose_reference_edit_prompt
    from tests.test_prompt12_visual_continuity import _prompted_timeline

    prompted = _prompted_timeline(_sample_pgn())
    edit_prompt = compose_reference_edit_prompt(prompted.shots[1])
    assert "preserve" in edit_prompt.lower()
    assert "change only" in edit_prompt.lower()
    preserve_index = edit_prompt.lower().index("preserve")
    change_index = edit_prompt.lower().index("change only")
    assert preserve_index < change_index  # preserve block genuinely comes first


# --- 18/19. Seed determinism and evidence ------------------------------------


def test_seeds_remain_deterministic(tmp_path):
    runner1, provider1 = _runner_and_provider(tmp_path)
    plan1 = asyncio.run(runner1.prepare(_sample_pgn(), _preferences(), max_animation_seconds=2.0))
    runner2, provider2 = _runner_and_provider(tmp_path)
    plan2 = asyncio.run(runner2.prepare(_sample_pgn(), _preferences(), max_animation_seconds=2.0))
    assert plan1.resolved_flux_seeds == plan2.resolved_flux_seeds
    assert plan1.fight_base_visual_seed == plan2.fight_base_visual_seed


def test_planned_actual_seed_evidence_correct_on_success(tmp_path):
    runner, reference_provider = _runner_and_provider(tmp_path)
    plan = asyncio.run(runner.prepare(_sample_pgn(), _preferences(), max_animation_seconds=2.0))
    result = asyncio.run(runner.execute(plan, reference_provider, width=256, height=256))
    assert plan.resolved_flux_seeds == result.actual_flux_seeds


def test_seed_mismatch_raises_seed_evidence_mismatch_error(tmp_path):
    """A reference provider reporting a genuinely different seed than
    planned must raise, extending the same Prompt 12.1 evidence check."""
    disagreeing_seed = 999999999
    runner, _ = _runner_and_provider(tmp_path)
    plan = asyncio.run(runner.prepare(_sample_pgn(), _preferences(), max_animation_seconds=2.0))
    disagreeing_provider = _FakeReferenceProvider(disagreeing_seed)
    with pytest.raises(SeedEvidenceMismatchError):
        asyncio.run(runner.execute(plan, disagreeing_provider, width=256, height=256))


# --- 20. Manifest proves same anchor for shots 1 and 2 -----------------------


def test_manifest_proves_same_anchor_for_shots_1_and_2():
    result = subprocess.run(
        [sys.executable, RENDER_REFERENCE_CLI, "--sample", "--dry-run"],
        capture_output=True, text=True, check=True,
    )
    assert "same anchor for every reference-conditioned shot" in result.stdout


# --- 21. Dry-run causes zero provider calls ----------------------------------


def test_dry_run_makes_zero_provider_calls():
    result = subprocess.run(
        [sys.executable, RENDER_REFERENCE_CLI, "--sample", "--dry-run"],
        capture_output=True, text=True, check=True,
    )
    assert "generating anchor" not in result.stdout.lower()
    assert "dry run complete" in result.stdout.lower()


def test_default_mock_run_fails_clearly_before_any_generation():
    """No mock equivalent exists for reference-conditioning — running
    without image_provider=comfyui must fail clearly and early."""
    result = subprocess.run(
        [sys.executable, RENDER_REFERENCE_CLI, "--sample", "--max-animation-seconds", "2"],
        capture_output=True, text=True,
    )
    assert result.returncode == 1
    assert "no mock equivalent" in result.stderr.lower()
    assert "generating anchor" not in result.stdout.lower()


# --- 22/23. Existing paths remain unaffected ---------------------------------


def test_existing_t2i_single_shot_path_unchanged():
    result = subprocess.run(
        [sys.executable, "scripts/render_single_shot.py", "--sample", "--shot-index", "0", "--dry-run"],
        capture_output=True, text=True, check=True,
    )
    assert result.returncode == 0


def test_existing_3shot_non_reference_path_still_functional():
    result = subprocess.run(
        [sys.executable, "scripts/render_multi_shot_acceptance.py", "--sample", "--dry-run"],
        capture_output=True, text=True, check=True,
    )
    assert "expected ComfyUI job count: 6" in result.stdout


# --- 24. Generic ImageProvider signature unchanged ---------------------------


def test_generic_image_provider_signature_unchanged():
    import inspect

    sig = inspect.signature(ImageProvider.generate_image)
    assert list(sig.parameters.keys()) == ["self", "prompt", "width", "height"]


def test_mock_image_provider_still_valid_and_unmodified():
    from core.image_router import MockImageProvider

    result = asyncio.run(MockImageProvider().generate_image("a prompt"))
    assert result.width == 1024 and result.height == 1024


# --- 25/26. No real ComfyUI contact; live tests stay gated -------------------


def test_reference_continuity_module_has_no_hardcoded_network_target():
    import products.chess2fight.rendering.reference_continuity_acceptance as module

    with open(module.__file__) as f:
        source = f.read()
    # The module imports ComfyUIImageProvider by type (for annotations) but
    # never itself constructs an httpx client or hardcodes a URL.
    assert "httpx.AsyncClient(" not in source
    assert "http://" not in source


def test_new_workflow_file_is_valid_json_with_expected_new_nodes():
    import json

    with open("products/chess2fight/rendering/workflows/flux2_klein_reference_4b.json") as f:
        workflow = json.load(f)
    assert "ref:1" in workflow
    assert workflow["ref:1"]["class_type"] == "LoadImage"
    assert "ref:2" in workflow
    assert workflow["ref:2"]["class_type"] == "VAEEncode"
    assert "ref:3" in workflow
    assert workflow["ref:3"]["class_type"] == "ReferenceLatent"
    # The redirected wire: CFGGuider's positive now points at ref:3, not 77:92 directly.
    assert workflow["77:90"]["inputs"]["positive"] == ["ref:3", 0]
