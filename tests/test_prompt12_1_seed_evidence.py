"""Regression tests for Sprint 4 Prompt 12.1's seed evidence integrity
hardening:

1/5/6. ComfyUIImageProvider resolves its seed exactly once per
   generate_image() call and reuses that single value everywhere —
   verified with a deliberately stateful seed_override that would
   expose disagreement if the fix were wrong.
2/3/4. That single resolved value ends up, verified independently: in
   the workflow's injected RandomNoise.noise_seed, in
   ImageGenerationResult.metadata["seed"], and in the output filename.
7/8. RenderPipeline's FrameMetadata.generation_seed uses the actual
   provider-reported seed when available, and its existing
   deterministic fallback otherwise.
9-16. MultiShotAcceptanceRunner's planned-vs-actual seed evidence:
   plan-time seeds, execute()-time actual seeds, their agreement in
   the happy path, and SeedEvidenceMismatchError firing before any
   Wan/animation call on disagreement.
17/18. Default-policy backward compatibility; the generic
   ImageProvider signature stays untouched.
19/20. No ordinary test contacts a real ComfyUI server; live tests
   stay gated.

TEST CATEGORY: MOCKED UNIT / LOCAL INTEGRATION TESTS — no real ComfyUI
server or GPU is ever contacted.
"""

import asyncio
import re

import pytest
from PIL import Image

from core.image_router import ImageGenerationResult, ImageProvider, ImageProviderRegistry, ImageRouter
from products.chess2fight.rendering.animation_pipeline import AnimationPipeline
from products.chess2fight.rendering.asset_manager import AssetManager
from products.chess2fight.rendering.multi_shot_acceptance import (
    MultiShotAcceptanceRunner,
    SeedEvidenceMismatchError,
)
from products.chess2fight.rendering.render_pipeline import RenderPipeline
from products.chess2fight.rendering.video_builder import VideoBuilder
from products.chess2fight.rendering.visual_continuity import VisualSeedPolicy
from tests.test_comfyui_image_provider import (
    _patch_httpx_client,
    _provider_with_transport,
    _success_handlers,
)
from tests.test_single_shot_acceptance import _preferences, _sample_pgn


# --- 1/5/6. Seed resolved exactly once, even under a stateful override ------


def test_seed_resolved_exactly_once_per_generation_even_when_stateful(tmp_path, monkeypatch):
    """The core Issue 1/2 regression: a deliberately stateful override
    (returns a different value on each call) must still be called
    exactly once — never twice, which would previously have let
    workflow injection and reported metadata silently disagree."""
    provider, transport = _provider_with_transport(tmp_path, _success_handlers())
    _patch_httpx_client(monkeypatch, transport)

    call_log = []

    def stateful_override(prompt):
        call_log.append(prompt)
        return 1000 + len(call_log)

    provider._seed_override = stateful_override

    result = asyncio.run(provider.generate_image("test prompt"))

    assert len(call_log) == 1
    assert result.metadata["seed"] == 1001


def test_default_behavior_also_resolves_exactly_once(tmp_path, monkeypatch):
    """Even without an override, _resolve_seed's underlying
    _derive_seed is a pure function — but confirm the call pattern
    itself (not just the value) is now single-resolution, by spying on
    _derive_seed directly."""
    import core.image_providers.comfyui as module

    call_count = {"n": 0}
    real_derive = module._derive_seed

    def counting_derive(prompt):
        call_count["n"] += 1
        return real_derive(prompt)

    monkeypatch.setattr(module, "_derive_seed", counting_derive)

    provider, transport = _provider_with_transport(tmp_path, _success_handlers())
    _patch_httpx_client(monkeypatch, transport)

    asyncio.run(provider.generate_image("test prompt"))

    assert call_count["n"] == 1


# --- 2/3/4. All three seed-bearing use sites agree ---------------------------


def test_injected_workflow_seed_metadata_seed_and_filename_all_agree(tmp_path, monkeypatch):
    provider, transport = _provider_with_transport(tmp_path, _success_handlers())
    _patch_httpx_client(monkeypatch, transport)
    provider._seed_override = lambda prompt: 777

    result = asyncio.run(provider.generate_image("any prompt"))

    import json as _json

    queue_request = next(r for r in transport.requests if r.url.path == "/prompt")
    submitted_workflow = _json.loads(queue_request.content)["prompt"]
    injected_seed = submitted_workflow["77:86"]["inputs"]["noise_seed"]

    filename_seed = int(re.search(r"comfyui_flux_(\d+)_", result.image_path).group(1))

    assert injected_seed == 777
    assert result.metadata["seed"] == 777
    assert filename_seed == 777


def test_different_prompts_with_shared_override_all_agree_on_all_three_sites(tmp_path, monkeypatch):
    """The exact shared-seed scenario the next paid experiment uses:
    different prompts, same override, every use site for every call
    must agree."""
    provider, transport = _provider_with_transport(tmp_path, _success_handlers())
    _patch_httpx_client(monkeypatch, transport)
    provider._seed_override = lambda prompt: 555555

    import json as _json

    for prompt_text in ["first shot prompt", "second shot prompt", "third shot prompt"]:
        transport.requests.clear()
        result = asyncio.run(provider.generate_image(prompt_text))
        queue_request = next(r for r in transport.requests if r.url.path == "/prompt")
        submitted_workflow = _json.loads(queue_request.content)["prompt"]
        assert submitted_workflow["77:86"]["inputs"]["noise_seed"] == 555555
        assert result.metadata["seed"] == 555555
        filename_seed = int(re.search(r"comfyui_flux_(\d+)_", result.image_path).group(1))
        assert filename_seed == 555555


# --- 7/8. RenderPipeline FrameMetadata uses provider seed when available ---


def test_frame_metadata_uses_provider_reported_seed_when_available(tmp_path):
    class _SeedReportingProvider(ImageProvider):
        async def generate_image(self, prompt, width=1024, height=1024):
            path = tmp_path / "img.png"
            Image.new("RGB", (width, height)).save(path)
            return ImageGenerationResult(
                image_path=str(path), provider="_SeedReportingProvider", prompt=prompt,
                width=width, height=height, generation_time_seconds=0.0,
                metadata={"seed": 424242},
            )

    registry = ImageProviderRegistry()
    registry.register("mock", lambda: _SeedReportingProvider())
    render_pipeline = RenderPipeline(
        image_router=ImageRouter(registry=registry), asset_manager=AssetManager(storage_root=str(tmp_path / "storage"))
    )

    frame, _prompt = asyncio.run(_render_one_sample_shot(render_pipeline, tmp_path))
    assert frame.metadata.generation_seed == 424242


def test_frame_metadata_falls_back_when_provider_reports_no_seed(tmp_path):
    class _NoSeedProvider(ImageProvider):
        async def generate_image(self, prompt, width=1024, height=1024):
            path = tmp_path / "img2.png"
            Image.new("RGB", (width, height)).save(path)
            return ImageGenerationResult(
                image_path=str(path), provider="_NoSeedProvider", prompt=prompt,
                width=width, height=height, generation_time_seconds=0.0,
                metadata={"no_seed_here": True},
            )

    registry = ImageProviderRegistry()
    registry.register("mock", lambda: _NoSeedProvider())
    render_pipeline = RenderPipeline(
        image_router=ImageRouter(registry=registry), asset_manager=AssetManager(storage_root=str(tmp_path / "storage2"))
    )

    frame, actual_prompt = asyncio.run(_render_one_sample_shot(render_pipeline, tmp_path))
    assert frame.metadata.generation_seed == _local_derive_seed(actual_prompt)


def _local_derive_seed(prompt: str) -> int:
    """Mirrors render_pipeline.py's own _derive_seed exactly, for
    fallback-value assertions without importing a private symbol
    across modules unnecessarily."""
    import hashlib

    return int(hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:8], 16)


async def _render_one_sample_shot(render_pipeline, tmp_path):
    from core.ai_router import TemplateProvider
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

    analysis = analyze_game(_sample_pgn())
    combat = generate_combat_intelligence(analysis)
    battle = generate_battle_intelligence(analysis, combat)
    profile = generate_style_profile(battle, combat, "anime")
    battle_mode = generate_battle_mode_intelligence(BattleMode.DUEL, combat, battle)
    story = await NarrativeGenerator(TemplateProvider()).generate(analysis, combat, battle, profile, battle_mode)
    timeline = generate_shot_timeline(battle, story)
    composed = compose_scene(timeline, battle, profile, battle_mode)
    prompted = generate_prompts(composed)
    prompted.shots = prompted.shots[:1]  # exactly one shot, for a focused test

    output = await render_pipeline.render(prompted, "test_fight")
    return output.frames[0], prompted.shots[0].image_prompt


# --- 9/10/11. Shared-policy plan and execution seed agreement --------------


def _acceptance_runner_with_seed_reporting_provider(tmp_path, fixed_seed):
    """A fake image provider that DOES respect a fixed seed and
    correctly reports it via metadata — simulating a correctly
    configured ComfyUIImageProvider(seed_override=...)."""

    class _SharedSeedProvider(ImageProvider):
        async def generate_image(self, prompt, width=1024, height=1024):
            path = tmp_path / f"img_{abs(hash(prompt))}.png"
            Image.new("RGB", (width, height)).save(path)
            return ImageGenerationResult(
                image_path=str(path), provider="_SharedSeedProvider", prompt=prompt,
                width=width, height=height, generation_time_seconds=0.0,
                metadata={"seed": fixed_seed},
            )

    from core.animation_router import AnimationProviderRegistry, AnimationRouter, MockAnimationProvider

    image_registry = ImageProviderRegistry()
    image_registry.register("mock", lambda: _SharedSeedProvider())
    render_pipeline = RenderPipeline(
        image_router=ImageRouter(registry=image_registry), asset_manager=AssetManager(storage_root=str(tmp_path / "storage"))
    )
    anim_registry = AnimationProviderRegistry()
    anim_registry.register("mock", lambda: MockAnimationProvider(output_dir=str(tmp_path / "anim")))
    animation_pipeline = AnimationPipeline(animation_router=AnimationRouter(registry=anim_registry))

    from core.ai_router import TemplateProvider

    return MultiShotAcceptanceRunner(
        TemplateProvider(), render_pipeline=render_pipeline, animation_pipeline=animation_pipeline,
        video_builder=VideoBuilder(), asset_manager=AssetManager(storage_root=str(tmp_path / "storage")),
    )


def test_shared_policy_plan_contains_three_identical_planned_seeds(tmp_path):
    from products.chess2fight.rendering.multi_shot_acceptance import MultiShotAcceptanceRunner as _Runner
    from core.ai_router import TemplateProvider

    runner = _Runner(TemplateProvider())
    plan = asyncio.run(
        runner.prepare(_sample_pgn(), _preferences(), max_animation_seconds=2.0, visual_seed_policy=VisualSeedPolicy.SHARED)
    )
    assert len(set(plan.resolved_flux_seeds)) == 1
    assert len(plan.resolved_flux_seeds) == 3


def test_shared_policy_execution_returns_three_identical_actual_seeds_matching_base(tmp_path):
    from products.chess2fight.rendering.multi_shot_acceptance import MultiShotAcceptanceRunner as _PlanRunner
    from core.ai_router import TemplateProvider

    plan_runner = _PlanRunner(TemplateProvider())
    plan = asyncio.run(
        plan_runner.prepare(_sample_pgn(), _preferences(), max_animation_seconds=2.0, visual_seed_policy=VisualSeedPolicy.SHARED)
    )
    base_seed = plan.fight_base_visual_seed

    runner = _acceptance_runner_with_seed_reporting_provider(tmp_path, base_seed)
    result = asyncio.run(runner.execute(plan, width=256, height=256))

    assert len(set(result.actual_flux_seeds)) == 1
    assert result.actual_flux_seeds[0] == base_seed
    assert all(seed == base_seed for seed in result.actual_flux_seeds)


# --- 12/13. Manifest distinguishes planned vs actual, values agree ---------


def test_manifest_distinguishes_and_agrees_on_planned_vs_actual(tmp_path):
    import json
    import subprocess
    import sys

    manifest_path = str(tmp_path / "manifest.json")
    subprocess.run(
        [sys.executable, "scripts/render_multi_shot_acceptance.py", "--sample", "--manifest-path", manifest_path],
        capture_output=True, text=True, check=True,
    )
    manifest = json.loads(open(manifest_path).read())
    for shot_entry in manifest["shots"]:
        assert "planned_flux_seed" in shot_entry
        assert "actual_flux_seed" in shot_entry
        assert shot_entry["planned_flux_seed"] == shot_entry["actual_flux_seed"]  # default policy: must agree

    import shutil

    shutil.rmtree("generated_animations", ignore_errors=True)
    shutil.rmtree("generated_images", ignore_errors=True)
    shutil.rmtree("storage", ignore_errors=True)


# --- 14/15. Mismatch fails before Wan, zero Wan calls ------------------------


def test_seed_mismatch_raises_before_any_wan_call(tmp_path):
    """Using a provider that reports a DIFFERENT seed than what was
    planned — simulating a misconfigured or non-seed-respecting
    provider under a non-default policy."""
    from core.animation_router import AnimationProvider, AnimationProviderRegistry, AnimationRouter

    wan_calls = []

    class _ExplodingAnimationProvider(AnimationProvider):
        async def generate_animation(self, instruction):
            wan_calls.append(instruction.shot_id)
            raise AssertionError("Wan must never be called after a seed mismatch")

    class _DisagreeingProvider(ImageProvider):
        async def generate_image(self, prompt, width=1024, height=1024):
            path = tmp_path / f"img_{abs(hash(prompt))}.png"
            Image.new("RGB", (width, height)).save(path)
            return ImageGenerationResult(
                image_path=str(path), provider="_DisagreeingProvider", prompt=prompt,
                width=width, height=height, generation_time_seconds=0.0,
                metadata={"seed": 999999999},  # deliberately NOT the planned shared seed
            )

    image_registry = ImageProviderRegistry()
    image_registry.register("mock", lambda: _DisagreeingProvider())
    render_pipeline = RenderPipeline(
        image_router=ImageRouter(registry=image_registry), asset_manager=AssetManager(storage_root=str(tmp_path / "storage"))
    )
    anim_registry = AnimationProviderRegistry()
    anim_registry.register("mock", lambda: _ExplodingAnimationProvider())
    animation_pipeline = AnimationPipeline(animation_router=AnimationRouter(registry=anim_registry))

    from core.ai_router import TemplateProvider

    runner = MultiShotAcceptanceRunner(
        TemplateProvider(), render_pipeline=render_pipeline, animation_pipeline=animation_pipeline,
        video_builder=VideoBuilder(), asset_manager=AssetManager(storage_root=str(tmp_path / "storage")),
    )
    plan = asyncio.run(
        runner.prepare(_sample_pgn(), _preferences(), max_animation_seconds=2.0, visual_seed_policy=VisualSeedPolicy.SHARED)
    )

    with pytest.raises(SeedEvidenceMismatchError):
        asyncio.run(runner.execute(plan, width=256, height=256))

    assert len(wan_calls) == 0


def test_default_policy_never_raises_seed_mismatch_even_with_disagreeing_provider(tmp_path):
    """DEFAULT policy makes no planned-vs-actual claim at all, so this
    validation must never fire under it — confirms the check is
    correctly scoped to non-default policies only."""
    class _AnySeedProvider(ImageProvider):
        async def generate_image(self, prompt, width=1024, height=1024):
            path = tmp_path / f"img_{abs(hash(prompt))}.png"
            Image.new("RGB", (width, height)).save(path)
            return ImageGenerationResult(
                image_path=str(path), provider="_AnySeedProvider", prompt=prompt,
                width=width, height=height, generation_time_seconds=0.0,
                metadata={"seed": 12345},  # unrelated to whatever _derive_seed(prompt) would give
            )

    from core.animation_router import AnimationProviderRegistry, AnimationRouter, MockAnimationProvider

    image_registry = ImageProviderRegistry()
    image_registry.register("mock", lambda: _AnySeedProvider())
    render_pipeline = RenderPipeline(
        image_router=ImageRouter(registry=image_registry), asset_manager=AssetManager(storage_root=str(tmp_path / "storage"))
    )
    anim_registry = AnimationProviderRegistry()
    anim_registry.register("mock", lambda: MockAnimationProvider(output_dir=str(tmp_path / "anim")))
    animation_pipeline = AnimationPipeline(animation_router=AnimationRouter(registry=anim_registry))

    from core.ai_router import TemplateProvider

    runner = MultiShotAcceptanceRunner(
        TemplateProvider(), render_pipeline=render_pipeline, animation_pipeline=animation_pipeline,
        video_builder=VideoBuilder(), asset_manager=AssetManager(storage_root=str(tmp_path / "storage")),
    )
    plan = asyncio.run(runner.prepare(_sample_pgn(), _preferences(), max_animation_seconds=2.0))  # default policy
    result = asyncio.run(runner.execute(plan, width=256, height=256))  # must NOT raise
    assert result.actual_flux_seeds == [12345, 12345, 12345]


# --- 16. Generation count remains exactly 3 FLUX + 3 Wan on success ---------


def test_six_job_expectation_unaffected_by_seed_evidence_changes(tmp_path):
    from core.ai_router import TemplateProvider

    plan = asyncio.run(
        MultiShotAcceptanceRunner(TemplateProvider()).prepare(
            _sample_pgn(), _preferences(), max_animation_seconds=2.0, visual_seed_policy=VisualSeedPolicy.SHARED
        )
    )
    assert plan.expected_comfyui_job_count == 6


# --- 17/18. Backward compatibility -------------------------------------------


def test_default_policy_dry_run_unaffected(tmp_path):
    from core.ai_router import TemplateProvider

    runner = MultiShotAcceptanceRunner(TemplateProvider())
    plan = asyncio.run(runner.prepare(_sample_pgn(), _preferences(), max_animation_seconds=2.0))
    assert plan.visual_seed_policy == "default"
    assert plan.fight_base_visual_seed is None


def test_generic_image_provider_signature_still_unchanged():
    import inspect

    sig = inspect.signature(ImageProvider.generate_image)
    assert list(sig.parameters.keys()) == ["self", "prompt", "width", "height"]


# --- 19/20. No real ComfyUI contact; live tests stay gated -------------------


def test_seed_evidence_module_makes_no_network_calls():
    import products.chess2fight.rendering.visual_continuity as module

    with open(module.__file__) as f:
        source = f.read()
    assert "httpx" not in source
    assert "requests" not in source
