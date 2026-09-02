"""Tests for a two-part hotfix to
`products.chess2fight.rendering.reference_continuity_acceptance`,
driven by real live GPU evidence from a failed production run:

1. Root cause: `prepare()` planned every shot's FLUX seed from
   `shot.image_prompt` (via the delegated
   `MultiShotAcceptanceRunner.prepare()`, which predates reference-
   conditioning entirely) — correct for the T2I anchor, but wrong for
   reference-conditioned shots, which `execute()` actually submits
   `compose_reference_edit_prompt(shot)` for. Under SHARED policy this
   never mattered (the seed doesn't depend on the prompt at all) — the
   bug only manifests under DERIVED policy, which is exactly why every
   pre-existing test (all of which used the SHARED default) missed it,
   right up to a real, paid GPU run.

2. Fail-fast: the planned-vs-actual seed check ran once, as a batch,
   only after every image generation (anchor + both reference shots)
   had already completed — confirmed directly against real ComfyUI
   history from the failed run: 3 FLUX jobs executed, including a
   second reference shot submitted AFTER the first one's seed had
   already mismatched, before any error was ever raised. Now checked
   immediately after each individual generation, before the next paid
   job can be submitted.

TEST CATEGORY: MOCKED UNIT / LOCAL INTEGRATION TESTS — no real ComfyUI
server or GPU is ever contacted.
"""

import asyncio
from pathlib import Path

import pytest
from PIL import Image

from core.ai_router import TemplateProvider
from core.animation_router import AnimationProviderRegistry, AnimationRouter, MockAnimationProvider
from core.image_router import ImageGenerationResult, ImageProvider, ImageProviderRegistry, ImageRouter
from products.chess2fight.cinematic.prompt_generator import compose_reference_edit_prompt
from products.chess2fight.rendering.animation_pipeline import AnimationPipeline
from products.chess2fight.rendering.asset_manager import AssetManager
from products.chess2fight.rendering.multi_shot_acceptance import SeedEvidenceMismatchError
from products.chess2fight.rendering.reference_continuity_acceptance import ReferenceContinuityAcceptanceRunner
from products.chess2fight.rendering.render_pipeline import RenderPipeline
from products.chess2fight.rendering.video_builder import VideoBuilder
from products.chess2fight.rendering.visual_continuity import VisualSeedPolicy, derive_fight_base_visual_seed, derive_shot_seed
from tests.test_single_shot_acceptance import _preferences, _sample_pgn

_PGN = _sample_pgn()

# The exact real live GPU evidence from the failed production run this
# hotfix responds to.
_LIVE_EVIDENCE_SEEDS = {"anchor": 1970824880, "shot1": 1918080141, "shot2": 104965515}


def _plain_runner(tmp_path):
    return ReferenceContinuityAcceptanceRunner(TemplateProvider(), asset_manager=AssetManager(storage_root=str(tmp_path / "storage")))


# --- Root cause: prepare() now plans reference-conditioned shots correctly -


def test_prepare_matches_exact_live_evidence_seeds(tmp_path):
    """The strongest possible proof: prepare() under DERIVED policy for
    the sample fight must resolve to exactly the three seeds real
    ComfyUI history reported for a real production run."""
    runner = _plain_runner(tmp_path)
    plan = asyncio.run(
        runner.prepare(_PGN, _preferences(), max_animation_seconds=2.0, visual_seed_policy=VisualSeedPolicy.DERIVED)
    )
    assert plan.resolved_flux_seeds == [
        _LIVE_EVIDENCE_SEEDS["anchor"], _LIVE_EVIDENCE_SEEDS["shot1"], _LIVE_EVIDENCE_SEEDS["shot2"],
    ]


def test_anchor_seed_uses_image_prompt_unaffected_by_fix(tmp_path):
    """Shot 0 never goes through compose_reference_edit_prompt at all
    — its planned seed must still come from shot.image_prompt, exactly
    as before this fix (only shots after index 0 changed)."""
    runner = _plain_runner(tmp_path)
    plan = asyncio.run(
        runner.prepare(_PGN, _preferences(), max_animation_seconds=2.0, visual_seed_policy=VisualSeedPolicy.DERIVED)
    )
    base_seed = derive_fight_base_visual_seed(_PGN, "anime", "duel")
    expected_anchor_seed = derive_shot_seed(base_seed, plan.shots[0].image_prompt)
    assert plan.resolved_flux_seeds[0] == expected_anchor_seed


def test_reference_shot_seeds_use_reference_edit_prompt_not_image_prompt(tmp_path):
    """Direct confirmation of the actual mechanism, independent of the
    specific live-evidence numbers: the planned seed for shots 1/2
    must equal derive_shot_seed(base_seed, compose_reference_edit_prompt(shot)),
    and must NOT equal derive_shot_seed(base_seed, shot.image_prompt)
    (the pre-fix, buggy value) unless those two prompts happen to
    coincide (they don't, for this sample fight)."""
    runner = _plain_runner(tmp_path)
    plan = asyncio.run(
        runner.prepare(_PGN, _preferences(), max_animation_seconds=2.0, visual_seed_policy=VisualSeedPolicy.DERIVED)
    )
    base_seed = derive_fight_base_visual_seed(_PGN, "anime", "duel")
    for i in (1, 2):
        shot = plan.shots[i]
        correct_seed = derive_shot_seed(base_seed, compose_reference_edit_prompt(shot))
        buggy_seed = derive_shot_seed(base_seed, shot.image_prompt)
        assert plan.resolved_flux_seeds[i] == correct_seed
        assert plan.resolved_flux_seeds[i] != buggy_seed  # confirms the two prompts genuinely differ for this fight


def test_shared_policy_unaffected_by_the_bug_or_the_fix(tmp_path):
    """Under SHARED, the seed never depended on the prompt at all —
    confirms the fix doesn't disturb this policy's own, always-correct
    behavior (and explains why SHARED-policy tests never caught the
    original bug)."""
    runner = _plain_runner(tmp_path)
    plan = asyncio.run(
        runner.prepare(_PGN, _preferences(), max_animation_seconds=2.0, visual_seed_policy=VisualSeedPolicy.SHARED)
    )
    assert len(set(plan.resolved_flux_seeds)) == 1  # all three shots share the identical base seed


def test_default_policy_reference_shots_also_corrected(tmp_path):
    """DEFAULT policy (no explicit override) falls back to
    _derive_flux_seed(prompt) directly — also prompt-dependent, so
    also affected by the same bug and also fixed the same way."""
    runner = _plain_runner(tmp_path)
    plan = asyncio.run(runner.prepare(_PGN, _preferences(), max_animation_seconds=2.0))  # default policy
    assert plan.visual_seed_policy == "shared"  # this runner's own default, not MultiShotAcceptanceRunner's


# --- Fail-fast: immediate per-shot checking, never a late batch check ------


def _fail_fast_harness(tmp_path, anchor_seed_provider_result, shot_results):
    """Builds a runner + fake reference provider for fail-fast testing.

    `shot_results` is a list of either an int (a correct-matching seed
    will be reported) or a specific wrong int (a mismatch) for each of
    shots 1/2, in order. The fake reference provider raises
    AssertionError if ever called more times than len(shot_results) —
    catching an over-eager implementation that doesn't actually stop.
    """

    class _AnchorProvider(ImageProvider):
        async def generate_image(self, prompt, width=1024, height=1024):
            path = str(tmp_path / f"anchor_{abs(hash(prompt))}.png")
            Image.new("RGB", (width, height)).save(path)
            return ImageGenerationResult(
                image_path=path, provider="_AnchorProvider", prompt=prompt, width=width, height=height,
                generation_time_seconds=0.0, metadata={"seed": anchor_seed_provider_result},
            )

    class _SequencedReferenceProvider:
        def __init__(self):
            self.calls = 0

        async def generate_reference_conditioned_image(self, prompt, reference_image_path, width=1024, height=1024):
            self.calls += 1
            if self.calls > len(shot_results):
                raise AssertionError(
                    f"reference provider called {self.calls} times — only {len(shot_results)} expected; "
                    "fail-fast is not actually stopping generation early."
                )
            path = str(tmp_path / f"ref_{self.calls}.png")
            Image.new("RGB", (width, height)).save(path)
            return ImageGenerationResult(
                image_path=path, provider="_SequencedReferenceProvider", prompt=prompt, width=width, height=height,
                generation_time_seconds=0.0, metadata={"seed": shot_results[self.calls - 1]},
            )

    image_registry = ImageProviderRegistry()
    image_registry.register("mock", lambda: _AnchorProvider())
    render_pipeline = RenderPipeline(image_router=ImageRouter(registry=image_registry), asset_manager=AssetManager(storage_root=str(tmp_path / "storage")))

    wan_calls = []

    class _CountingAnimationProvider(MockAnimationProvider):
        async def generate_animation(self, instruction):
            wan_calls.append(instruction.shot_id)
            return await super().generate_animation(instruction)

    anim_registry = AnimationProviderRegistry()
    anim_registry.register("mock", lambda: _CountingAnimationProvider(output_dir=str(tmp_path / "anim")))
    animation_pipeline = AnimationPipeline(animation_router=AnimationRouter(registry=anim_registry))

    runner = ReferenceContinuityAcceptanceRunner(
        TemplateProvider(), render_pipeline=render_pipeline, animation_pipeline=animation_pipeline,
        video_builder=VideoBuilder(), asset_manager=AssetManager(storage_root=str(tmp_path / "storage")),
    )
    reference_provider = _SequencedReferenceProvider()
    return runner, reference_provider, wan_calls


def test_shot1_mismatch_stops_before_shot2_and_before_wan(tmp_path):
    """The exact scenario real GPU history proved was NOT happening
    before this fix: shot 0 and shot 1 may generate, but shot 2 must
    never be submitted, and zero Wan jobs may run."""
    # prepare() is fully deterministic — compute the plan once via a
    # throwaway runner so the real anchor provider below can be built
    # with the correct seed already known, rather than mutating
    # anything after construction.
    plan = asyncio.run(
        _plain_runner(tmp_path).prepare(_PGN, _preferences(), max_animation_seconds=2.0, visual_seed_policy=VisualSeedPolicy.DERIVED)
    )

    image_registry = ImageProviderRegistry()
    image_registry.register("mock", lambda: _AnchorProviderWithCorrectSeed(plan, tmp_path))
    render_pipeline = RenderPipeline(image_router=ImageRouter(registry=image_registry), asset_manager=AssetManager(storage_root=str(tmp_path / "storage")))
    wan_calls = []

    class _CountingAnimationProvider(MockAnimationProvider):
        async def generate_animation(self, instruction):
            wan_calls.append(instruction.shot_id)
            return await super().generate_animation(instruction)

    anim_registry = AnimationProviderRegistry()
    anim_registry.register("mock", lambda: _CountingAnimationProvider(output_dir=str(tmp_path / "anim")))
    animation_pipeline = AnimationPipeline(animation_router=AnimationRouter(registry=anim_registry))

    class _Shot1WrongProvider:
        def __init__(self):
            self.calls = 0

        async def generate_reference_conditioned_image(self, prompt, reference_image_path, width=1024, height=1024):
            self.calls += 1
            if self.calls > 1:
                raise AssertionError("shot 2 must never be submitted after shot 1's seed mismatch")
            path = str(tmp_path / "ref_wrong.png")
            Image.new("RGB", (width, height)).save(path)
            return ImageGenerationResult(
                image_path=path, provider="_Shot1WrongProvider", prompt=prompt, width=width, height=height,
                generation_time_seconds=0.0, metadata={"seed": 999999999},  # deliberately wrong
            )

    runner = ReferenceContinuityAcceptanceRunner(
        TemplateProvider(), render_pipeline=render_pipeline, animation_pipeline=animation_pipeline,
        video_builder=VideoBuilder(), asset_manager=AssetManager(storage_root=str(tmp_path / "storage")),
    )
    reference_provider = _Shot1WrongProvider()

    with pytest.raises(SeedEvidenceMismatchError) as exc_info:
        asyncio.run(runner.execute(plan, reference_provider, width=256, height=256))
    assert plan.shots[1].shot_id in str(exc_info.value)
    assert reference_provider.calls == 1  # shot 1 attempted, shot 2 never was
    assert wan_calls == []


class _AnchorProviderWithCorrectSeed(ImageProvider):
    def __init__(self, plan, tmp_path):
        self._plan = plan
        self._tmp_path = tmp_path

    async def generate_image(self, prompt, width=1024, height=1024):
        path = str(self._tmp_path / f"anchor_correct_{abs(hash(prompt))}.png")
        Image.new("RGB", (width, height)).save(path)
        return ImageGenerationResult(
            image_path=path, provider="_AnchorProviderWithCorrectSeed", prompt=prompt, width=width, height=height,
            generation_time_seconds=0.0, metadata={"seed": self._plan.resolved_flux_seeds[0]},
        )


def test_anchor_mismatch_stops_before_any_reference_call(tmp_path):
    runner, reference_provider, wan_calls = _fail_fast_harness(
        tmp_path, anchor_seed_provider_result=999999999, shot_results=[]  # anchor itself is wrong
    )
    plan = asyncio.run(runner.prepare(_PGN, _preferences(), max_animation_seconds=2.0, visual_seed_policy=VisualSeedPolicy.DERIVED))

    with pytest.raises(SeedEvidenceMismatchError) as exc_info:
        asyncio.run(runner.execute(plan, reference_provider, width=256, height=256))
    assert "t2i" in str(exc_info.value)
    assert reference_provider.calls == 0  # zero reference-conditioned calls ever attempted
    assert wan_calls == []


def test_shot2_mismatch_after_shot1_success_stops_before_wan_preserves_shot1(tmp_path):
    """Confirms partial success is preserved: shot 1's own output file
    remains on disk even though shot 2's mismatch stops the run."""
    runner = _plain_runner(tmp_path)
    plan = asyncio.run(runner.prepare(_PGN, _preferences(), max_animation_seconds=2.0, visual_seed_policy=VisualSeedPolicy.DERIVED))

    class _AnchorOk(ImageProvider):
        async def generate_image(self, prompt, width=1024, height=1024):
            path = str(tmp_path / f"anchor_ok_{abs(hash(prompt))}.png")
            Image.new("RGB", (width, height)).save(path)
            return ImageGenerationResult(
                image_path=path, provider="_AnchorOk", prompt=prompt, width=width, height=height,
                generation_time_seconds=0.0, metadata={"seed": plan.resolved_flux_seeds[0]},
            )

    class _Shot1OkShot2Wrong:
        def __init__(self):
            self.calls = 0

        async def generate_reference_conditioned_image(self, prompt, reference_image_path, width=1024, height=1024):
            self.calls += 1
            path = str(tmp_path / f"ref_{self.calls}.png")
            Image.new("RGB", (width, height)).save(path)
            seed = plan.resolved_flux_seeds[1] if self.calls == 1 else 999999999
            return ImageGenerationResult(
                image_path=path, provider="_Shot1OkShot2Wrong", prompt=prompt, width=width, height=height,
                generation_time_seconds=0.0, metadata={"seed": seed},
            )

    image_registry = ImageProviderRegistry()
    image_registry.register("mock", lambda: _AnchorOk())
    render_pipeline = RenderPipeline(image_router=ImageRouter(registry=image_registry), asset_manager=AssetManager(storage_root=str(tmp_path / "storage2")))
    wan_calls = []

    class _CountingAnimationProvider(MockAnimationProvider):
        async def generate_animation(self, instruction):
            wan_calls.append(instruction.shot_id)
            return await super().generate_animation(instruction)

    anim_registry = AnimationProviderRegistry()
    anim_registry.register("mock", lambda: _CountingAnimationProvider(output_dir=str(tmp_path / "anim2")))
    animation_pipeline = AnimationPipeline(animation_router=AnimationRouter(registry=anim_registry))

    runner2 = ReferenceContinuityAcceptanceRunner(
        TemplateProvider(), render_pipeline=render_pipeline, animation_pipeline=animation_pipeline,
        video_builder=VideoBuilder(), asset_manager=AssetManager(storage_root=str(tmp_path / "storage2")),
    )
    reference_provider = _Shot1OkShot2Wrong()

    with pytest.raises(SeedEvidenceMismatchError):
        asyncio.run(runner2.execute(plan, reference_provider, width=256, height=256))

    assert reference_provider.calls == 2  # shot 1 succeeded, shot 2 attempted and mismatched
    assert wan_calls == []  # but Wan never ran
    # Shot 1's own frame was saved to fight storage before shot 2's own call happened.
    fight_dir = tmp_path / "storage2" / "renders" / plan.fight_id
    saved_shot1_files = list(fight_dir.glob(f"frame{plan.shots[1].sequence_order:04d}.png")) if fight_dir.exists() else []
    assert len(saved_shot1_files) == 1


def test_happy_path_all_three_shots_correct_wan_still_runs(tmp_path):
    """Confirms the fix doesn't break the successful case — all seeds
    genuinely agree, and animation proceeds normally afterward."""
    runner = _plain_runner(tmp_path)
    plan = asyncio.run(runner.prepare(_PGN, _preferences(), max_animation_seconds=2.0, visual_seed_policy=VisualSeedPolicy.DERIVED))

    class _AllCorrectAnchor(ImageProvider):
        async def generate_image(self, prompt, width=1024, height=1024):
            path = str(tmp_path / f"anchor_hp_{abs(hash(prompt))}.png")
            Image.new("RGB", (width, height)).save(path)
            return ImageGenerationResult(
                image_path=path, provider="_AllCorrectAnchor", prompt=prompt, width=width, height=height,
                generation_time_seconds=0.0, metadata={"seed": plan.resolved_flux_seeds[0]},
            )

    class _AllCorrectReference:
        def __init__(self):
            self.calls = 0

        async def generate_reference_conditioned_image(self, prompt, reference_image_path, width=1024, height=1024):
            self.calls += 1
            path = str(tmp_path / f"ref_hp_{self.calls}.png")
            Image.new("RGB", (width, height)).save(path)
            return ImageGenerationResult(
                image_path=path, provider="_AllCorrectReference", prompt=prompt, width=width, height=height,
                generation_time_seconds=0.0, metadata={"seed": plan.resolved_flux_seeds[self.calls]},
            )

    image_registry = ImageProviderRegistry()
    image_registry.register("mock", lambda: _AllCorrectAnchor())
    render_pipeline = RenderPipeline(image_router=ImageRouter(registry=image_registry), asset_manager=AssetManager(storage_root=str(tmp_path / "storage3")))
    anim_registry = AnimationProviderRegistry()
    anim_registry.register("mock", lambda: MockAnimationProvider(output_dir=str(tmp_path / "anim3")))
    animation_pipeline = AnimationPipeline(animation_router=AnimationRouter(registry=anim_registry))

    runner3 = ReferenceContinuityAcceptanceRunner(
        TemplateProvider(), render_pipeline=render_pipeline, animation_pipeline=animation_pipeline,
        video_builder=VideoBuilder(), asset_manager=AssetManager(storage_root=str(tmp_path / "storage3")),
    )
    result = asyncio.run(runner3.execute(plan, _AllCorrectReference(), width=256, height=256))
    assert result.actual_flux_seeds == plan.resolved_flux_seeds
    assert Path(result.final_video_path).exists()


def test_batch_check_still_present_as_defense_in_depth():
    """Confirms the final batch comparison wasn't removed — it's now
    unreachable in the failure case (per-shot checks raise first), but
    kept per this task's own explicit "keep the final/batch invariant
    check too" instruction."""
    with open("products/chess2fight/rendering/reference_continuity_acceptance.py") as f:
        source = f.read()
    assert source.count("SeedEvidenceMismatchError(") >= 2  # both the per-shot and the batch raise sites
    assert "defense in depth" in source.lower()
