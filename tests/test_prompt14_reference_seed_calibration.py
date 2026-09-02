"""Tests for Sprint 4 Prompt 14's reference-conditioning seed
calibration experiment.

TEST CATEGORY: MOCKED UNIT / LOCAL INTEGRATION TESTS — no real ComfyUI
server or GPU is ever contacted.
"""

import asyncio
import hashlib
import subprocess
import sys

import pytest
from PIL import Image

from core.ai_router import TemplateProvider
from core.exceptions import ImageProviderError
from core.image_router import ImageGenerationResult
from products.chess2fight.rendering.asset_manager import AssetManager
from products.chess2fight.rendering.multi_shot_acceptance import SeedEvidenceMismatchError
from products.chess2fight.rendering.reference_seed_calibration import (
    AnchorValidationError,
    ReferenceSeedCalibrationRunner,
)
from products.chess2fight.rendering.visual_continuity import VisualSeedPolicy, build_seed_override
from tests.test_single_shot_acceptance import _preferences, _sample_pgn

RENDER_CALIBRATION_CLI = "scripts/render_reference_seed_calibration.py"


def _make_anchor(path: str, width: int = 1280, height: int = 704) -> None:
    Image.new("RGB", (width, height), color=(50, 100, 150)).save(path)


class _CorrectFakeReferenceProvider:
    """A fake reference provider that correctly reports whichever seed
    the caller's seed_override would produce for each prompt — a
    genuinely correctly-wired provider, used for happy-path tests."""

    def __init__(self, base_seed: int):
        self._override = build_seed_override(VisualSeedPolicy.DERIVED, base_seed)
        self.calls: list[tuple[str, str]] = []

    async def generate_reference_conditioned_image(self, prompt, reference_image_path, width=1024, height=1024):
        self.calls.append((prompt, reference_image_path))
        path = f"/tmp/calib_correct_{len(self.calls)}_{abs(hash(prompt))}.png"
        Image.new("RGB", (width, height)).save(path)
        return ImageGenerationResult(
            image_path=path, provider="_CorrectFakeReferenceProvider", prompt=prompt,
            width=width, height=height, generation_time_seconds=0.0,
            metadata={"seed": self._override(prompt)},
        )


class _DisagreeingReferenceProvider:
    """Reports a fixed, unrelated seed regardless of the planned one."""

    def __init__(self, fixed_seed: int, fail_on_call: int | None = None):
        self._fixed_seed = fixed_seed
        self.calls: list[tuple[str, str]] = []
        self._fail_on_call = fail_on_call

    async def generate_reference_conditioned_image(self, prompt, reference_image_path, width=1024, height=1024):
        self.calls.append((prompt, reference_image_path))
        if self._fail_on_call is not None and len(self.calls) == self._fail_on_call:
            raise ImageProviderError("simulated failure")
        path = f"/tmp/calib_disagree_{len(self.calls)}.png"
        Image.new("RGB", (width, height)).save(path)
        return ImageGenerationResult(
            image_path=path, provider="_DisagreeingReferenceProvider", prompt=prompt,
            width=width, height=height, generation_time_seconds=0.0,
            metadata={"seed": self._fixed_seed},
        )


def _runner(tmp_path):
    return ReferenceSeedCalibrationRunner(TemplateProvider(), asset_manager=AssetManager(storage_root=str(tmp_path / "storage")))


# --- 1. Supplied anchor is never generated -----------------------------------


def test_supplied_anchor_is_never_generated(tmp_path):
    anchor_path = str(tmp_path / "anchor.png")
    _make_anchor(anchor_path)
    original_bytes = open(anchor_path, "rb").read()

    runner = _runner(tmp_path)
    plan = asyncio.run(runner.prepare(_sample_pgn(), _preferences(), anchor_path=anchor_path, anchor_original_seed=42, style="anime", battle_mode="duel"))
    provider = _CorrectFakeReferenceProvider(plan.fight_base_visual_seed)
    asyncio.run(runner.execute(plan, provider))

    # The anchor file itself is byte-for-byte unchanged.
    assert open(anchor_path, "rb").read() == original_bytes


# --- 2/3. Exactly two image-provider calls, both reference-conditioned ------


def test_exactly_two_image_provider_calls(tmp_path):
    anchor_path = str(tmp_path / "anchor.png")
    _make_anchor(anchor_path)
    runner = _runner(tmp_path)
    plan = asyncio.run(runner.prepare(_sample_pgn(), _preferences(), anchor_path=anchor_path, anchor_original_seed=42, style="anime", battle_mode="duel"))
    provider = _CorrectFakeReferenceProvider(plan.fight_base_visual_seed)
    asyncio.run(runner.execute(plan, provider))
    assert len(provider.calls) == 2


def test_both_calls_are_reference_conditioned():
    """The provider interface used has no T2I method at all — calling
    it is inherently reference-conditioned; confirmed by the fact the
    fake provider class only implements generate_reference_conditioned_image."""
    assert hasattr(_CorrectFakeReferenceProvider, "generate_reference_conditioned_image")
    assert not hasattr(_CorrectFakeReferenceProvider, "generate_image")


# --- 4/5. Zero Wan calls, zero VideoBuilder calls ----------------------------


def test_zero_wan_or_videobuilder_calls(tmp_path):
    """The calibration module imports neither AnimationPipeline nor
    VideoBuilder at all — confirmed directly against its own source.
    Checks for actual import statements, not the bare word (which
    legitimately appears in the module's own docstring, explaining
    what this experiment deliberately doesn't do)."""
    import products.chess2fight.rendering.reference_seed_calibration as module

    with open(module.__file__) as f:
        source = f.read()
    assert "import AnimationPipeline" not in source
    assert "import VideoBuilder" not in source
    assert "import AnimationRouter" not in source


# --- 6/7/8. Same anchor, same SHA256, no chaining ---------------------------


def test_both_jobs_reference_exactly_the_same_anchor(tmp_path):
    anchor_path = str(tmp_path / "anchor.png")
    _make_anchor(anchor_path)
    runner = _runner(tmp_path)
    plan = asyncio.run(runner.prepare(_sample_pgn(), _preferences(), anchor_path=anchor_path, anchor_original_seed=42, style="anime", battle_mode="duel"))
    provider = _CorrectFakeReferenceProvider(plan.fight_base_visual_seed)
    asyncio.run(runner.execute(plan, provider))
    assert provider.calls[0][1] == provider.calls[1][1] == anchor_path


def test_anchor_sha256_identical_for_both(tmp_path):
    anchor_path = str(tmp_path / "anchor.png")
    _make_anchor(anchor_path)
    runner = _runner(tmp_path)
    plan = asyncio.run(runner.prepare(_sample_pgn(), _preferences(), anchor_path=anchor_path, anchor_original_seed=42, style="anime", battle_mode="duel"))
    provider = _CorrectFakeReferenceProvider(plan.fight_base_visual_seed)
    result = asyncio.run(runner.execute(plan, provider))
    assert result.shot_results[0].reference_anchor_sha256 == result.shot_results[1].reference_anchor_sha256
    expected_sha256 = hashlib.sha256(open(anchor_path, "rb").read()).hexdigest()
    assert result.shot_results[0].reference_anchor_sha256 == expected_sha256


def test_shot_1_never_becomes_shot_2_reference(tmp_path):
    anchor_path = str(tmp_path / "anchor.png")
    _make_anchor(anchor_path)
    runner = _runner(tmp_path)
    plan = asyncio.run(runner.prepare(_sample_pgn(), _preferences(), anchor_path=anchor_path, anchor_original_seed=42, style="anime", battle_mode="duel"))
    provider = _CorrectFakeReferenceProvider(plan.fight_base_visual_seed)
    result = asyncio.run(runner.execute(plan, provider))
    shot1_output = result.shot_results[0].output_path
    shot2_reference = provider.calls[1][1]
    assert shot2_reference != shot1_output
    assert shot2_reference == anchor_path


# --- 9/10/11/12. Derived seeds -----------------------------------------------


def test_derived_seeds_differ_by_shot(tmp_path):
    anchor_path = str(tmp_path / "anchor.png")
    _make_anchor(anchor_path)
    runner = _runner(tmp_path)
    plan = asyncio.run(runner.prepare(_sample_pgn(), _preferences(), anchor_path=anchor_path, anchor_original_seed=42, style="anime", battle_mode="duel"))
    seeds = [s.planned_flux_seed for s in plan.shots]
    assert seeds[0] != seeds[1]


def test_derived_seeds_are_deterministic(tmp_path):
    anchor_path = str(tmp_path / "anchor.png")
    _make_anchor(anchor_path)
    runner1 = _runner(tmp_path)
    plan1 = asyncio.run(runner1.prepare(_sample_pgn(), _preferences(), anchor_path=anchor_path, anchor_original_seed=42, style="anime", battle_mode="duel"))
    runner2 = _runner(tmp_path)
    plan2 = asyncio.run(runner2.prepare(_sample_pgn(), _preferences(), anchor_path=anchor_path, anchor_original_seed=42, style="anime", battle_mode="duel"))
    assert [s.planned_flux_seed for s in plan1.shots] == [s.planned_flux_seed for s in plan2.shots]


def test_planned_and_actual_seeds_agree_on_success(tmp_path):
    anchor_path = str(tmp_path / "anchor.png")
    _make_anchor(anchor_path)
    runner = _runner(tmp_path)
    plan = asyncio.run(runner.prepare(_sample_pgn(), _preferences(), anchor_path=anchor_path, anchor_original_seed=42, style="anime", battle_mode="duel"))
    provider = _CorrectFakeReferenceProvider(plan.fight_base_visual_seed)
    result = asyncio.run(runner.execute(plan, provider))
    for r in result.shot_results:
        assert r.planned_flux_seed == r.actual_flux_seed


def test_seed_mismatch_raises(tmp_path):
    anchor_path = str(tmp_path / "anchor.png")
    _make_anchor(anchor_path)
    runner = _runner(tmp_path)
    plan = asyncio.run(runner.prepare(_sample_pgn(), _preferences(), anchor_path=anchor_path, anchor_original_seed=42, style="anime", battle_mode="duel"))
    disagreeing_provider = _DisagreeingReferenceProvider(999999999)
    with pytest.raises(SeedEvidenceMismatchError):
        asyncio.run(runner.execute(plan, disagreeing_provider))


def test_shared_seed_policy_is_not_used_for_this_calibration(tmp_path):
    """Both derived seeds must differ from the shared (base) seed —
    confirming the CLI's own use of the DERIVED policy, not SHARED."""
    anchor_path = str(tmp_path / "anchor.png")
    _make_anchor(anchor_path)
    runner = _runner(tmp_path)
    plan = asyncio.run(runner.prepare(_sample_pgn(), _preferences(), anchor_path=anchor_path, anchor_original_seed=42, style="anime", battle_mode="duel"))
    for shot in plan.shots:
        assert shot.planned_flux_seed != plan.fight_base_visual_seed  # SHARED would equal the base seed exactly


# --- 13/14. Prompt text unchanged from Prompt 13 -----------------------------


def test_prompt_text_matches_existing_prompt13_composer(tmp_path):
    from products.chess2fight.cinematic.prompt_generator import compose_reference_edit_prompt
    from tests.test_prompt12_visual_continuity import _prompted_timeline

    anchor_path = str(tmp_path / "anchor.png")
    _make_anchor(anchor_path)
    runner = _runner(tmp_path)
    plan = asyncio.run(runner.prepare(_sample_pgn(), _preferences(), anchor_path=anchor_path, anchor_original_seed=42, style="anime", battle_mode="duel"))

    prompted = _prompted_timeline(_sample_pgn())
    expected_prompt_1 = compose_reference_edit_prompt(prompted.shots[1])
    expected_prompt_2 = compose_reference_edit_prompt(prompted.shots[2])
    assert plan.shots[0].prompt == expected_prompt_1
    assert plan.shots[1].prompt == expected_prompt_2


def test_no_prompt_wording_changes_introduced():
    """The calibration module itself never constructs prompt text —
    confirmed directly: no f-string containing "preserve" or "change"
    appears in its own source; it only calls compose_reference_edit_prompt."""
    import products.chess2fight.rendering.reference_seed_calibration as module

    with open(module.__file__) as f:
        source = f.read()
    assert "preserve" not in source.lower()
    assert '"change' not in source.lower()
    assert "compose_reference_edit_prompt" in source


# --- 15/16. Anchor validation -------------------------------------------------


def test_wrong_anchor_dimensions_fails_before_generation(tmp_path):
    anchor_path = str(tmp_path / "wrong_size.png")
    _make_anchor(anchor_path, width=640, height=480)
    runner = _runner(tmp_path)
    with pytest.raises(AnchorValidationError):
        asyncio.run(runner.prepare(_sample_pgn(), _preferences(), anchor_path=anchor_path, anchor_original_seed=42, style="anime", battle_mode="duel"))


def test_missing_anchor_fails_before_generation(tmp_path):
    runner = _runner(tmp_path)
    with pytest.raises(AnchorValidationError):
        asyncio.run(runner.prepare(_sample_pgn(), _preferences(), anchor_path=str(tmp_path / "nonexistent.png"), anchor_original_seed=42, style="anime", battle_mode="duel"))


# --- 17/18. Failure semantics -------------------------------------------------


def test_first_reference_failure_stops_immediately(tmp_path):
    anchor_path = str(tmp_path / "anchor.png")
    _make_anchor(anchor_path)
    runner = _runner(tmp_path)
    plan = asyncio.run(runner.prepare(_sample_pgn(), _preferences(), anchor_path=anchor_path, anchor_original_seed=42, style="anime", battle_mode="duel"))
    provider = _DisagreeingReferenceProvider(plan.shots[0].planned_flux_seed, fail_on_call=1)
    with pytest.raises(ImageProviderError):
        asyncio.run(runner.execute(plan, provider))
    assert len(provider.calls) == 1  # never attempted the second


def test_partial_output_preserved_after_second_reference_failure(tmp_path):
    anchor_path = str(tmp_path / "anchor.png")
    _make_anchor(anchor_path)
    runner = _runner(tmp_path)
    plan = asyncio.run(runner.prepare(_sample_pgn(), _preferences(), anchor_path=anchor_path, anchor_original_seed=42, style="anime", battle_mode="duel"))

    class _FailOnSecond:
        def __init__(self, base_seed):
            self._override = build_seed_override(VisualSeedPolicy.DERIVED, base_seed)
            self.calls = []

        async def generate_reference_conditioned_image(self, prompt, reference_image_path, width=1024, height=1024):
            self.calls.append((prompt, reference_image_path))
            if len(self.calls) == 2:
                raise ImageProviderError("simulated second-shot failure")
            path = f"/tmp/calib_partial_{len(self.calls)}.png"
            Image.new("RGB", (width, height)).save(path)
            return ImageGenerationResult(
                image_path=path, provider="_FailOnSecond", prompt=prompt, width=width, height=height,
                generation_time_seconds=0.0, metadata={"seed": self._override(prompt)},
            )

    provider = _FailOnSecond(plan.fight_base_visual_seed)
    with pytest.raises(ImageProviderError):
        asyncio.run(runner.execute(plan, provider))

    calibration_dirs = list((tmp_path / "storage" / "reference_calibration").glob("*")) if (tmp_path / "storage" / "reference_calibration").exists() else []
    assert len(calibration_dirs) == 1
    files_in_dir = list(calibration_dirs[0].glob("*.png"))
    assert len(files_in_dir) == 1  # only the first shot's output was saved


# --- 19/20. Dry-run and generation count -------------------------------------


def test_dry_run_makes_zero_provider_calls():
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        anchor_path = f"{tmp}/anchor.png"
        _make_anchor(anchor_path)
        result = subprocess.run(
            [sys.executable, RENDER_CALIBRATION_CLI, "--sample", "--anchor-path", anchor_path,
             "--anchor-original-seed", "1697950441", "--dry-run"],
            capture_output=True, text=True, check=True,
        )
        assert "generating" not in result.stdout.lower()
        assert "dry run complete" in result.stdout.lower()


def test_expected_comfyui_job_count_is_exactly_two(tmp_path):
    anchor_path = str(tmp_path / "anchor.png")
    _make_anchor(anchor_path)
    runner = _runner(tmp_path)
    plan = asyncio.run(runner.prepare(_sample_pgn(), _preferences(), anchor_path=anchor_path, anchor_original_seed=42, style="anime", battle_mode="duel"))
    assert plan.expected_comfyui_jobs == 2


# --- 21/22/23. Existing paths and contracts unchanged ------------------------


def test_existing_prompt13_full_acceptance_unchanged():
    result = subprocess.run(
        [sys.executable, "scripts/render_reference_continuity_acceptance.py", "--sample", "--dry-run"],
        capture_output=True, text=True, check=True,
    )
    assert "expected comfyui job count: 6" in result.stdout.lower()


def test_existing_reference_workflow_unchanged():
    """Sprint 4 Prompt 18: production now routes through
    FluxKontextMultiReferenceLatentMethod (offset) — see that
    prompt's own production promotion. Updated from the direct
    ref:3/ref:4 links this test originally asserted (Sprint 4
    Prompt 13.1)."""
    import json

    with open("products/chess2fight/rendering/workflows/flux2_klein_reference_4b.json") as f:
        wf = json.load(f)
    assert wf["77:90"]["inputs"]["positive"] == ["method:1", 0]
    assert wf["77:90"]["inputs"]["negative"] == ["method:2", 0]
    assert wf["method:1"]["inputs"]["conditioning"] == ["ref:3", 0]
    assert wf["method:2"]["inputs"]["conditioning"] == ["ref:4", 0]


def test_existing_image_provider_api_unchanged():
    import inspect

    from core.image_router import ImageProvider

    sig = inspect.signature(ImageProvider.generate_image)
    assert list(sig.parameters.keys()) == ["self", "prompt", "width", "height"]


# --- 24. No ordinary test contacts real ComfyUI ------------------------------


def test_calibration_module_has_no_hardcoded_network_target():
    import products.chess2fight.rendering.reference_seed_calibration as module

    with open(module.__file__) as f:
        source = f.read()
    assert "httpx.AsyncClient(" not in source
    assert "http://" not in source
