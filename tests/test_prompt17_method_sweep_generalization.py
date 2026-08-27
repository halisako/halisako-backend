"""Tests for Sprint 4 Prompt 17's generalization of the reference-
method sweep CLI/runner to a configurable timeline index and method
list — while preserving Sprint 4 Prompt 16's exact default behavior.

TEST CATEGORY: MOCKED UNIT / LOCAL INTEGRATION TESTS — no real ComfyUI
server or GPU is ever contacted.
"""

import asyncio
import subprocess
import sys

from products.chess2fight.rendering.reference_method_calibration import (
    CANDIDATE_METHODS,
    method_workflow_path,
)
from tests.test_prompt16_reference_method_sweep import _plan, _runner
from tests.test_prompt14_reference_seed_calibration import _make_anchor
from tests.test_single_shot_acceptance import _preferences, _sample_pgn

RENDER_METHOD_CLI = "scripts/render_reference_method_calibration.py"
_PGN = _sample_pgn()


# --- Prompt 16 default behavior unchanged -----------------------------------


def test_runner_default_still_plans_timeline_index_2(tmp_path):
    """Calling prepare() without timeline_index/candidate_methods at
    all — exactly the Prompt 16 call shape — must still plan index 2
    with all three methods."""
    _, plan = asyncio.run(_plan(tmp_path))  # helper doesn't pass timeline_index/candidate_methods
    assert plan.timeline_index == 2
    assert [c.method for c in plan.candidates] == list(CANDIDATE_METHODS)
    assert plan.expected_comfyui_jobs == 3


def test_cli_default_dry_run_matches_prompt16_exactly():
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        anchor_path = f"{tmp}/anchor.png"
        _make_anchor(anchor_path)
        result = subprocess.run(
            [sys.executable, RENDER_METHOD_CLI, "--sample", "--anchor-path", anchor_path,
             "--anchor-original-seed", "1697950441", "--seed", "981216397", "--dry-run"],
            capture_output=True, text=True, check=True,
        )
        assert "timeline index: 2" in result.stdout
        assert "expected comfyui jobs: 3" in result.stdout.lower()
        for method in CANDIDATE_METHODS:
            assert method in result.stdout


# --- --methods "offset" yields exactly one candidate, 1 expected job -------


def test_runner_single_method_yields_one_candidate_and_one_job(tmp_path):
    anchor_path = str(tmp_path / "anchor.png")
    _make_anchor(anchor_path)
    runner = _runner(tmp_path)
    plan = asyncio.run(
        runner.prepare(
            _PGN, _preferences(), anchor_path=anchor_path, anchor_original_seed=1697950441,
            seed=2727023522, style="anime", battle_mode="duel", timeline_index=1,
            candidate_methods=("offset",),
        )
    )
    assert len(plan.candidates) == 1
    assert plan.candidates[0].method == "offset"
    assert plan.candidates[0].workflow_path == method_workflow_path("offset")
    assert plan.expected_comfyui_jobs == 1


def test_cli_methods_offset_yields_one_job():
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        anchor_path = f"{tmp}/anchor.png"
        _make_anchor(anchor_path)
        result = subprocess.run(
            [sys.executable, RENDER_METHOD_CLI, "--sample", "--anchor-path", anchor_path,
             "--anchor-original-seed", "1697950441", "--seed", "2727023522",
             "--timeline-index", "1", "--methods", "offset", "--dry-run"],
            capture_output=True, text=True, check=True,
        )
        assert "expected comfyui jobs: 1" in result.stdout.lower()
        assert "uxo/uno" not in result.stdout
        assert "index_timestep_zero" not in result.stdout


# --- --timeline-index 1 changes the selected sample shot correctly ---------


def test_timeline_index_1_selects_the_correct_shot(tmp_path):
    from products.chess2fight.cinematic.prompt_generator import compose_reference_edit_prompt
    from tests.test_prompt12_visual_continuity import _prompted_timeline

    anchor_path = str(tmp_path / "anchor.png")
    _make_anchor(anchor_path)
    runner = _runner(tmp_path)
    plan = asyncio.run(
        runner.prepare(
            _PGN, _preferences(), anchor_path=anchor_path, anchor_original_seed=1697950441,
            seed=2727023522, style="anime", battle_mode="duel", timeline_index=1, candidate_methods=("offset",),
        )
    )
    prompted = _prompted_timeline(_PGN)
    expected_shot1_prompt = compose_reference_edit_prompt(prompted.shots[1])
    expected_shot2_prompt = compose_reference_edit_prompt(prompted.shots[2])
    assert plan.timeline_index == 1
    assert plan.prompt == expected_shot1_prompt
    assert plan.prompt != expected_shot2_prompt  # confirms it's genuinely a different shot, not accidentally still 2


# --- Dry-run output reflects the chosen index/method set --------------------


def test_dry_run_output_reflects_chosen_index_and_methods():
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        anchor_path = f"{tmp}/anchor.png"
        _make_anchor(anchor_path)
        result = subprocess.run(
            [sys.executable, RENDER_METHOD_CLI, "--sample", "--anchor-path", anchor_path,
             "--anchor-original-seed", "1697950441", "--seed", "2727023522",
             "--timeline-index", "1", "--methods", "offset,index_timestep_zero", "--dry-run"],
            capture_output=True, text=True, check=True,
        )
        assert "timeline index: 1" in result.stdout
        assert "expected comfyui jobs: 2" in result.stdout.lower()
        assert "offset" in result.stdout
        assert "index_timestep_zero" in result.stdout
        assert "uxo/uno" not in result.stdout


def test_cli_rejects_unknown_method():
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        anchor_path = f"{tmp}/anchor.png"
        _make_anchor(anchor_path)
        result = subprocess.run(
            [sys.executable, RENDER_METHOD_CLI, "--sample", "--anchor-path", anchor_path,
             "--anchor-original-seed", "1697950441", "--seed", "2727023522", "--methods", "not_a_real_method"],
            capture_output=True, text=True,
        )
        assert result.returncode == 1
        assert "unknown method" in result.stderr.lower()


# --- No AnimationPipeline / AnimationRouter / VideoBuilder in the new logic -


def test_no_animation_or_videobuilder_imports_in_cli():
    with open(RENDER_METHOD_CLI) as f:
        source = f.read()
    assert "import AnimationPipeline" not in source
    assert "import AnimationRouter" not in source
    assert "import VideoBuilder" not in source


def test_no_animation_or_videobuilder_imports_in_runner():
    import products.chess2fight.rendering.reference_method_calibration as module

    with open(module.__file__) as f:
        source = f.read()
    assert "import AnimationPipeline" not in source
    assert "import AnimationRouter" not in source
    assert "import VideoBuilder" not in source
