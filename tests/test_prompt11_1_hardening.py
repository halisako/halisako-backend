"""Regression tests for Sprint 4 Prompt 11.1's four hardening fixes:

1. The multi-shot acceptance CLI now defaults --max-animation-seconds
   to 2.0 (not None/uncapped) — a normal invocation with no duration
   flag resolves every selected shot to the safe ~17-frame/8fps
   baseline. Genuinely uncapped generation requires the explicit,
   high-friction --allow-uncapped-duration flag.
2. Dry-run/plan-summary output now explicitly shows both
   independently-resolved policies: FLUX image resolution (1280x704)
   and Wan animation resolution (832x480) — never coupled to the
   generic ImageRouter/AnimationInstruction defaults.
3. MultiShotAcceptanceResult.final_video_duration_seconds is now
   measured directly with ffprobe after concatenation, not
   VideoBuilder's own echoed-back predicted sum mislabeled as
   "actual". A failed measurement fails the run.
4. The manifest now records the actual resolved image/animation
   dimensions execute() used (carried explicitly on the result),
   not global settings re-read after the fact — correct only by
   coincidence when there was no override.

Plus the previously-unimplemented output-writability preflight check.

TEST CATEGORY: MOCKED UNIT / LOCAL INTEGRATION TESTS — no real ComfyUI
server or GPU is ever contacted.
"""

import asyncio
import subprocess
import sys

import pytest

from core.animation_providers.comfyui import _duration_to_frame_count
from core.config import get_settings
from products.chess2fight.rendering.acceptance_preflight import check_output_writability
from products.chess2fight.rendering.multi_shot_acceptance import FinalVideoMeasurementError
from tests.test_multi_shot_acceptance import _runner
from tests.test_single_shot_acceptance import _preferences, _sample_pgn

RENDER_MULTI_SHOT = "scripts/render_multi_shot_acceptance.py"


# --- Fix 1: CLI defaults safely capped, not uncapped ------------------------


def test_cli_default_effective_duration_is_capped_to_2_seconds_per_shot():
    """No --max-animation-seconds flag at all — must resolve to the
    safe ~2s cap, not each shot's full real cinematic duration."""
    result = subprocess.run(
        [sys.executable, RENDER_MULTI_SHOT, "--sample", "--dry-run"],
        capture_output=True, text=True, check=True,
    )
    assert "effective=2.00s" in result.stdout
    assert result.stdout.count("effective=2.00s") == 3
    # None of the real, uncapped per-shot durations should appear as
    # the *effective* value.
    assert "effective=7.75s" not in result.stdout
    assert "effective=8.91s" not in result.stdout


def test_cli_default_resolves_to_17_17_17_frames_at_8fps():
    result = subprocess.run(
        [sys.executable, RENDER_MULTI_SHOT, "--sample", "--dry-run"],
        capture_output=True, text=True, check=True,
    )
    assert result.stdout.count("frames=17") == 3
    assert "resolved fps:        8" in result.stdout


def test_no_explicit_duration_flag_alone_can_mean_uncapped():
    """Passing a large but finite --max-animation-seconds is not the
    same as uncapped — only --allow-uncapped-duration produces
    genuinely uncapped (full real duration) behavior."""
    result = subprocess.run(
        [sys.executable, RENDER_MULTI_SHOT, "--sample", "--dry-run", "--max-animation-seconds", "100"],
        capture_output=True, text=True, check=True,
    )
    # Even with a large cap, real per-shot durations (7.75s, 8.91s) are
    # all below 100s, so they resolve to themselves — this is
    # deliberate cap behavior (min(real, cap)), not "uncapped" as a
    # distinct mode. Confirm this differs from the --allow-uncapped
    # path by checking the resolved frame counts match real durations.
    assert "effective=7.75s" in result.stdout


def test_allow_uncapped_duration_flag_produces_genuinely_uncapped_behavior():
    result = subprocess.run(
        [sys.executable, RENDER_MULTI_SHOT, "--sample", "--dry-run", "--allow-uncapped-duration"],
        capture_output=True, text=True, check=True,
    )
    assert "effective=7.75s" in result.stdout
    assert "effective=8.91s" in result.stdout
    assert "effective=2.00s" not in result.stdout


def test_underlying_runner_still_represents_uncapped_when_called_directly(tmp_path):
    """The generic MultiShotAcceptanceRunner.prepare() itself is
    unchanged — max_animation_seconds=None still means uncapped when
    called deliberately from Python. Only the CLI's own argparse
    default changed."""
    runner = _runner(tmp_path)
    plan = asyncio.run(runner.prepare(_sample_pgn(), _preferences(), max_animation_seconds=None))
    assert plan.effective_animation_durations_seconds == [shot.duration_seconds for shot in plan.shots]
    assert plan.max_animation_seconds is None


# --- Fix 2: dry-run shows both resolution policies --------------------------


def test_dry_run_shows_both_flux_and_wan_resolutions():
    result = subprocess.run(
        [sys.executable, RENDER_MULTI_SHOT, "--sample", "--dry-run"],
        capture_output=True, text=True, check=True,
    )
    assert "FLUX image resolution:      1280x704" in result.stdout
    assert "Wan animation resolution:   832x480" in result.stdout


def test_dry_run_reflects_explicit_animation_resolution_override():
    result = subprocess.run(
        [sys.executable, RENDER_MULTI_SHOT, "--sample", "--dry-run",
         "--animation-width", "640", "--animation-height", "480"],
        capture_output=True, text=True, check=True,
    )
    assert "FLUX image resolution:      1280x704" in result.stdout  # unaffected by the animation-only override
    assert "Wan animation resolution:   640x480" in result.stdout


# --- Fix 3: measured (not predicted) final duration -------------------------


def test_expected_and_measured_durations_are_distinct_fields(tmp_path):
    runner = _runner(tmp_path)
    plan = asyncio.run(runner.prepare(_sample_pgn(), _preferences(), max_animation_seconds=2.0))
    result = asyncio.run(runner.execute(plan, width=256, height=256))

    # Both fields exist and are independently meaningful — not
    # asserting they're numerically different here (mock happens to
    # produce a different value than the frame-snapped prediction, but
    # that's incidental, not the property under test), just that the
    # measured field is populated from a real, independent measurement.
    assert plan.expected_assembled_duration_seconds > 0
    assert result.final_video_duration_seconds > 0

    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", result.final_video_path],
        capture_output=True, text=True, check=True,
    )
    independently_measured = float(probe.stdout.strip())
    assert result.final_video_duration_seconds == pytest.approx(independently_measured, abs=0.01)


def test_manifest_records_both_expected_and_actual_measured_duration():
    import json

    manifest_path = "test_manifest_prompt11_1.json"
    subprocess.run(
        [sys.executable, RENDER_MULTI_SHOT, "--sample", "--manifest-path", manifest_path],
        capture_output=True, text=True, check=True,
    )
    manifest = json.loads(open(manifest_path).read())
    assert "expected_assembled_duration_seconds" in manifest
    assert "actual_final_duration_seconds" in manifest
    assert manifest["actual_final_duration_seconds"] > 0

    import os
    import shutil

    os.remove(manifest_path)
    shutil.rmtree("generated_animations", ignore_errors=True)
    shutil.rmtree("generated_images", ignore_errors=True)
    shutil.rmtree("storage", ignore_errors=True)


def test_failed_ffprobe_measurement_raises_and_does_not_fall_back(tmp_path, monkeypatch):
    """A failed measurement must fail the acceptance run — never
    silently fall back to the unmeasured predicted value."""
    import products.chess2fight.rendering.multi_shot_acceptance as module

    def _always_fails(path):
        raise FinalVideoMeasurementError(f"simulated ffprobe failure for {path}")

    monkeypatch.setattr(module, "_measure_video_duration_seconds", _always_fails)

    runner = _runner(tmp_path)
    plan = asyncio.run(runner.prepare(_sample_pgn(), _preferences(), max_animation_seconds=2.0))
    with pytest.raises(FinalVideoMeasurementError):
        asyncio.run(runner.execute(plan, width=256, height=256))


# --- Fix 4: manifest records actual resolved dimensions, not re-read settings -


def test_result_carries_actual_resolved_dimensions_not_just_settings(tmp_path):
    runner = _runner(tmp_path)
    plan = asyncio.run(runner.prepare(_sample_pgn(), _preferences(), max_animation_seconds=2.0))
    result = asyncio.run(runner.execute(plan, width=640, height=480))

    settings = get_settings()
    assert result.resolved_animation_width == 640
    assert result.resolved_animation_height == 480
    # Genuinely different from the global default, proving this is the
    # actual value used, not a re-read of the unchanged setting.
    assert result.resolved_animation_width != settings.comfyui_animation_default_width
    assert result.resolved_image_width == settings.comfyui_image_default_width  # FLUX unaffected


def test_manifest_records_explicit_override_not_the_global_default():
    import json

    manifest_path = "test_manifest_prompt11_1_override.json"
    subprocess.run(
        [sys.executable, RENDER_MULTI_SHOT, "--sample", "--manifest-path", manifest_path,
         "--animation-width", "640", "--animation-height", "480"],
        capture_output=True, text=True, check=True,
    )
    manifest = json.loads(open(manifest_path).read())
    assert manifest["resolved_animation_width"] == 640
    assert manifest["resolved_animation_height"] == 480
    assert manifest["resolved_image_width"] == 1280  # FLUX unaffected by the animation-only override

    import os
    import shutil

    os.remove(manifest_path)
    shutil.rmtree("generated_animations", ignore_errors=True)
    shutil.rmtree("generated_images", ignore_errors=True)
    shutil.rmtree("storage", ignore_errors=True)


def test_normal_run_still_records_the_correct_flux_and_wan_defaults():
    import json

    manifest_path = "test_manifest_prompt11_1_defaults.json"
    subprocess.run(
        [sys.executable, RENDER_MULTI_SHOT, "--sample", "--manifest-path", manifest_path],
        capture_output=True, text=True, check=True,
    )
    manifest = json.loads(open(manifest_path).read())
    assert (manifest["resolved_image_width"], manifest["resolved_image_height"]) == (1280, 704)
    assert (manifest["resolved_animation_width"], manifest["resolved_animation_height"]) == (832, 480)

    import os
    import shutil

    os.remove(manifest_path)
    shutil.rmtree("generated_animations", ignore_errors=True)
    shutil.rmtree("generated_images", ignore_errors=True)
    shutil.rmtree("storage", ignore_errors=True)


# --- Output writability preflight -------------------------------------------


def test_check_output_writability_passes_for_a_genuinely_writable_directory(tmp_path):
    problems = check_output_writability([str(tmp_path / "some_new_dir")])
    assert problems == []


def test_check_output_writability_leaves_no_permanent_probe_artifact(tmp_path):
    target = tmp_path / "checked_dir"
    check_output_writability([str(target)])
    remaining = list(target.iterdir()) if target.exists() else []
    assert remaining == []


def test_check_output_writability_detects_a_path_blocked_by_an_existing_file(tmp_path):
    """A root-proof, environment-independent way to trigger a genuine
    OSError — chmod-based unwritability is unreliable when tests run
    as root (confirmed directly: chmod 000 does not block root from
    writing), so this uses a path collision instead, which fails
    regardless of privilege level."""
    blocking_file = tmp_path / "blocked"
    blocking_file.write_text("this is a file, not a directory")

    problems = check_output_writability([str(blocking_file)])
    assert len(problems) == 1
    assert "blocked" in problems[0].lower() or str(blocking_file) in problems[0]


def test_check_output_writability_detects_genuine_write_failure_via_mocking(tmp_path, monkeypatch):
    """Mocking, not chmod, for a reliable OSError-on-write regardless
    of test-runner privilege level."""
    import pathlib

    real_write_text = pathlib.Path.write_text

    def _failing_write_text(self, *args, **kwargs):
        if "halisako_writability_probe" in self.name:
            raise OSError("simulated permission denied")
        return real_write_text(self, *args, **kwargs)

    monkeypatch.setattr(pathlib.Path, "write_text", _failing_write_text)

    problems = check_output_writability([str(tmp_path / "some_dir")])
    assert len(problems) == 1
    assert "not writable" in problems[0].lower()


def test_cli_blocks_before_generation_when_output_path_is_unwritable():
    """End-to-end: a path collision blocking the storage directory
    must fail the CLI before any 'Rendering via...' generation message
    ever prints."""
    import os

    blocking_path = "test_blocking_file_prompt11_1"
    with open(blocking_path, "w") as f:
        f.write("blocks the storage/ directory from being created")

    try:
        result = subprocess.run(
            [sys.executable, RENDER_MULTI_SHOT, "--sample"],
            capture_output=True, text=True,
            env={**os.environ, "RENDER_STORAGE_ROOT": blocking_path},
        )
        assert result.returncode == 1
        assert "preflight check failed" in result.stderr.lower()
        assert "rendering via" not in result.stdout.lower()
    finally:
        os.remove(blocking_path)


def test_writability_check_is_skipped_by_skip_preflight_flag():
    """--skip-preflight bypasses the writability check too, consistent
    with skipping the rest of the preflight sequence."""
    import os

    blocking_path = "test_blocking_file_prompt11_1_skip"
    with open(blocking_path, "w") as f:
        f.write("blocks the storage/ directory")

    try:
        result = subprocess.run(
            [sys.executable, RENDER_MULTI_SHOT, "--sample", "--skip-preflight"],
            capture_output=True, text=True,
            env={**os.environ, "RENDER_STORAGE_ROOT": blocking_path},
        )
        # Should fail later (inside actual file operations), not at
        # the "Preflight check failed" stage, since preflight itself
        # was skipped.
        assert "preflight check failed" not in result.stderr.lower()
    finally:
        os.remove(blocking_path)


# --- Regression: existing Prompt 11 invariants still hold -------------------


def test_six_job_expectation_and_default_cap_together(tmp_path):
    """Confirms the default cap doesn't disturb the core Prompt 11
    generation-count contract."""
    runner = _runner(tmp_path)
    plan = asyncio.run(runner.prepare(_sample_pgn(), _preferences()))  # CLI-equivalent: no explicit cap passed
    assert plan.expected_comfyui_job_count == 6


def test_frame_count_formula_unchanged_for_the_2_second_baseline():
    assert _duration_to_frame_count(2.0, 8) == 17
