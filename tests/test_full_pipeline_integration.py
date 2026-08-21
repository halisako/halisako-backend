"""End-to-end integration test for the complete Sprint 3 pipeline:

    PGN -> Battle Director -> Battle Intelligence -> Fight Story ->
    Timeline Engine -> Scene Composer -> Prompt Generator ->
    ImageRouter -> Render Pipeline -> Video Builder -> fight.mp4

Exercised through the real HTTP API (a single POST to `/render`), with
`MockImageProvider` standing in for image generation — real ffmpeg
still runs, real frames are still written to disk, so this is a
genuine end-to-end run of everything except the one external service
this environment can't call.
"""

import json
import os
import subprocess
import tempfile

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from core import config
from main import app

SCHOLARS_MATE_PGN = (
    '[Event "Example"]\n[White "Halisako"]\n[Black "Guest"]\n[Result "1-0"]\n\n'
    "1. e4 e5 2. Qh5 Nc6 3. Bc4 Nf6 4. Qxf7# 1-0"
)


@pytest.fixture(autouse=True)
def _use_mock_image_provider(tmp_path, monkeypatch):
    """Points the whole app at MockImageProvider/MockAnimationProvider
    and temp storage/image/animation directories for every test in
    this file, and resets both shared router singletons afterward so
    this doesn't leak into other test files."""
    import core.animation_router as animation_router_module
    import core.image_router as image_router_module

    original_image_singleton = image_router_module._router_instance
    original_animation_singleton = animation_router_module._router_instance
    image_router_module._router_instance = None
    animation_router_module._router_instance = None

    monkeypatch.setattr(config.settings, "image_provider", "mock")
    monkeypatch.setattr(config.settings, "image_output_dir", str(tmp_path / "generated_images"))
    monkeypatch.setattr(config.settings, "render_storage_root", str(tmp_path / "storage"))
    monkeypatch.setattr(config.settings, "animation_provider", "mock")
    monkeypatch.setattr(config.settings, "animation_output_dir", str(tmp_path / "generated_animations"))

    yield

    image_router_module._router_instance = original_image_singleton
    animation_router_module._router_instance = original_animation_singleton


client = TestClient(app)


def test_full_pipeline_returns_the_required_response_shape():
    res = client.post("/api/v1/chess2fight/render", json={"pgn": SCHOLARS_MATE_PGN, "style": "anime"})
    assert res.status_code == 200
    data = res.json()

    assert set(data.keys()) == {"status", "fight_id", "frame_count", "video_path", "timeline", "frames"}
    assert data["status"] == "completed"
    assert isinstance(data["fight_id"], str) and data["fight_id"]
    assert data["frame_count"] == len(data["frames"])
    assert data["frame_count"] == len(data["timeline"])


def test_a_real_video_file_is_actually_produced_on_disk():
    res = client.post("/api/v1/chess2fight/render", json={"pgn": SCHOLARS_MATE_PGN, "style": "anime"})
    data = res.json()

    assert os.path.exists(data["video_path"])
    assert os.path.getsize(data["video_path"]) > 0
    assert data["video_path"].endswith("fight.mp4")


def test_video_is_a_valid_mp4_with_expected_properties():
    res = client.post(
        "/api/v1/chess2fight/render",
        json={
            "pgn": SCHOLARS_MATE_PGN, "style": "anime",
            "fps": 24, "width": 512, "height": 512, "frame_duration_seconds": 1.0,
        },
    )
    data = res.json()

    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-print_format", "json", "-show_format", "-show_streams", data["video_path"]],
        capture_output=True, text=True, check=True,
    )
    probe_data = json.loads(probe.stdout)
    video_stream = next(s for s in probe_data["streams"] if s["codec_type"] == "video")

    assert video_stream["width"] == 512
    assert video_stream["height"] == 512
    assert video_stream["r_frame_rate"] == "24/1"
    # Sprint 4 Prompt 2: total duration now comes from summing each
    # shot's own duration_seconds (via AnimationInstruction), not the
    # old frame_count * frame_duration_seconds uniform calculation —
    # data["timeline"] is ShotTimeline.shots, the same durations each
    # shot's animated clip was actually built from.
    expected_duration = sum(shot["duration_seconds"] for shot in data["timeline"])
    assert abs(float(probe_data["format"]["duration"]) - expected_duration) < 0.5


def test_all_frame_files_actually_exist_on_disk():
    res = client.post("/api/v1/chess2fight/render", json={"pgn": SCHOLARS_MATE_PGN, "style": "fantasy"})
    data = res.json()

    for frame in data["frames"]:
        assert os.path.exists(frame["frame_path"])
        assert os.path.getsize(frame["frame_path"]) > 0


def test_frame_metadata_is_complete_for_every_frame():
    res = client.post("/api/v1/chess2fight/render", json={"pgn": SCHOLARS_MATE_PGN, "style": "scifi"})
    data = res.json()

    required_metadata_fields = {
        "frame_number", "prompt", "camera_angle", "camera_motion",
        "shot_id", "shot_type", "source_moves", "timestamp", "generation_seed",
    }
    for frame in data["frames"]:
        assert set(frame["metadata"].keys()) == required_metadata_fields


def test_timeline_matches_frame_count_and_is_in_order():
    res = client.post("/api/v1/chess2fight/render", json={"pgn": SCHOLARS_MATE_PGN, "style": "anime"})
    data = res.json()

    assert [shot["sequence_order"] for shot in data["timeline"]] == list(range(1, len(data["timeline"]) + 1))
    assert [frame["frame_number"] for frame in data["frames"]] == list(range(1, len(data["frames"]) + 1))


def test_metadata_json_manifest_was_also_written_to_disk():
    """RenderPipeline/AssetManager's own metadata.json manifest (from
    the earlier rendering task) should exist alongside the video —
    the video builder is additive, not a replacement for it."""
    res = client.post("/api/v1/chess2fight/render", json={"pgn": SCHOLARS_MATE_PGN, "style": "anime"})
    data = res.json()

    manifest_path = os.path.join(os.path.dirname(data["video_path"]), "metadata.json")
    assert os.path.exists(manifest_path)


def test_different_fps_and_resolution_are_honored_end_to_end():
    res = client.post(
        "/api/v1/chess2fight/render",
        json={"pgn": SCHOLARS_MATE_PGN, "style": "anime", "fps": 12, "width": 256, "height": 256},
    )
    data = res.json()

    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-print_format", "json", "-show_streams", data["video_path"]],
        capture_output=True, text=True, check=True,
    )
    video_stream = next(s for s in json.loads(probe.stdout)["streams"] if s["codec_type"] == "video")
    assert video_stream["r_frame_rate"] == "12/1"
    assert (video_stream["width"], video_stream["height"]) == (256, 256)


def test_army_mode_renders_successfully_end_to_end():
    res = client.post(
        "/api/v1/chess2fight/render",
        json={"pgn": SCHOLARS_MATE_PGN, "preferences": {"battle_mode": "army", "style": "modern_warfare"}},
    )
    assert res.status_code == 200
    data = res.json()
    assert data["frame_count"] == len(data["frames"])


def test_legacy_style_only_request_works_without_preferences():
    """Same backward-compatibility pattern already established for
    `/generate` — a request with just `pgn` and the legacy top-level
    `style`, no `preferences`, must work."""
    res = client.post("/api/v1/chess2fight/render", json={"pgn": SCHOLARS_MATE_PGN, "style": "superhero"})
    assert res.status_code == 200


def test_two_separate_renders_get_different_fight_ids_and_separate_storage():
    res1 = client.post("/api/v1/chess2fight/render", json={"pgn": SCHOLARS_MATE_PGN, "style": "anime"})
    res2 = client.post("/api/v1/chess2fight/render", json={"pgn": SCHOLARS_MATE_PGN, "style": "anime"})
    data1, data2 = res1.json(), res2.json()

    assert data1["fight_id"] != data2["fight_id"]
    assert data1["video_path"] != data2["video_path"]


def test_invalid_pgn_returns_400_not_a_broken_video():
    res = client.post("/api/v1/chess2fight/render", json={"pgn": "garbage"})
    assert res.status_code == 400


def test_generate_endpoint_still_works_unaffected_by_the_new_route():
    """The existing Sprint 2 endpoint must be completely unaffected by
    this task's changes."""
    res = client.post("/api/v1/chess2fight/generate", json={"pgn": SCHOLARS_MATE_PGN, "style": "anime"})
    assert res.status_code == 200
    data = res.json()
    assert "prompted_timeline" in data
    assert "video_path" not in data  # /generate never produces a video


def test_missing_ffmpeg_surfaces_as_a_clean_502_not_a_silent_failure(monkeypatch, tmp_path):
    """If ffmpeg genuinely isn't available, the pipeline should fail
    clearly rather than silently succeeding with no video — verified
    by actually removing ffmpeg from PATH for this one test only."""
    monkeypatch.setenv("PATH", str(tmp_path))  # a PATH with no ffmpeg on it
    res = client.post("/api/v1/chess2fight/render", json={"pgn": SCHOLARS_MATE_PGN, "style": "anime"})
    assert res.status_code == 502


# --- Sprint 4 Prompt 2: the final MP4 genuinely contains animated segments -


def test_final_video_duration_reflects_per_shot_durations_not_a_uniform_value():
    """The core acceptance property distinguishing this from the old
    Sprint 3 static-frame path: total duration is the sum of each
    shot's own duration_seconds, not frame_count * a single uniform
    frame_duration_seconds."""
    res = client.post("/api/v1/chess2fight/render", json={"pgn": SCHOLARS_MATE_PGN, "style": "anime"})
    data = res.json()

    expected_duration = sum(shot["duration_seconds"] for shot in data["timeline"])
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-print_format", "json", "-show_format", data["video_path"]],
        capture_output=True, text=True, check=True,
    )
    actual_duration = float(json.loads(probe.stdout)["format"]["duration"])
    assert abs(actual_duration - expected_duration) < 0.5

    # And explicitly not the old calculation — regression guard against
    # silently reverting to the static-frame path.
    old_style_duration = data["frame_count"] * 2.0  # old default frame_duration_seconds
    assert abs(actual_duration - old_style_duration) > 0.5 or len({s["duration_seconds"] for s in data["timeline"]}) == 1


def test_final_video_contains_multiple_distinct_shot_segments_not_one_repeated_clip():
    """The most direct proof this is genuinely animated per-shot
    assembly: sampling frames at different timestamps across the final
    video shows different colors, matching different shots' different
    (MockImageProvider-generated) reference images — not one clip
    repeated or a single static frame held for the whole duration."""
    res = client.post(
        "/api/v1/chess2fight/render",
        json={"pgn": SCHOLARS_MATE_PGN, "style": "anime", "width": 320, "height": 240},
    )
    data = res.json()
    assert len(data["timeline"]) > 1  # multiple shots — the interesting case

    with tempfile.TemporaryDirectory() as extract_dir:
        subprocess.run(
            ["ffmpeg", "-y", "-i", data["video_path"], "-vf", "fps=2", f"{extract_dir}/f%03d.png"],
            capture_output=True, check=True,
        )
        frame_files = sorted(os.listdir(extract_dir))
        assert len(frame_files) >= 2

        colors = []
        for f in frame_files:
            img = Image.open(f"{extract_dir}/{f}").convert("RGB")
            colors.append(img.getpixel((160, 120)))

        # Not every sampled frame is the same color — genuine content
        # variation across the timeline, not one static image held
        # throughout.
        assert len(set(colors)) > 1


def test_render_response_frame_count_matches_animated_shot_count():
    res = client.post("/api/v1/chess2fight/render", json={"pgn": SCHOLARS_MATE_PGN, "style": "fantasy"})
    data = res.json()
    assert data["frame_count"] == len(data["timeline"])
    assert data["frame_count"] == len(data["frames"])


def test_multi_shot_timeline_end_to_end_acceptance():
    """The full Sprint 4 Prompt 2 acceptance path in one test: a real
    PGN through /render, through FightVideoPipeline, through reference
    image generation, AnimationInstruction, AnimationRouter,
    MockAnimationProvider, per-shot clips, to one final MP4 — checking
    every acceptance criterion from the task in one place."""
    res = client.post(
        "/api/v1/chess2fight/render",
        json={"pgn": SCHOLARS_MATE_PGN, "style": "anime", "fps": 24, "width": 480, "height": 480},
    )
    assert res.status_code == 200  # 1. HTTP/render request succeeds
    data = res.json()

    assert os.path.exists(data["video_path"])  # 2. a final MP4 is created
    assert os.path.getsize(data["video_path"]) > 0  # 3. playable/inspectable (non-empty)

    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-print_format", "json", "-show_format", "-show_streams", data["video_path"]],
        capture_output=True, text=True, check=True,
    )
    probe_data = json.loads(probe.stdout)
    video_stream = next(s for s in probe_data["streams"] if s["codec_type"] == "video")

    assert float(probe_data["format"]["duration"]) > 0  # 4. duration is non-zero
    assert video_stream["width"] == 480 and video_stream["height"] == 480  # 5. width/height valid
    assert len(data["timeline"]) > 1  # multiple shots in this timeline

    # 6. contains multiple shot segments — verified via color sampling,
    # same technique as the dedicated segment test above.
    with tempfile.TemporaryDirectory() as extract_dir:
        subprocess.run(
            ["ffmpeg", "-y", "-i", data["video_path"], "-vf", "fps=2", f"{extract_dir}/f%03d.png"],
            capture_output=True, check=True,
        )
        colors = {
            Image.open(f"{extract_dir}/{f}").convert("RGB").getpixel((240, 240))
            for f in sorted(os.listdir(extract_dir))
        }
        assert len(colors) > 1

    # 7. no longer merely the old static-frame path — duration matches
    # summed per-shot durations, not frame_count * uniform value.
    expected_duration = sum(shot["duration_seconds"] for shot in data["timeline"])
    assert abs(float(probe_data["format"]["duration"]) - expected_duration) < 0.5
