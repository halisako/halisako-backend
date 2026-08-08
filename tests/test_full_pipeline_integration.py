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

import pytest
from fastapi.testclient import TestClient

from core import config
from main import app

SCHOLARS_MATE_PGN = (
    '[Event "Example"]\n[White "Halisako"]\n[Black "Guest"]\n[Result "1-0"]\n\n'
    "1. e4 e5 2. Qh5 Nc6 3. Bc4 Nf6 4. Qxf7# 1-0"
)


@pytest.fixture(autouse=True)
def _use_mock_image_provider(tmp_path, monkeypatch):
    """Points the whole app at MockImageProvider and a temp
    storage/image directory for every test in this file, and resets
    the shared ImageRouter singleton afterward so this doesn't leak
    into other test files."""
    import core.image_router as image_router_module

    original_singleton = image_router_module._router_instance
    image_router_module._router_instance = None

    monkeypatch.setattr(config.settings, "image_provider", "mock")
    monkeypatch.setattr(config.settings, "image_output_dir", str(tmp_path / "generated_images"))
    monkeypatch.setattr(config.settings, "render_storage_root", str(tmp_path / "storage"))

    yield

    image_router_module._router_instance = original_singleton


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
    expected_duration = data["frame_count"] * 1.0
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
