"""Unit tests for AssetManager — the storage layout for rendered frames."""

import json

import pytest

from products.chess2fight.rendering.asset_manager import AssetManager, FrameMetadata, RenderManifest


def _sample_metadata(frame_number: int = 1) -> FrameMetadata:
    return FrameMetadata(
        frame_number=frame_number,
        prompt="a test prompt",
        camera_angle="wide",
        camera_motion="static",
        shot_id=f"shot_{frame_number}",
        shot_type="establishing",
        source_moves=["1. e4"],
        timestamp="2026-01-01T00:00:00+00:00",
        generation_seed=12345,
    )


# --- Storage layout ---------------------------------------------------------


def test_fight_directory_matches_the_required_layout(tmp_path):
    manager = AssetManager(storage_root=str(tmp_path))
    fight_dir = manager.fight_directory("fight_abc")
    assert fight_dir == tmp_path / "renders" / "fight_abc"
    assert fight_dir.exists()


def test_frame_filename_is_four_digit_zero_padded():
    manager = AssetManager(storage_root="/irrelevant")
    assert manager.frame_filename(1) == "frame0001.png"
    assert manager.frame_filename(42) == "frame0042.png"
    assert manager.frame_filename(9999) == "frame9999.png"


def test_fight_directory_is_created_if_missing(tmp_path):
    manager = AssetManager(storage_root=str(tmp_path))
    assert not (tmp_path / "renders" / "brand_new_fight").exists()
    manager.fight_directory("brand_new_fight")
    assert (tmp_path / "renders" / "brand_new_fight").exists()


def test_different_fight_ids_get_separate_directories(tmp_path):
    manager = AssetManager(storage_root=str(tmp_path))
    dir_a = manager.fight_directory("fight_a")
    dir_b = manager.fight_directory("fight_b")
    assert dir_a != dir_b


# --- save_frame ---------------------------------------------------------------


def test_save_frame_copies_the_source_file_exactly(tmp_path):
    source = tmp_path / "source.png"
    source.write_bytes(b"FAKE_IMAGE_BYTES")

    manager = AssetManager(storage_root=str(tmp_path / "storage"))
    destination = manager.save_frame("fight_1", 1, str(source))

    assert destination.exists()
    assert destination.name == "frame0001.png"
    assert destination.read_bytes() == b"FAKE_IMAGE_BYTES"


def test_save_frame_leaves_the_original_source_file_intact(tmp_path):
    """Copies, doesn't move — the image provider may want to keep its
    own copy (e.g. for content-hash deduplication)."""
    source = tmp_path / "source.png"
    source.write_bytes(b"ORIGINAL_BYTES")

    manager = AssetManager(storage_root=str(tmp_path / "storage"))
    manager.save_frame("fight_1", 1, str(source))

    assert source.exists()
    assert source.read_bytes() == b"ORIGINAL_BYTES"


def test_multiple_frames_saved_under_the_same_fight_id(tmp_path):
    manager = AssetManager(storage_root=str(tmp_path))
    for i in range(1, 4):
        source = tmp_path / f"src_{i}.png"
        source.write_bytes(f"frame {i}".encode())
        manager.save_frame("fight_multi", i, str(source))

    fight_dir = manager.fight_directory("fight_multi")
    saved_files = sorted(p.name for p in fight_dir.iterdir())
    assert saved_files == ["frame0001.png", "frame0002.png", "frame0003.png"]


# --- Manifest read/write ---------------------------------------------------------


def test_write_and_read_manifest_round_trips_exactly(tmp_path):
    manager = AssetManager(storage_root=str(tmp_path))
    manifest = RenderManifest(fight_id="fight_1", frame_count=2, frames=[_sample_metadata(1), _sample_metadata(2)])

    manager.write_manifest("fight_1", manifest)
    reloaded = manager.read_manifest("fight_1")

    assert reloaded == manifest


def test_manifest_is_valid_json_on_disk(tmp_path):
    manager = AssetManager(storage_root=str(tmp_path))
    manifest = RenderManifest(fight_id="fight_1", frame_count=1, frames=[_sample_metadata(1)])
    path = manager.write_manifest("fight_1", manifest)

    with open(path) as f:
        data = json.load(f)
    assert data["fight_id"] == "fight_1"
    assert data["frame_count"] == 1
    assert len(data["frames"]) == 1


def test_reading_a_manifest_that_was_never_written_raises():
    manager = AssetManager(storage_root="/tmp/definitely_never_written_xyz")
    with pytest.raises(FileNotFoundError):
        manager.read_manifest("never_rendered")


def test_manifest_path_is_inside_the_fight_directory(tmp_path):
    manager = AssetManager(storage_root=str(tmp_path))
    manifest_path = manager.manifest_path("fight_1")
    assert manifest_path.parent == tmp_path / "renders" / "fight_1"
    assert manifest_path.name == "metadata.json"


# --- Schema validation ---------------------------------------------------------


def test_frame_metadata_requires_frame_number_at_least_one():
    with pytest.raises(ValueError):
        FrameMetadata(
            frame_number=0, prompt="x", camera_angle="wide", camera_motion="static",
            shot_id="s1", shot_type="establishing", source_moves=[],
            timestamp="2026-01-01T00:00:00Z", generation_seed=1,
        )


def test_render_manifest_requires_at_least_one_frame():
    with pytest.raises(ValueError):
        RenderManifest(fight_id="fight_1", frame_count=0, frames=[])


def test_frame_metadata_source_moves_defaults_to_empty_list():
    metadata = FrameMetadata(
        frame_number=1, prompt="x", camera_angle="wide", camera_motion="static",
        shot_id="s1", shot_type="establishing", timestamp="2026-01-01T00:00:00Z", generation_seed=1,
    )
    assert metadata.source_moves == []
