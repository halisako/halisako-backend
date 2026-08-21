"""Tests for VideoBuilder.

Uses the real `ffmpeg` binary — genuinely assembling and verifying
video files, rather than mocking the subprocess call, since ffmpeg is
actually available in this environment and a mocked subprocess call
would prove nothing about whether the constructed command is actually
correct. Verifies output independently via `ffprobe` (the same tool a
human would reach for), not just by trusting this module's own
self-reported result.
"""

import asyncio
import json
import subprocess

import pytest
from PIL import Image

from core.exceptions import VideoBuilderError
from products.chess2fight.rendering.video_builder import VideoBuilder


def _make_frames(directory, count: int, size=(320, 240)) -> None:
    """Creates `count` real PNG frames, each a different solid color,
    named frame0001.png, frame0002.png, ... in `directory`."""
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)]
    for i in range(1, count + 1):
        color = colors[(i - 1) % len(colors)]
        Image.new("RGB", size, color=color).save(directory / f"frame{i:04d}.png")


def _probe(video_path: str) -> dict:
    """Runs real ffprobe and returns parsed JSON — an independent
    check of what was actually produced, not this module's own claim."""
    result = subprocess.run(
        [
            "ffprobe", "-v", "error", "-print_format", "json",
            "-show_format", "-show_streams", video_path,
        ],
        capture_output=True, text=True, check=True,
    )
    return json.loads(result.stdout)


# --- Real, successful builds --------------------------------------------------


def test_builds_a_real_valid_mp4_file(tmp_path):
    frames_dir = tmp_path / "frames"
    frames_dir.mkdir()
    _make_frames(frames_dir, 5)

    builder = VideoBuilder()
    result = asyncio.run(
        builder.build_video(
            frame_directory=str(frames_dir), output_path=str(tmp_path / "fight.mp4"),
            frame_count=5, fps=24, width=320, height=240, frame_duration_seconds=1.0,
        )
    )

    assert (tmp_path / "fight.mp4").exists()
    probe = _probe(result.video_path)
    video_stream = next(s for s in probe["streams"] if s["codec_type"] == "video")
    assert video_stream["width"] == 320
    assert video_stream["height"] == 240


def test_resolution_is_respected(tmp_path):
    frames_dir = tmp_path / "frames"
    frames_dir.mkdir()
    _make_frames(frames_dir, 3)

    builder = VideoBuilder()
    result = asyncio.run(
        builder.build_video(
            frame_directory=str(frames_dir), output_path=str(tmp_path / "fight.mp4"),
            frame_count=3, width=640, height=480, frame_duration_seconds=1.0,
        )
    )

    probe = _probe(result.video_path)
    video_stream = next(s for s in probe["streams"] if s["codec_type"] == "video")
    assert (int(video_stream["width"]), int(video_stream["height"])) == (640, 480)
    assert (result.width, result.height) == (640, 480)


def test_fps_is_respected(tmp_path):
    frames_dir = tmp_path / "frames"
    frames_dir.mkdir()
    _make_frames(frames_dir, 3)

    builder = VideoBuilder()
    result = asyncio.run(
        builder.build_video(
            frame_directory=str(frames_dir), output_path=str(tmp_path / "fight.mp4"),
            frame_count=3, fps=30, frame_duration_seconds=1.0,
        )
    )

    probe = _probe(result.video_path)
    video_stream = next(s for s in probe["streams"] if s["codec_type"] == "video")
    assert video_stream["r_frame_rate"] == "30/1"
    assert result.fps == 30


def test_total_duration_matches_frame_count_times_frame_duration(tmp_path):
    frames_dir = tmp_path / "frames"
    frames_dir.mkdir()
    _make_frames(frames_dir, 4)

    builder = VideoBuilder()
    result = asyncio.run(
        builder.build_video(
            frame_directory=str(frames_dir), output_path=str(tmp_path / "fight.mp4"),
            frame_count=4, frame_duration_seconds=2.5,
        )
    )

    assert result.duration_seconds == 10.0  # 4 * 2.5
    probe = _probe(result.video_path)
    actual_duration = float(probe["format"]["duration"])
    assert abs(actual_duration - 10.0) < 0.5  # ffmpeg encoding introduces small rounding


def test_different_frame_duration_changes_total_video_length(tmp_path):
    frames_dir = tmp_path / "frames"
    frames_dir.mkdir()
    _make_frames(frames_dir, 3)
    builder = VideoBuilder()

    short = asyncio.run(
        builder.build_video(
            frame_directory=str(frames_dir), output_path=str(tmp_path / "short.mp4"),
            frame_count=3, frame_duration_seconds=0.5,
        )
    )
    long = asyncio.run(
        builder.build_video(
            frame_directory=str(frames_dir), output_path=str(tmp_path / "long.mp4"),
            frame_count=3, frame_duration_seconds=3.0,
        )
    )
    assert long.duration_seconds > short.duration_seconds


def test_output_directory_created_if_missing(tmp_path):
    frames_dir = tmp_path / "frames"
    frames_dir.mkdir()
    _make_frames(frames_dir, 2)

    nested_output = tmp_path / "does" / "not" / "exist" / "fight.mp4"
    builder = VideoBuilder()
    result = asyncio.run(
        builder.build_video(frame_directory=str(frames_dir), output_path=str(nested_output), frame_count=2)
    )
    assert result.video_path == str(nested_output)
    assert nested_output.exists()


# --- Error handling, against real conditions --------------------------------


def test_missing_ffmpeg_binary_raises(tmp_path):
    frames_dir = tmp_path / "frames"
    frames_dir.mkdir()
    _make_frames(frames_dir, 1)

    builder = VideoBuilder(ffmpeg_binary="definitely_not_a_real_ffmpeg_binary")
    with pytest.raises(VideoBuilderError, match="not found"):
        asyncio.run(
            builder.build_video(frame_directory=str(frames_dir), output_path=str(tmp_path / "out.mp4"), frame_count=1)
        )


def test_missing_first_frame_raises_without_invoking_ffmpeg(tmp_path):
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    builder = VideoBuilder()
    with pytest.raises(VideoBuilderError, match="not found"):
        asyncio.run(
            builder.build_video(frame_directory=str(empty_dir), output_path=str(tmp_path / "out.mp4"), frame_count=5)
        )


def test_real_ffmpeg_failure_surfaces_as_video_builder_error(tmp_path):
    """Odd (non-even) dimensions are genuinely incompatible with
    yuv420p encoding — a real ffmpeg failure, not a simulated one."""
    frames_dir = tmp_path / "frames"
    frames_dir.mkdir()
    _make_frames(frames_dir, 1, size=(101, 101))

    builder = VideoBuilder()
    with pytest.raises(VideoBuilderError):
        asyncio.run(
            builder.build_video(
                frame_directory=str(frames_dir), output_path=str(tmp_path / "out.mp4"),
                frame_count=1, width=101, height=101,
            )
        )


def test_no_partial_output_left_when_ffmpeg_fails(tmp_path):
    frames_dir = tmp_path / "frames"
    frames_dir.mkdir()
    _make_frames(frames_dir, 1, size=(101, 101))
    output_path = tmp_path / "out.mp4"

    builder = VideoBuilder()
    with pytest.raises(VideoBuilderError):
        asyncio.run(
            builder.build_video(
                frame_directory=str(frames_dir), output_path=str(output_path),
                frame_count=1, width=101, height=101,
            )
        )
    assert not output_path.exists() or output_path.stat().st_size == 0


# --- Frame ordering is respected (visually verifiable) -----------------------


def test_frames_are_assembled_in_correct_numeric_order(tmp_path):
    """Uses ffprobe to extract the first frame's average color and
    confirms it matches frame0001's color, not some other frame's —
    a real check that frame ordering (not just count) is correct."""
    frames_dir = tmp_path / "frames"
    frames_dir.mkdir()
    # frame0001 = pure red, frame0002 = pure green
    Image.new("RGB", (64, 64), color=(255, 0, 0)).save(frames_dir / "frame0001.png")
    Image.new("RGB", (64, 64), color=(0, 255, 0)).save(frames_dir / "frame0002.png")

    builder = VideoBuilder()
    result = asyncio.run(
        builder.build_video(
            frame_directory=str(frames_dir), output_path=str(tmp_path / "fight.mp4"),
            frame_count=2, width=64, height=64, frame_duration_seconds=1.0, fps=1,
        )
    )

    # Extract the very first output frame as a PNG and check its color.
    extracted = tmp_path / "extracted.png"
    subprocess.run(
        ["ffmpeg", "-y", "-i", result.video_path, "-vframes", "1", str(extracted)],
        capture_output=True, check=True,
    )
    first_frame_color = Image.open(extracted).convert("RGB").getpixel((32, 32))
    # H.264/yuv420p encoding is lossy — RGB->YUV->RGB introduces a few
    # units of rounding error, so this checks "clearly red, not green"
    # rather than bit-exact (255, 0, 0). That's what actually confirms
    # frame ordering; pixel-perfect color preservation through a lossy
    # codec isn't a real property to assert on.
    red, green, blue = first_frame_color
    assert red > 240
    assert green < 15
    assert blue < 15


# --- concatenate_clips (Sprint 4 Prompt 2: assembling animated clips) -------


async def _make_single_frame_clip(builder, tmp_path, name: str, color, duration=1.0, size=(320, 240)):
    """Builds one tiny real clip via build_video, for use as
    concatenate_clips input — mirrors how AnimationPipeline's
    MockAnimationProvider actually produces its per-shot clips."""
    frame_dir = tmp_path / f"{name}_frames"
    frame_dir.mkdir()
    Image.new("RGB", size, color=color).save(frame_dir / "frame0001.png")
    result = await builder.build_video(
        frame_directory=str(frame_dir), output_path=str(tmp_path / f"{name}.mp4"),
        frame_count=1, width=size[0], height=size[1], frame_duration_seconds=duration,
    )
    return result.video_path


def test_concatenate_clips_produces_a_real_valid_mp4(tmp_path):
    builder = VideoBuilder()
    clip1 = asyncio.run(_make_single_frame_clip(builder, tmp_path, "c1", (255, 0, 0)))
    clip2 = asyncio.run(_make_single_frame_clip(builder, tmp_path, "c2", (0, 255, 0)))

    result = asyncio.run(
        builder.concatenate_clips(
            clip_paths=[clip1, clip2], output_path=str(tmp_path / "final.mp4"),
            total_duration_seconds=2.0, width=320, height=240,
        )
    )
    assert (tmp_path / "final.mp4").exists()
    probe = _probe(result.video_path)
    video_stream = next(s for s in probe["streams"] if s["codec_type"] == "video")
    assert video_stream["width"] == 320
    assert video_stream["height"] == 240


def test_concatenate_clips_preserves_total_duration(tmp_path):
    builder = VideoBuilder()
    clip1 = asyncio.run(_make_single_frame_clip(builder, tmp_path, "c1", (255, 0, 0), duration=1.5))
    clip2 = asyncio.run(_make_single_frame_clip(builder, tmp_path, "c2", (0, 255, 0), duration=2.5))
    clip3 = asyncio.run(_make_single_frame_clip(builder, tmp_path, "c3", (0, 0, 255), duration=1.0))

    result = asyncio.run(
        builder.concatenate_clips(
            clip_paths=[clip1, clip2, clip3], output_path=str(tmp_path / "final.mp4"),
            total_duration_seconds=5.0, width=320, height=240,
        )
    )
    assert result.duration_seconds == 5.0
    probe = _probe(result.video_path)
    actual_duration = float(probe["format"]["duration"])
    assert abs(actual_duration - 5.0) < 0.5


def test_concatenate_clips_preserves_order_of_multiple_distinct_segments(tmp_path):
    """The core acceptance property: three visually distinct clips,
    concatenated, must appear in the final video in the same order —
    not merged, not reordered, not just the first clip repeated."""
    builder = VideoBuilder()
    clip1 = asyncio.run(_make_single_frame_clip(builder, tmp_path, "c1", (255, 0, 0)))
    clip2 = asyncio.run(_make_single_frame_clip(builder, tmp_path, "c2", (0, 255, 0)))
    clip3 = asyncio.run(_make_single_frame_clip(builder, tmp_path, "c3", (0, 0, 255)))

    result = asyncio.run(
        builder.concatenate_clips(
            clip_paths=[clip1, clip2, clip3], output_path=str(tmp_path / "final.mp4"),
            total_duration_seconds=3.0, width=320, height=240,
        )
    )

    extracted_dir = tmp_path / "extracted"
    extracted_dir.mkdir()
    subprocess.run(
        ["ffmpeg", "-y", "-i", result.video_path, "-vf", "fps=1", str(extracted_dir / "f%02d.png")],
        capture_output=True, check=True,
    )
    frames = sorted(extracted_dir.iterdir())
    assert len(frames) == 3

    def _dominant(path):
        r, g, b = Image.open(path).convert("RGB").getpixel((160, 120))
        return max(((r, "red"), (g, "green"), (b, "blue")), key=lambda pair: pair[0])[1]

    assert [_dominant(f) for f in frames] == ["red", "green", "blue"]


def test_concatenate_clips_raises_on_empty_list():
    builder = VideoBuilder()
    with pytest.raises(VideoBuilderError, match="No clips"):
        asyncio.run(
            builder.concatenate_clips(clip_paths=[], output_path="/tmp/out.mp4", total_duration_seconds=1.0)
        )


def test_concatenate_clips_raises_on_missing_clip(tmp_path):
    builder = VideoBuilder()
    with pytest.raises(VideoBuilderError, match="not found"):
        asyncio.run(
            builder.concatenate_clips(
                clip_paths=[str(tmp_path / "does_not_exist.mp4")],
                output_path=str(tmp_path / "out.mp4"), total_duration_seconds=1.0,
            )
        )


def test_concatenate_clips_cleans_up_its_temp_list_file(tmp_path):
    builder = VideoBuilder()
    clip1 = asyncio.run(_make_single_frame_clip(builder, tmp_path, "c1", (255, 0, 0)))

    asyncio.run(
        builder.concatenate_clips(
            clip_paths=[clip1], output_path=str(tmp_path / "final.mp4"),
            total_duration_seconds=1.0, width=320, height=240,
        )
    )
    leftover = list(tmp_path.glob(".*_concat_list.txt"))
    assert leftover == []
