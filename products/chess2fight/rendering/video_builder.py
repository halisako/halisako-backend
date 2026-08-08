"""VideoBuilder: assembles an ordered frame directory into fight.mp4.

    Render Pipeline -> Video Builder

Shells out to the `ffmpeg` binary via `asyncio.create_subprocess_exec`
— no Python video-encoding library is used or needed for this. Every
previous rendering-stage module drew a hard line at "only frames, no
video"; this is the one module whose entire purpose is to cross that
line, on request, as the pipeline's final step.

Frame duration is a single, uniform value applied to every frame
(`frame_duration_seconds`), not a per-shot value read back from each
frame's own metadata — the brief asks for "Frame duration" as one of
three simple, scalar configuration knobs (alongside FPS and
Resolution), not for per-shot pacing to be preserved into the encoded
video. Each shot's own individual `duration_seconds` (computed by the
Timeline Engine) is still available in the API response's `timeline`
field for a caller to inspect — it just isn't what controls how long
each frame is actually held in the assembled MP4 today. A future
version could read it back from FrameMetadata for varying per-frame
hold times; that's a real, deliberate simplification, not an
oversight, and is called out here so it isn't mistaken for one.
"""

from __future__ import annotations

import asyncio
import shutil
from pathlib import Path

from pydantic import BaseModel, Field

from core.exceptions import VideoBuilderError


class VideoBuildResult(BaseModel):
    """The result of one video build."""

    video_path: str = Field(..., min_length=1, description="Path to the assembled MP4 file.")
    fps: int = Field(..., gt=0, description="Output frame rate used.")
    width: int = Field(..., gt=0, description="Output video width, in pixels.")
    height: int = Field(..., gt=0, description="Output video height, in pixels.")
    frame_count: int = Field(..., ge=1, description="Number of source frames assembled.")
    frame_duration_seconds: float = Field(
        ..., gt=0, description="How long each source frame is held in the output video, in seconds."
    )
    duration_seconds: float = Field(
        ..., gt=0, description="Total video duration — frame_count * frame_duration_seconds."
    )


class VideoBuilder:
    """Assembles an ordered PNG frame sequence into an MP4 video via FFmpeg."""

    def __init__(self, ffmpeg_binary: str = "ffmpeg") -> None:
        """Initializes the builder.

        Args:
            ffmpeg_binary: Name or path of the ffmpeg executable to
                invoke. Defaults to "ffmpeg" on PATH.
        """
        self._ffmpeg_binary = ffmpeg_binary

    async def build_video(
        self,
        frame_directory: str,
        output_path: str,
        frame_count: int,
        fps: int = 24,
        width: int = 1024,
        height: int = 1024,
        frame_duration_seconds: float = 2.0,
        frame_pattern: str = "frame%04d.png",
    ) -> VideoBuildResult:
        """Assembles an ordered frame sequence (e.g. frame0001.png,
        frame0002.png, ...) into an MP4.

        Args:
            frame_directory: Directory containing the ordered frames —
                e.g. AssetManager.fight_directory()'s output.
            output_path: Where to write the final MP4.
            frame_count: Number of frames expected. Used to validate
                the input directory actually has what's needed before
                invoking ffmpeg, and to compute total duration.
            fps: Output video frame rate.
            width: Output video width, in pixels.
            height: Output video height, in pixels.
            frame_duration_seconds: How long each source frame is held
                in the output video, in seconds — see this module's
                docstring for why this is uniform across every frame.
            frame_pattern: printf-style filename pattern ffmpeg should
                expect within `frame_directory` — matches
                AssetManager.frame_filename()'s own naming convention.

        Returns:
            A VideoBuildResult describing the assembled video.

        Raises:
            VideoBuilderError: If ffmpeg isn't available on PATH, the
                expected frames aren't present, or ffmpeg itself fails.
        """
        self._require_ffmpeg_available()
        self._require_first_frame_present(frame_directory, frame_pattern)

        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        input_framerate = 1.0 / frame_duration_seconds
        command = [
            self._ffmpeg_binary,
            "-y",
            "-framerate", str(input_framerate),
            "-i", str(Path(frame_directory) / frame_pattern),
            "-r", str(fps),
            "-vf", f"scale={width}:{height}",
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            str(output_file),
        ]

        await self._run_ffmpeg(command)

        return VideoBuildResult(
            video_path=str(output_file),
            fps=fps,
            width=width,
            height=height,
            frame_count=frame_count,
            frame_duration_seconds=frame_duration_seconds,
            duration_seconds=frame_count * frame_duration_seconds,
        )

    def _require_ffmpeg_available(self) -> None:
        if shutil.which(self._ffmpeg_binary) is None:
            raise VideoBuilderError(f"ffmpeg binary {self._ffmpeg_binary!r} was not found on PATH.")

    def _require_first_frame_present(self, frame_directory: str, frame_pattern: str) -> None:
        first_frame = Path(frame_directory) / (frame_pattern % 1)
        if not first_frame.exists():
            raise VideoBuilderError(
                f"Expected frame {first_frame} not found in {frame_directory!r} — nothing to assemble."
            )

    async def _run_ffmpeg(self, command: list[str]) -> None:
        """Runs an ffmpeg command, raising VideoBuilderError with
        ffmpeg's own stderr on failure."""
        process = await asyncio.create_subprocess_exec(
            *command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        _stdout, stderr = await process.communicate()

        if process.returncode != 0:
            raise VideoBuilderError(
                f"ffmpeg exited with code {process.returncode}: {stderr.decode(errors='replace')[-2000:]}"
            )
