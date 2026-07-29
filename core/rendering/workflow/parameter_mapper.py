"""Maps abstract, renderer-independent cinematic language into concrete
render parameters.

Every mapping here is a pure function of an `IntensityLevel` (and, for
frame count, a duration/fps pair) — nothing renderer-specific appears
in this module. A ComfyUI adapter, a Modal adapter, and an Unreal
Engine adapter would all consume the same numbers this module
produces; how each of them uses "motion strength" or "particle
density" internally is entirely their own concern.
"""

from __future__ import annotations

from core.rendering.workflow.workflow_templates import IntensityLevel

# Every table below is keyed by IntensityLevel and nothing else,
# deliberately — see module docstring. Values are hand-tuned starting
# points, not derived from any renderer's actual behavior (there is no
# renderer wired up yet to tune them against).

_CFG_DELTA: dict[IntensityLevel, float] = {
    IntensityLevel.LOW: -1.0,
    IntensityLevel.MEDIUM: 0.0,
    IntensityLevel.HIGH: 1.0,
    IntensityLevel.EXTREME: 2.0,
}

_MOTION_STRENGTH: dict[IntensityLevel, float] = {
    IntensityLevel.LOW: 0.3,
    IntensityLevel.MEDIUM: 0.55,
    IntensityLevel.HIGH: 0.8,
    IntensityLevel.EXTREME: 1.0,
}

_CAMERA_SHAKE: dict[IntensityLevel, float] = {
    IntensityLevel.LOW: 0.0,
    IntensityLevel.MEDIUM: 0.15,
    IntensityLevel.HIGH: 0.4,
    IntensityLevel.EXTREME: 0.75,
}

_PARTICLE_DENSITY: dict[IntensityLevel, float] = {
    IntensityLevel.LOW: 0.2,
    IntensityLevel.MEDIUM: 0.5,
    IntensityLevel.HIGH: 0.8,
    IntensityLevel.EXTREME: 1.0,
}

# Multiplier on a beat's "natural" frame count (duration * fps) — higher
# intensity beats get proportionally more frames, so fast motion doesn't
# undersample.
_FRAME_COUNT_MULTIPLIER: dict[IntensityLevel, float] = {
    IntensityLevel.LOW: 1.0,
    IntensityLevel.MEDIUM: 1.1,
    IntensityLevel.HIGH: 1.25,
    IntensityLevel.EXTREME: 1.4,
}


class ParameterMapper:
    """Stateless mapper from cinematic intensity to concrete render
    parameters.

    Every method takes whatever base/template value applies (e.g. a
    style's default CFG scale) and an `IntensityLevel`, and returns
    the adjusted concrete value — never the other way around, so a
    style's own defaults always remain the starting point and
    intensity is always an adjustment on top of them.
    """

    def cfg_scale(self, base_cfg: float, intensity: IntensityLevel) -> float:
        """Returns a CFG scale adjusted for cinematic intensity.

        Higher intensity nudges CFG up (more literal adherence to the
        prompt, appropriate for a decisive, high-stakes beat); lower
        intensity nudges it down (more generative looseness,
        appropriate for a quiet, exploratory beat).
        """
        return max(1.0, base_cfg + _CFG_DELTA[intensity])

    def motion_strength(self, intensity: IntensityLevel) -> float:
        """Returns a 0-1 motion strength for animation/motion-module
        nodes."""
        return _MOTION_STRENGTH[intensity]

    def camera_shake(self, intensity: IntensityLevel) -> float:
        """Returns a 0-1 camera shake amount."""
        return _CAMERA_SHAKE[intensity]

    def particle_density(self, intensity: IntensityLevel) -> float:
        """Returns a 0-1 particle density for visual effect nodes."""
        return _PARTICLE_DENSITY[intensity]

    def frame_count(self, duration_seconds: float, fps: int, intensity: IntensityLevel) -> int:
        """Returns how many frames a beat of the given duration and
        fps should render, adjusted for intensity.

        Args:
            duration_seconds: The beat's duration.
            fps: The style's frame rate.
            intensity: The beat's cinematic intensity.
        """
        base_frames = duration_seconds * fps
        return max(1, round(base_frames * _FRAME_COUNT_MULTIPLIER[intensity]))
