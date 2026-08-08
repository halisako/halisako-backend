"""Schemas for the Cinematic Timeline Engine.

A ShotTimeline is a deterministic shot-by-shot visual plan for a fight
scene — the layer between narrative (FightStory) and any future actual
rendering. Nothing here generates an image or video; a Shot describes
*what a camera crew would be told to set up*, not pixels.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class ShotType(str, Enum):
    """The narrative role a shot plays in the scene's arc — distinct
    from camera framing (that's CameraAngle/CameraMotion). Every
    timeline starts with ESTABLISHING and ends with AFTERMATH; what's
    in between depends on the battle."""

    ESTABLISHING = "establishing"
    BUILD_UP = "build_up"
    EXCHANGE = "exchange"
    TURNING_POINT = "turning_point"
    CLIMAX = "climax"
    AFTERMATH = "aftermath"


class CameraAngle(str, Enum):
    """Camera framing distance/position for a shot."""

    WIDE = "wide"
    MEDIUM = "medium"
    CLOSE_UP = "close_up"
    EXTREME_CLOSE_UP = "extreme_close_up"
    OVERHEAD = "overhead"
    LOW_ANGLE = "low_angle"
    HIGH_ANGLE = "high_angle"


class CameraMotion(str, Enum):
    """How the camera moves during a shot."""

    STATIC = "static"
    PAN = "pan"
    TRACK = "track"
    ORBIT = "orbit"
    PUSH_IN = "push_in"
    PULL_OUT = "pull_out"
    SHAKE = "shake"
    CRANE = "crane"


class ShotFocus(str, Enum):
    """Which side of the battle a shot centers on."""

    WHITE = "white"
    BLACK = "black"
    BOTH = "both"
    ENVIRONMENT = "environment"


class Shot(BaseModel):
    """One planned shot within the fight scene's visual timeline."""

    shot_id: str = Field(..., min_length=1, description="Unique identifier for this shot within its timeline.")
    sequence_order: int = Field(..., ge=1, description="1-indexed position of this shot within the timeline.")
    shot_type: ShotType = Field(..., description="The narrative role this shot plays in the scene's arc.")
    camera_angle: CameraAngle = Field(..., description="Camera framing distance/position for this shot.")
    camera_motion: CameraMotion = Field(..., description="How the camera moves during this shot.")
    focus: ShotFocus = Field(..., description="Which side of the battle this shot centers on.")
    duration_seconds: float = Field(..., gt=0, description="Planned duration of this shot, in seconds.")
    environment: str = Field(..., min_length=1, description="Where this shot takes place.")
    lighting: str = Field(..., min_length=1, description="Lighting description for this shot.")
    mood: str = Field(..., min_length=1, description="The emotional register this shot should carry.")
    source_moves: list[str] = Field(
        default_factory=list,
        description="Chess move labels (e.g. \"4. Qxf7#\") this shot dramatizes, if any. Empty for "
        "shots not tied to a specific move (e.g. a pure establishing or aftermath shot).",
    )
    description: str = Field(..., min_length=1, description="What happens in this shot, in plain language.")


class ShotTimeline(BaseModel):
    """The complete, ordered shot-by-shot visual plan for a fight scene."""

    shots: list[Shot] = Field(..., min_length=1, description="Every shot, in sequence_order.")
    total_duration_seconds: float = Field(..., gt=0, description="Sum of every shot's duration_seconds.")
    shot_count: int = Field(..., ge=1, description="Number of shots in this timeline — len(shots).")

# --- Scene Composer (Sprint 3) -----------------------------------------------
#
# The Scene Composer never generates prompts or images — it produces
# structured, reusable continuity data and attaches it to every shot,
# so a downstream renderer processing shots one at a time (or out of
# order) never has to reconcile drift between them. Every value below
# is computed once per battle and held constant across the whole
# scene — that constancy *is* the continuity guarantee, not an
# implementation detail of it.


class FighterAppearance(BaseModel):
    """A fighter's persistent physical description — identical across
    every shot they appear in."""

    hair: str = Field(..., min_length=1, description="Hair style/color description.")
    facial_features: str = Field(..., min_length=1, description="Facial feature description.")
    clothing: str = Field(..., min_length=1, description="Clothing description.")
    armor: str = Field(..., min_length=1, description="Armor description.")
    weapon: str = Field(..., min_length=1, description="Weapon carried, as a genre-appropriate identifier.")


class ArenaContinuity(BaseModel):
    """The persistent physical setting — identical across every shot
    in the scene."""

    layout: str = Field(..., min_length=1, description="Physical layout/geography of the arena.")
    weather: str = Field(..., min_length=1, description="Weather conditions throughout the scene.")
    time_of_day: str = Field(..., min_length=1, description="Time of day throughout the scene.")


class SceneContinuity(BaseModel):
    """The complete visual continuity bible for one fight scene: every
    field a renderer needs to keep every frame looking like it belongs
    to the same, single continuous scene.
    """

    white_fighter: FighterAppearance = Field(..., description="White fighter's persistent appearance.")
    black_fighter: FighterAppearance = Field(..., description="Black fighter's persistent appearance.")
    arena: ArenaContinuity = Field(..., description="The scene's persistent physical setting.")
    lighting_continuity: str = Field(
        ..., min_length=1,
        description="How lighting is kept consistent across every shot — the scene's lighting "
        "approach, not any one shot's individual lighting description (see Shot.lighting).",
    )
    cinematic_art_style: str = Field(..., min_length=1, description="The overall visual/art style for the scene.")
    color_palette: list[str] = Field(..., min_length=1, description="The scene's persistent color palette.")


class EnrichedShot(Shot):
    """A Shot enriched with the scene's persistent continuity
    information. Every field Shot already has is unchanged; `scene`
    is the only addition."""

    scene: SceneContinuity = Field(..., description="The scene continuity this shot must stay consistent with.")


class ComposedTimeline(BaseModel):
    """The Scene Composer's output: every shot from the ShotTimeline,
    enriched with scene continuity, plus that continuity once more at
    the top level for convenient reference."""

    shots: list[EnrichedShot] = Field(..., min_length=1, description="Every enriched shot, in sequence_order.")
    total_duration_seconds: float = Field(..., gt=0, description="Sum of every shot's duration_seconds.")
    shot_count: int = Field(..., ge=1, description="Number of shots in this timeline — len(shots).")
    scene_continuity: SceneContinuity = Field(
        ..., description="The same continuity data embedded in every shot, for convenient top-level access."
    )

# --- Prompt Generator (Sprint 3) ----------------------------------------------
#
# The Prompt Generator never calls an image API and never produces an
# image — it only assembles the text prompt a future renderer would
# send to one. `image_prompt` is a single ready-to-use string, not a
# structured object, because that's the actual input shape a
# text-to-image model's encoder expects — unlike FightStory.prompt
# (a labeled-section screenplay document for a human to read),
# `image_prompt` is written to be consumed directly.


class PromptedShot(EnrichedShot):
    """An EnrichedShot with its cinematic image-generation prompt
    attached. Every field EnrichedShot (and, through it, Shot) already
    has is unchanged; `image_prompt` is the only addition."""

    image_prompt: str = Field(
        ...,
        min_length=1,
        description="The complete, ready-to-use text-to-image prompt for this shot.",
    )


class PromptedTimeline(BaseModel):
    """The Prompt Generator's output: every shot from the
    ComposedTimeline, each now carrying its own image_prompt."""

    shots: list[PromptedShot] = Field(
        ...,
        min_length=1,
        description="Every prompted shot, in sequence_order.",
    )
    total_duration_seconds: float = Field(
        ...,
        gt=0,
        description="Sum of every shot's duration_seconds.",
    )
    shot_count: int = Field(
        ...,
        ge=1,
        description="Number of shots in this timeline — len(shots).",
    )
    scene_continuity: SceneContinuity = Field(
        ...,
        description="The same continuity data embedded in every shot, for convenient top-level access.",
    )
