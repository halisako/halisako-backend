"""Shared vocabulary and per-style configuration for the Workflow Builder.

Two distinct things live in this module, deliberately kept together:

1. The **input contract** this whole subpackage builds against —
   `CinematicSequence` and everything it's made of (`CinematicActor`,
   `CameraDirection`, `VisualEffect`, `ActorAction`, `AudioDirection`,
   `CinematicBeat`), plus `StyleProfile`. **These types are proposed,
   not confirmed against an authoritative source.** No definition for
   any of them exists elsewhere in the codebase available to this
   module's author. They are reconstructed from the vocabulary the
   Workflow Builder's own specification enumerates (camera shot types,
   visual effect types, actor action types), which gives reasonable
   confidence in the *vocabulary*, but the exact *shape* (which fields
   exist, how beats relate to actors, etc.) is a design proposal. If
   an authoritative CinematicSequence definition exists upstream (e.g.
   in the Halisako Cinematic Engine that is said to already produce
   one), this module should be reconciled against it before anything
   downstream of the Workflow Builder is treated as load-bearing.

   `StyleProfile` here is intentionally its own type, not an import of
   any single product's style schema — this subpackage lives under
   `core/rendering/`, which is cross-product infrastructure, and must
   not depend on a specific product's package. Its shape (weapons,
   powers, environment, visual_effects, finisher) mirrors the style
   concept used elsewhere in the platform, but any product wanting to
   call `WorkflowBuilder.build()` is responsible for translating its
   own style representation into this one first.

2. The **per-style template registry** (section 12 of the spec):
   default model, lighting, camera defaults, render defaults,
   scheduler, sampler, CFG, steps, frame interpolation, fps, and
   resolution for each of the five supported styles.

Both live here, rather than split further, because every other module
in this subpackage needs at least one of them, and putting them in one
dependency-free module is what keeps the subpackage's import graph
acyclic — see this package's `__init__.py` for the full dependency
order.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field

from core.rendering.workflow.exceptions import UnknownStyleError

# =====================================================================
# Style identity
# =====================================================================


class RenderStyleId(str, Enum):
    """The five visual styles the Workflow Builder currently supports.

    Adding a sixth style means adding a WorkflowTemplate below, not
    changing any builder's logic — every builder reads its per-style
    defaults from the template, never from a hardcoded branch on this
    enum.
    """

    ANIME = "anime"
    FANTASY = "fantasy"
    SCIFI = "scifi"
    MODERN_WARFARE = "modern_warfare"
    SUPERHERO = "superhero"


class StyleProfile(BaseModel):
    """A rendering-layer style descriptor — see module docstring for
    why this is not the same class as any single product's own style
    schema.
    """

    style: RenderStyleId = Field(..., description="Which of the five supported styles this is.")
    weapons: list[str] = Field(
        default_factory=list, description="Genre-appropriate weapon vocabulary for this battle."
    )
    powers: list[str] = Field(
        default_factory=list, description="Genre-appropriate ability/power vocabulary."
    )
    environment: str = Field(default="", description="Genre-appropriate setting description.")
    visual_effects: list[str] = Field(
        default_factory=list, description="Genre-appropriate visual effect vocabulary."
    )
    finisher: str = Field(default="", description="Description of the sequence's finishing beat.")


# =====================================================================
# Cinematic vocabulary (proposed — see module docstring)
# =====================================================================


class CameraShotType(str, Enum):
    """Every camera shot type the Camera Builder must support."""

    WIDE = "wide"
    MEDIUM = "medium"
    CLOSE = "close"
    EXTREME_CLOSE = "extreme_close"
    OVERHEAD = "overhead"
    TRACKING = "tracking"
    ORBIT = "orbit"
    HANDHELD = "handheld"
    SHAKE = "shake"
    CRANE = "crane"
    DOLLY = "dolly"
    ZOOM = "zoom"


class VisualEffectType(str, Enum):
    """Every visual effect type the Visual Effects Builder must support."""

    ENERGY = "energy"
    FIRE = "fire"
    MAGIC = "magic"
    DUST = "dust"
    BLOOD = "blood"
    LIGHTNING = "lightning"
    SHOCKWAVE = "shockwave"
    EMBERS = "embers"
    SMOKE = "smoke"
    PARTICLES = "particles"
    DEBRIS = "debris"
    WIND = "wind"


class ActorActionType(str, Enum):
    """Every actor action type the Animation Builder must map to a
    clip."""

    ADVANCE = "advance"
    ATTACK = "attack"
    BLOCK = "block"
    ROLL = "roll"
    JUMP = "jump"
    COUNTER = "counter"
    SPECIAL = "special"
    DEATH = "death"
    FINISH = "finish"


class IntensityLevel(str, Enum):
    """Abstract, renderer-independent intensity used throughout the
    cinematic vocabulary and mapped to concrete render parameters by
    the ParameterMapper."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    EXTREME = "extreme"


class CinematicActor(BaseModel):
    """One participant in the cinematic sequence.

    `character_model`, `weapon`, `armor`, and `lora` are asset
    *identifiers* — semantic names the AssetResolver looks up, never
    file paths. The Cinematic Engine is assumed to have already
    decided who this actor is and what they're carrying; the Workflow
    Builder only resolves those decisions into renderer inputs.
    """

    actor_id: str = Field(..., min_length=1, description="Unique identifier within the sequence.")
    display_name: str = Field(..., min_length=1, description="Human-readable name for logs/UI.")
    character_model: str = Field(
        ..., min_length=1, description="Semantic identifier for this actor's base character model."
    )
    weapon: str | None = Field(default=None, description="Semantic identifier for a carried weapon, if any.")
    armor: str | None = Field(default=None, description="Semantic identifier for worn armor, if any.")
    lora: str | None = Field(
        default=None, description="Semantic identifier for a character-specific LoRA, if any."
    )
    pose_reference: str | None = Field(
        default=None, description="Semantic identifier for a reference pose/rig, if any."
    )
    facial_preset: str | None = Field(
        default=None, description="Semantic identifier for a facial expression preset, if any."
    )


class CameraDirection(BaseModel):
    """One camera instruction — what shot, on whom, how intense."""

    shot_type: CameraShotType = Field(..., description="Which of the 12 supported shot types this is.")
    focus_actor_id: str | None = Field(
        default=None, description="Which actor this shot is centered on, if any (None for an "
        "establishing/environment shot)."
    )
    intensity: IntensityLevel = Field(
        default=IntensityLevel.MEDIUM,
        description="Drives motion blur / shake / movement-spline aggressiveness.",
    )


class VisualEffect(BaseModel):
    """One visual effect instance to render."""

    effect_type: VisualEffectType = Field(..., description="Which of the 12 supported effect types this is.")
    intensity: IntensityLevel = Field(default=IntensityLevel.MEDIUM, description="Effect intensity/scale.")
    origin_actor_id: str | None = Field(
        default=None, description="Which actor this effect originates from, if any."
    )


class ActorAction(BaseModel):
    """One action a specific actor performs within a beat."""

    actor_id: str = Field(..., min_length=1, description="Which actor performs this action.")
    action: ActorActionType = Field(..., description="Which of the 9 supported action types this is.")
    intensity: IntensityLevel = Field(default=IntensityLevel.MEDIUM, description="Action intensity.")


class AudioDirection(BaseModel):
    """Audio intent for a beat. No actual audio is generated here —
    only enough intent for the Audio Builder to emit workflow nodes."""

    mood: str = Field(..., min_length=1, description="Musical/emotional mood for this beat's audio.")
    impact_beats: int = Field(default=0, ge=0, description="Number of percussive impact hits to place.")
    voice_placeholder: bool = Field(
        default=False, description="Whether to reserve a voice/dialogue track slot for this beat."
    )


class CinematicBeat(BaseModel):
    """One beat (moment) within the sequence — the unit every builder
    stage ultimately reads from."""

    beat_id: str = Field(..., min_length=1, description="Unique identifier within the sequence.")
    timestamp_start: float = Field(
        ..., ge=0, description="When this beat starts, in seconds from sequence start."
    )
    duration_seconds: float = Field(..., gt=0, description="How long this beat lasts, in seconds.")
    actions: list[ActorAction] = Field(
        default_factory=list, description="Actions performed during this beat."
    )
    camera: CameraDirection = Field(..., description="The camera instruction for this beat.")
    effects: list[VisualEffect] = Field(
        default_factory=list, description="Visual effects active during this beat."
    )
    audio: AudioDirection | None = Field(default=None, description="Audio intent for this beat, if any.")
    emotion: str | None = Field(
        default=None, description="What the Cinematic Engine decided the audience should feel "
        "during this beat — informs the Lighting Builder's mood reasoning."
    )


class CinematicSequence(BaseModel):
    """The complete cinematic intent the Workflow Builder converts
    into a renderer workflow. Produced upstream (by whatever engine
    decided the actors, choreography, timing, camera, effects, and
    emotions) — the Workflow Builder only consumes this, it never
    invents story.
    """

    sequence_id: str = Field(..., min_length=1, description="Unique identifier for this sequence.")
    actors: list[CinematicActor] = Field(
        ..., min_length=1, description="Every actor appearing in the sequence."
    )
    beats: list[CinematicBeat] = Field(
        ..., min_length=1, description="The ordered beats making up the sequence."
    )
    environment: str = Field(
        ..., min_length=1, description="Semantic identifier for the environment/setting."
    )
    total_duration_seconds: float = Field(..., gt=0, description="Total sequence duration, in seconds.")
    seed: int = Field(
        ...,
        description=(
            "Pinned random seed for this sequence's generation. Required, not optional, "
            "matching the determinism this platform's rendering infrastructure requires "
            "throughout (see core/rendering/job_models.py's SceneJob.seed) — a seed decided "
            "at generation time rather than pinned upstream would make the resulting workflow "
            "non-reproducible."
        ),
    )


# =====================================================================
# Per-style workflow templates (section 12)
# =====================================================================


class LightingDefaults(BaseModel):
    """Baseline lighting configuration for a style, before any
    per-beat mood adjustment the Lighting Builder applies."""

    key_light_intensity: float = Field(..., gt=0)
    fill_light_ratio: float = Field(
        ..., gt=0, le=1, description="Fill light intensity as a ratio of key light."
    )
    rim_light_intensity: float = Field(..., ge=0)
    ambient_intensity: float = Field(..., ge=0)
    color_temperature_kelvin: float = Field(..., gt=0)
    contrast: float = Field(..., gt=0)


class CameraDefaults(BaseModel):
    """Baseline camera configuration for a style."""

    default_fov_degrees: float = Field(..., gt=0, lt=180)
    motion_blur_strength: float = Field(..., ge=0, le=1)


class RenderDefaults(BaseModel):
    """Baseline renderer sampling configuration for a style."""

    scheduler: str = Field(..., min_length=1)
    sampler: str = Field(..., min_length=1)
    cfg_scale: float = Field(..., gt=0)
    steps: int = Field(..., gt=0)
    frame_interpolation: bool = Field(...)
    fps: int = Field(..., gt=0)
    resolution: tuple[int, int] = Field(..., description="(width, height) in pixels.")


class WorkflowTemplate(BaseModel):
    """The complete default configuration for one style."""

    style: RenderStyleId
    default_model: str = Field(
        ..., min_length=1, description="Semantic identifier for the base checkpoint model."
    )
    default_vae: str = Field(..., min_length=1, description="Semantic identifier for the default VAE.")
    lighting: LightingDefaults
    camera: CameraDefaults
    render: RenderDefaults


_TEMPLATES: dict[RenderStyleId, WorkflowTemplate] = {
    RenderStyleId.ANIME: WorkflowTemplate(
        style=RenderStyleId.ANIME,
        default_model="anime_diffusion_base",
        default_vae="anime_vae",
        lighting=LightingDefaults(
            key_light_intensity=1.2, fill_light_ratio=0.35, rim_light_intensity=0.8,
            ambient_intensity=0.3, color_temperature_kelvin=6500, contrast=1.15,
        ),
        camera=CameraDefaults(default_fov_degrees=35, motion_blur_strength=0.25),
        render=RenderDefaults(
            scheduler="karras", sampler="dpmpp_2m", cfg_scale=7.0, steps=28,
            frame_interpolation=True, fps=24, resolution=(1024, 576),
        ),
    ),
    RenderStyleId.FANTASY: WorkflowTemplate(
        style=RenderStyleId.FANTASY,
        default_model="fantasy_diffusion_base",
        default_vae="fantasy_vae",
        lighting=LightingDefaults(
            key_light_intensity=1.0, fill_light_ratio=0.3, rim_light_intensity=0.5,
            ambient_intensity=0.4, color_temperature_kelvin=4500, contrast=1.05,
        ),
        camera=CameraDefaults(default_fov_degrees=40, motion_blur_strength=0.2),
        render=RenderDefaults(
            scheduler="karras", sampler="dpmpp_2m", cfg_scale=7.5, steps=32,
            frame_interpolation=True, fps=24, resolution=(1024, 576),
        ),
    ),
    RenderStyleId.SCIFI: WorkflowTemplate(
        style=RenderStyleId.SCIFI,
        default_model="scifi_diffusion_base",
        default_vae="scifi_vae",
        lighting=LightingDefaults(
            key_light_intensity=1.3, fill_light_ratio=0.2, rim_light_intensity=1.0,
            ambient_intensity=0.15, color_temperature_kelvin=8000, contrast=1.3,
        ),
        camera=CameraDefaults(default_fov_degrees=45, motion_blur_strength=0.35),
        render=RenderDefaults(
            scheduler="karras", sampler="dpmpp_2m_sde", cfg_scale=6.5, steps=30,
            frame_interpolation=True, fps=30, resolution=(1280, 720),
        ),
    ),
    RenderStyleId.MODERN_WARFARE: WorkflowTemplate(
        style=RenderStyleId.MODERN_WARFARE,
        default_model="realistic_diffusion_base",
        default_vae="realistic_vae",
        lighting=LightingDefaults(
            key_light_intensity=1.0, fill_light_ratio=0.25, rim_light_intensity=0.4,
            ambient_intensity=0.25, color_temperature_kelvin=5600, contrast=1.2,
        ),
        camera=CameraDefaults(default_fov_degrees=50, motion_blur_strength=0.4),
        render=RenderDefaults(
            scheduler="normal", sampler="dpmpp_2m_sde", cfg_scale=6.0, steps=25,
            frame_interpolation=False, fps=30, resolution=(1280, 720),
        ),
    ),
    RenderStyleId.SUPERHERO: WorkflowTemplate(
        style=RenderStyleId.SUPERHERO,
        default_model="superhero_diffusion_base",
        default_vae="superhero_vae",
        lighting=LightingDefaults(
            key_light_intensity=1.4, fill_light_ratio=0.4, rim_light_intensity=1.1,
            ambient_intensity=0.3, color_temperature_kelvin=6000, contrast=1.25,
        ),
        camera=CameraDefaults(default_fov_degrees=38, motion_blur_strength=0.3),
        render=RenderDefaults(
            scheduler="karras", sampler="dpmpp_2m", cfg_scale=7.0, steps=30,
            frame_interpolation=True, fps=24, resolution=(1152, 648),
        ),
    ),
}


def get_template(style: RenderStyleId) -> WorkflowTemplate:
    """Returns the WorkflowTemplate for the given style.

    Raises:
        UnknownStyleError: If no template is registered for `style`.
    """
    template = _TEMPLATES.get(style)
    if template is None:
        raise UnknownStyleError(f"No WorkflowTemplate registered for style {style!r}.")
    return template


def registered_styles() -> list[RenderStyleId]:
    """Returns every style currently registered with a template."""
    return list(_TEMPLATES.keys())
