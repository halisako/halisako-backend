"""Prompt Generator: ComposedTimeline -> PromptedTimeline.

Runs immediately after the Scene Composer:

    Timeline Engine -> Scene Composer -> Prompt Generator

Its one job is assembling a ready-to-use text-to-image prompt for
every shot, from data every earlier stage already computed. It
generates no images and calls no image API — every output is a plain
string, stored on the shot it describes.

`ComposedTimeline` is deliberately the *only* input this module needs:
by this point in the pipeline, everything a prompt requires (fighter
appearance, arena, weather, lighting continuity, art style, palette)
is already embedded on every `EnrichedShot` via its own fields and its
`scene`. Nothing here reaches back to BattleIntelligence, StyleProfile,
or FightStory directly.

On the "Anime cinematic style" requirement specifically: the style
element in every generated prompt comes from
`shot.scene.cinematic_art_style` — genre-appropriate (anime, fantasy,
modern warfare, superhero, or sci-fi) — not a hardcoded "anime"
string. Hardcoding "anime" regardless of the actual selected style
would silently break the multi-style support every earlier stage of
this pipeline (style_engine.py, battle_mode_engine.py, the Scene
Composer) was built to preserve. See this module's test suite for the
explicit check that a non-anime style never contains the word "anime"
in its generated prompts.

Deterministic throughout: the same ComposedTimeline always produces
the exact same PromptedTimeline. No randomness, no AI calls.
"""

from __future__ import annotations

from products.chess2fight.cinematic.schemas import (
    CameraAngle,
    CameraMotion,
    ComposedTimeline,
    EnrichedShot,
    FighterAppearance,
    PromptedShot,
    PromptedTimeline,
    ShotFocus,
)

# Fixed, genre-independent quality-boosting tags, appended to every
# prompt exactly as a real text-to-image prompt would use them —
# these are intentionally NOT style-specific; the style element
# (`cinematic_art_style`) already carries genre flavor, so duplicating
# it here would blur the two apart rather than reinforce either.
_QUALITY_MODIFIERS = (
    "highly detailed, sharp focus, dramatic lighting, cinematic composition, "
    "8k resolution, masterpiece quality, trending on artstation"
)

_ANGLE_PHRASE: dict[CameraAngle, str] = {
    CameraAngle.WIDE: "wide shot",
    CameraAngle.MEDIUM: "medium shot",
    CameraAngle.CLOSE_UP: "close-up shot",
    CameraAngle.EXTREME_CLOSE_UP: "extreme close-up shot",
    CameraAngle.OVERHEAD: "overhead shot",
    CameraAngle.LOW_ANGLE: "low-angle shot",
    CameraAngle.HIGH_ANGLE: "high-angle shot",
}

_MOVEMENT_PHRASE: dict[CameraMotion, str] = {
    CameraMotion.STATIC: "static, locked-off camera",
    CameraMotion.PAN: "camera panning smoothly",
    CameraMotion.TRACK: "camera tracking alongside the action",
    CameraMotion.ORBIT: "camera orbiting around the subject",
    CameraMotion.PUSH_IN: "camera slowly pushing in",
    CameraMotion.PULL_OUT: "camera pulling back",
    CameraMotion.SHAKE: "handheld, shaking camera",
    CameraMotion.CRANE: "sweeping crane camera movement",
}

_COMPOSITION_PHRASE: dict[CameraAngle, str] = {
    CameraAngle.WIDE: "wide, symmetrical composition with generous negative space",
    CameraAngle.MEDIUM: "balanced medium-shot composition",
    CameraAngle.CLOSE_UP: "tight, off-center composition drawing focus to the subject",
    CameraAngle.EXTREME_CLOSE_UP: "extreme close framing filling the frame",
    CameraAngle.OVERHEAD: "top-down composition emphasizing scale",
    CameraAngle.LOW_ANGLE: "low-angle composition emphasizing power and height",
    CameraAngle.HIGH_ANGLE: "high-angle composition emphasizing vulnerability",
}


def _fighter_descriptor(fighter: FighterAppearance, prominent: bool) -> str:
    """Describes one fighter's persistent appearance.

    Every element (hair, facial features, clothing, armor, weapon) is
    always included for both fighters in every shot — the brief
    requires clothing and weapons in every prompt, not just for
    whichever fighter the shot happens to focus on. `prominent` only
    changes framing emphasis (main compositional subject vs. present
    in the background), never which details are included.
    """
    full_description = (
        f"{fighter.hair}, {fighter.facial_features}, wearing {fighter.clothing} and {fighter.armor}, "
        f"wielding a {fighter.weapon}"
    )
    if prominent:
        return f"a fighter with {full_description}"
    return f"another fighter in the background with {full_description}"


def _character_clause(shot: EnrichedShot) -> str:
    """Builds the subject-description clause. Both fighters are always
    mentioned somewhere in the prompt, regardless of focus — a
    background fighter is still part of the frame."""
    white, black = shot.scene.white_fighter, shot.scene.black_fighter
    if shot.focus == ShotFocus.WHITE:
        return f"{_fighter_descriptor(white, prominent=True)}, {_fighter_descriptor(black, prominent=False)}"
    if shot.focus == ShotFocus.BLACK:
        return f"{_fighter_descriptor(black, prominent=True)}, {_fighter_descriptor(white, prominent=False)}"
    # BOTH or ENVIRONMENT: describe both roughly equally facing off.
    return f"{_fighter_descriptor(white, prominent=True)} facing {_fighter_descriptor(black, prominent=True)}"


def _environment_clause(shot: EnrichedShot) -> str:
    """Combines the shot's own environment description — which
    Timeline Engine already built from style + battle-mode framing —
    with the scene's persistent weather/time-of-day continuity.

    Deliberately does NOT also append `shot.scene.arena.layout`: that
    value is exactly the battle-mode environment description already
    folded into `shot.environment` upstream (see
    timeline_engine.py's `_extract_section` usage and
    scene_composer.py's `compose_scene`, which sets `arena.layout`
    directly from the same source) — appending it again produced a
    literal duplicate parenthetical during testing ("a fateful final
    arena (a fateful final arena)") before this was caught.

    Also checks whether `shot.description` already states the
    environment verbatim — true for Timeline Engine's own
    establishing-shot description ("Setting the scene: ..., in
    {environment}.") — and drops the redundant restatement here rather
    than emitting the same environment name twice in one prompt.
    """
    arena = shot.scene.arena
    if shot.environment in shot.description:
        return f"{arena.time_of_day}, {arena.weather}"
    return f"in {shot.environment}, {arena.time_of_day}, {arena.weather}"


def _camera_clause(shot: EnrichedShot) -> tuple[str, str, str]:
    """Returns (angle phrase, movement phrase, composition phrase)."""
    return (
        _ANGLE_PHRASE[shot.camera_angle],
        _MOVEMENT_PHRASE[shot.camera_motion],
        _COMPOSITION_PHRASE[shot.camera_angle],
    )


def _lighting_clause(shot: EnrichedShot) -> str:
    """Combines the shot's own per-shot lighting description with the
    scene's persistent lighting-continuity approach."""
    return f"{shot.lighting}; {shot.scene.lighting_continuity}"


def _build_prompt(shot: EnrichedShot) -> str:
    """Assembles the complete text-to-image prompt for one shot,
    covering every required element: character appearance, clothing,
    weapons (all three via `_character_clause`), environment, action,
    camera angle, camera movement, composition, lighting, mood,
    cinematic style, and quality modifiers.
    """
    angle_phrase, movement_phrase, composition_phrase = _camera_clause(shot)

    clauses = [
        _character_clause(shot),
        shot.description,
        _environment_clause(shot),
        angle_phrase,
        movement_phrase,
        composition_phrase,
        _lighting_clause(shot),
        f"{shot.mood} atmosphere",
        shot.scene.cinematic_art_style,
        _QUALITY_MODIFIERS,
    ]
    return ", ".join(clause.strip().rstrip(",") for clause in clauses if clause.strip())


def generate_prompts(composed: ComposedTimeline) -> PromptedTimeline:
    """Generates a cinematic image-generation prompt for every shot in
    a ComposedTimeline.

    Args:
        composed: The Scene Composer's output.

    Returns:
        A PromptedTimeline whose every shot carries its own
        `image_prompt`, in addition to everything it already had.
    """
    prompted_shots = [
        PromptedShot(**shot.model_dump(), image_prompt=_build_prompt(shot)) for shot in composed.shots
    ]
    return PromptedTimeline(
        shots=prompted_shots,
        total_duration_seconds=composed.total_duration_seconds,
        shot_count=composed.shot_count,
        scene_continuity=composed.scene_continuity,
    )
