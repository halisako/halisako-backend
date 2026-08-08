"""Cinematic Timeline Engine: BattleIntelligence + FightStory -> ShotTimeline.

Deterministic, pure logic — no LLM calls, no image generation, no
randomness. Given the same BattleIntelligence and FightStory, this
always produces the same ShotTimeline.

Deliberately consumes only what the brief specifies (BattleIntelligence
and FightStory), not CombatIntelligence or GameAnalysis directly, even
though those exist earlier in the pipeline. Move-level detail this
engine needs (which moves happened, in what order, described how)
comes from parsing FightStory.prompt's own structured screenplay
sections (ENVIRONMENT, LIGHTING, COMBAT CHOREOGRAPHY) — sections
narrative_generator.py already writes in a fixed, parseable format —
rather than reaching for an object outside the stated input contract.
This keeps the Timeline Engine's dependency surface exactly as narrow
as specified, at the cost of being coupled to FightStory.prompt's
section format; see `_extract_section`'s docstring for what happens if
that format ever changes without this module changing too.
"""

from __future__ import annotations

import re

from products.chess2fight.cinematic.schemas import (
    CameraAngle,
    CameraMotion,
    Shot,
    ShotFocus,
    ShotTimeline,
    ShotType,
)
from products.chess2fight.schemas import BattleIntelligence, CombatStyle, FightStory

# Maximum number of EXCHANGE/BUILD_UP shots drawn from choreography
# lines, on top of the four fixed shots (establishing, turning point,
# climax, aftermath) — keeps the timeline's pacing sensible regardless
# of how long the underlying game was.
_MAX_MIDDLE_SHOTS = 4

# Proportion of total_duration_seconds each fixed shot gets. The
# remaining proportion splits evenly across however many middle shots
# exist. Chosen so the emotional peaks (turning point, climax) run
# longer than the bookends, and always sums to a fixed total below 1.0
# regardless of style — the remainder always goes to middle shots.
_ESTABLISHING_SHARE = 0.10
_TURNING_POINT_SHARE = 0.16
_CLIMAX_SHARE = 0.18
_AFTERMATH_SHARE = 0.10


def _extract_section(prompt: str, header: str) -> str:
    """Extracts one named section's body from a FightStory.prompt
    screenplay string.

    narrative_generator.py's `_build_prompt` always separates sections
    with a blank line and puts each section's header (e.g.
    "ENVIRONMENT", "COMBAT CHOREOGRAPHY") on its own line immediately
    before the section's content — this reverses that exact format.
    If that format ever changes, this function degrades gracefully to
    an empty string rather than raising, and every caller here already
    has a documented fallback for a missing section — but the intent
    (dramatizing what the screenplay already describes) would need
    re-checking if narrative_generator.py's section format changes.
    """
    chunks = prompt.split("\n\n")
    for chunk in chunks:
        lines = chunk.split("\n", 1)
        if lines[0].strip() == header:
            return lines[1].strip() if len(lines) > 1 else ""
    return ""


def _parse_choreography(prompt: str) -> list[tuple[str, str]]:
    """Parses the COMBAT CHOREOGRAPHY section into an ordered list of
    (move_label, description) pairs, in the same order the moves were
    played.

    Each line has the form "- {move_label}: {description}." — this
    splits each line on the *first* ": " only, since a description may
    itself contain other colons/dashes.
    """
    section = _extract_section(prompt, "COMBAT CHOREOGRAPHY")
    pairs: list[tuple[str, str]] = []
    for line in section.split("\n"):
        line = line.strip().lstrip("- ").strip()
        if not line or ": " not in line:
            continue
        move_label, _, description = line.partition(": ")
        pairs.append((move_label.strip(), description.strip().rstrip(".")))
    return pairs


def _move_side(move_label: str) -> ShotFocus:
    """Determines which side played a move from its label's format —
    "N. san" for White, "N...san" for Black (pgn_analyzer.py's
    move_label convention)."""
    return ShotFocus.BLACK if "..." in move_label else ShotFocus.WHITE


def _parse_estimated_length_seconds(estimated_length: str) -> float:
    """Parses a "10-15 sec" style range into its midpoint, in seconds.
    Falls back to 30.0 if nothing parseable is found."""
    numbers = [int(n) for n in re.findall(r"\d+", estimated_length)]
    if not numbers:
        return 30.0
    return sum(numbers) / len(numbers)


_CAMERA_BY_SHOT_TYPE: dict[ShotType, tuple[CameraAngle, CameraMotion]] = {
    ShotType.ESTABLISHING: (CameraAngle.WIDE, CameraMotion.PUSH_IN),
    ShotType.BUILD_UP: (CameraAngle.MEDIUM, CameraMotion.TRACK),
    ShotType.EXCHANGE: (CameraAngle.CLOSE_UP, CameraMotion.PAN),
    ShotType.TURNING_POINT: (CameraAngle.CLOSE_UP, CameraMotion.PUSH_IN),
    ShotType.CLIMAX: (CameraAngle.EXTREME_CLOSE_UP, CameraMotion.SHAKE),
    ShotType.AFTERMATH: (CameraAngle.WIDE, CameraMotion.PULL_OUT),
}

# Combat-style-driven overrides for the shot types whose energy should
# visibly track how the battle was actually fought, not just its role
# in the arc — e.g. an EXCHANGE shot in an OVERWHELMING battle reads
# differently than one in a PATIENT battle.
_DYNAMIC_STYLES = {CombatStyle.AGGRESSIVE, CombatStyle.OVERWHELMING, CombatStyle.CHAOTIC}
_MEASURED_STYLES = {CombatStyle.DEFENSIVE, CombatStyle.PATIENT, CombatStyle.DESPERATE}


def _camera_for(shot_type: ShotType, combat_style: CombatStyle) -> tuple[CameraAngle, CameraMotion]:
    angle, motion = _CAMERA_BY_SHOT_TYPE[shot_type]
    if shot_type in (ShotType.EXCHANGE, ShotType.CLIMAX):
        if combat_style in _DYNAMIC_STYLES:
            motion = CameraMotion.SHAKE if shot_type == ShotType.CLIMAX else CameraMotion.CRANE
        elif combat_style in _MEASURED_STYLES:
            motion = CameraMotion.STATIC if shot_type == ShotType.EXCHANGE else CameraMotion.PUSH_IN
    return angle, motion


_MOOD_BY_SHOT_TYPE: dict[ShotType, str] = {
    ShotType.ESTABLISHING: "anticipatory calm",
    ShotType.BUILD_UP: "rising tension",
    ShotType.EXCHANGE: "fierce exchange",
    ShotType.TURNING_POINT: "decisive shift",
    ShotType.CLIMAX: "explosive climax",
    ShotType.AFTERMATH: "settling calm",
}
_MOOD_STYLE_QUALIFIER: dict[CombatStyle, str] = {
    CombatStyle.AGGRESSIVE: "aggressive", CombatStyle.DEFENSIVE: "guarded",
    CombatStyle.BALANCED: "even", CombatStyle.CALCULATED: "controlled",
    CombatStyle.CHAOTIC: "chaotic", CombatStyle.PATIENT: "patient",
    CombatStyle.DESPERATE: "desperate", CombatStyle.OVERWHELMING: "overwhelming",
}


def _mood_for(shot_type: ShotType, combat_style: CombatStyle) -> str:
    base = _MOOD_BY_SHOT_TYPE[shot_type]
    if shot_type in (ShotType.ESTABLISHING, ShotType.AFTERMATH):
        return base
    qualifier = _MOOD_STYLE_QUALIFIER.get(combat_style, "even")
    return f"{qualifier} {base}"


def _select_middle_choreography(
    choreography: list[tuple[str, str]], exclude_labels: set[str]
) -> list[tuple[str, str]]:
    """Picks up to `_MAX_MIDDLE_SHOTS` choreography entries (excluding
    any already used for the turning-point/climax shots) to become
    their own EXCHANGE/BUILD_UP shots.

    If there are more candidates than slots, they're evenly sampled
    across the full list (not just the first N) so a long battle's
    middle shots still span its whole arc rather than clustering at
    the start.
    """
    candidates = [pair for pair in choreography if pair[0] not in exclude_labels]
    if len(candidates) <= _MAX_MIDDLE_SHOTS:
        return candidates
    step = len(candidates) / _MAX_MIDDLE_SHOTS
    indices = sorted({int(i * step) for i in range(_MAX_MIDDLE_SHOTS)})
    return [candidates[i] for i in indices]


_MOVE_LABEL_PATTERN = re.compile(r"^\d+(\.\s|\.\.\.)")
_MOVE_NUMBER_PATTERN = re.compile(r"^(\d+)")


def _extract_move_label(text_with_prefix: str) -> str:
    """Extracts a leading "N. san" / "N...san" move label from a
    FightStory field like `best_move` or `turning_point`, which are
    formatted as "{move_label} — {explanation}" when a real move
    exists.

    Returns "" when there is no real move to extract — critically,
    including narrative_generator.py's own "no standout tactic" /
    "no clear turning point" fallback sentences, which *also* contain
    " — " as ordinary prose punctuation and would otherwise be
    misread as a move label if this only checked for that separator.
    """
    if " — " not in text_with_prefix:
        return ""
    candidate = text_with_prefix.split(" — ")[0].strip()
    return candidate if _MOVE_LABEL_PATTERN.match(candidate) else ""


def _move_number_for_sort(move_label: str) -> int:
    """Extracts the leading move number from a label like "7. Nd5#" or
    "3...Bg4", for chronological sorting. Returns a large sentinel if
    unparseable, so an unparseable label sorts last rather than
    crashing."""
    match = _MOVE_NUMBER_PATTERN.match(move_label)
    return int(match.group(1)) if match else 10_000


def generate_shot_timeline(battle: BattleIntelligence, fight_story: FightStory) -> ShotTimeline:
    """Builds a deterministic Shot Timeline from BattleIntelligence and
    FightStory.

    Args:
        battle: The battle's arc/style/personality classification.
        fight_story: The generated screenplay — source of the parsed
            environment, lighting, and combat choreography this
            timeline dramatizes, plus the facts (winner, opening,
            best_move, turning_point) driving specific shots.

    Returns:
        A ShotTimeline whose shots always sum their durations to the
        same total for the same inputs. Every shot between the fixed
        establishing (first) and aftermath (last) shots is ordered by
        the actual move it dramatizes, chronologically — best_move and
        turning_point are not assumed to occur in any fixed order
        relative to each other, since either can be the earlier event
        depending on the game.
    """
    environment = _extract_section(fight_story.prompt, "ENVIRONMENT") or "an unspecified arena"
    lighting = _extract_section(fight_story.prompt, "LIGHTING") or "even, neutral lighting"
    choreography = _parse_choreography(fight_story.prompt)
    total_duration = _parse_estimated_length_seconds(fight_story.estimated_length)

    turning_point_label = _extract_move_label(fight_story.turning_point)
    best_move_label = _extract_move_label(fight_story.best_move)

    fixed_labels = {label for label in (turning_point_label, best_move_label) if label}
    middle_choreography = _select_middle_choreography(choreography, exclude_labels=fixed_labels)

    # Build every middle-shot candidate — plain choreography beats plus
    # the two "special" ones — as (move_number, tie_break, shot_type,
    # source_moves, description), then sort chronologically. tie_break
    # keeps ordering stable when two entries share a move number (e.g.
    # a capture and the checkmate on the same move): plain choreography
    # first, then turning point, then climax.
    middle: list[tuple[int, int, ShotType, list[str], str]] = []
    for label, description in middle_choreography:
        shot_type = ShotType.BUILD_UP if len(middle) < _MAX_MIDDLE_SHOTS // 2 else ShotType.EXCHANGE
        middle.append((_move_number_for_sort(label), 0, shot_type, [label], f"{description}."))
    if turning_point_label:
        turning_text = (
            fight_story.turning_point.split(" — ", 1)[1]
            if " — " in fight_story.turning_point else fight_story.turning_point
        )
        middle.append((_move_number_for_sort(turning_point_label), 1, ShotType.TURNING_POINT, [turning_point_label], turning_text))
    if best_move_label:
        climax_text = (
            fight_story.best_move.split(" — ", 1)[1] if " — " in fight_story.best_move else fight_story.best_move
        )
        middle.append((_move_number_for_sort(best_move_label), 2, ShotType.CLIMAX, [best_move_label], climax_text))
    middle.sort(key=lambda entry: (entry[0], entry[1]))

    plan: list[tuple[ShotType, list[str], str]] = [
        (ShotType.ESTABLISHING, [], f"Setting the scene: {fight_story.opening}, in {environment}.")
    ]
    plan.extend((shot_type, source_moves, description) for _, _, shot_type, source_moves, description in middle)
    aftermath_moves = [best_move_label] if best_move_label else []
    plan.append(
        (ShotType.AFTERMATH, aftermath_moves, f"{fight_story.winner}. {fight_story.battle_summary}")
    )

    middle_count = sum(1 for shot_type, _, _ in plan if shot_type in (ShotType.BUILD_UP, ShotType.EXCHANGE))
    fixed_share = _ESTABLISHING_SHARE + _AFTERMATH_SHARE
    fixed_share += _TURNING_POINT_SHARE if turning_point_label else 0.0
    fixed_share += _CLIMAX_SHARE if best_move_label else 0.0
    middle_share_each = (1.0 - fixed_share) / middle_count if middle_count else 0.0

    shots: list[Shot] = []
    for order, (shot_type, source_moves, description) in enumerate(plan, start=1):
        angle, motion = _camera_for(shot_type, battle.combat_style)
        if shot_type == ShotType.ESTABLISHING:
            share, focus = _ESTABLISHING_SHARE, ShotFocus.ENVIRONMENT
        elif shot_type == ShotType.AFTERMATH:
            share, focus = _AFTERMATH_SHARE, ShotFocus.BOTH
        elif shot_type == ShotType.TURNING_POINT:
            share, focus = _TURNING_POINT_SHARE, _move_side(source_moves[0])
        elif shot_type == ShotType.CLIMAX:
            share, focus = _CLIMAX_SHARE, _move_side(source_moves[0])
        else:
            share, focus = middle_share_each, _move_side(source_moves[0])

        shots.append(
            Shot(
                shot_id=f"shot_{order}",
                sequence_order=order,
                shot_type=shot_type,
                camera_angle=angle,
                camera_motion=motion,
                focus=focus,
                duration_seconds=round(total_duration * share, 2),
                environment=environment,
                lighting=lighting,
                mood=_mood_for(shot_type, battle.combat_style),
                source_moves=source_moves,
                description=description,
            )
        )

    return ShotTimeline(
        shots=shots,
        total_duration_seconds=round(sum(shot.duration_seconds for shot in shots), 2),
        shot_count=len(shots),
    )
