"""FightOrchestrator: the single entry point tying analysis and
narrative generation together. This is the class api/chess2fight.py
calls — it has no HTTP or FastAPI awareness of its own, so it's
trivially reusable (a CLI script, a batch job, a test) without dragging
in the web layer.

v1.1: added the Combat Mapper step. v1.2: added the Battle Director
step. v1.3: added the Style Engine step. v1.4: reordered the pipeline
so Combat Mapper / Battle Director / Style Engine run before narrative
generation. v1.5: added the Battle Mode Interpreter, and
`generate_fight` now takes a `BattlePreferences` object instead of a
bare `style` string — the API layer (api/chess2fight.py) is
responsible for building one from either an explicit `preferences`
object or the legacy top-level `style` field, so this class only ever
deals with one already-resolved preferences object regardless of which
request shape the client used.

    PGN -> Metadata Normalizer -> Analysis -> Combat Mapper ->
    Battle Director -> Style Engine -> Battle Mode Interpreter ->
    Narrative Generator

Style Engine and the Battle Mode Interpreter both run off Combat
Intelligence + Battle Intelligence independently of each other (neither
imports the other) — see battle_mode_engine.py's module docstring for
why that independence is the point, not an oversight."""

from __future__ import annotations

import logging
import re

from core.ai_router import AIProvider
from products.chess2fight.battle_director import generate_battle_intelligence
from products.chess2fight.battle_mode_engine import generate_battle_mode_intelligence
from products.chess2fight.combat_mapper import generate_combat_intelligence
from products.chess2fight.narrative_generator import NarrativeGenerator
from products.chess2fight.pgn_analyzer import analyze_game
from products.chess2fight.schemas import BattlePreferences, GenerateResponse, VideoPlaceholder
from products.chess2fight.style_engine import generate_style_profile
from products.cinema.cinematic_engine import CinematicEngine

logger = logging.getLogger(__name__)


class FightOrchestrator:
    """
    Flow: receive PGN -> analyze chess game (metadata-normalized) ->
    map combat events -> direct the battle arc -> select a style
    profile -> interpret the battle mode -> generate the cinematic
    battle screenplay -> return structured result.
    """

    def __init__(self, ai_provider: AIProvider):
        self._narrative_generator = NarrativeGenerator(ai_provider)
        self._cinematic_engine = CinematicEngine()

    async def generate_fight(self, pgn: str, preferences: BattlePreferences) -> GenerateResponse:
        analysis = analyze_game(pgn)  # raises InvalidPGNError — let the API layer map it to 400
        logger.info(
            "Analyzed game: %s vs %s, %d moves, winner=%s",
            analysis.white_player,
            analysis.black_player,
            analysis.num_moves,
            analysis.winner,
        )

        combat_intelligence = generate_combat_intelligence(analysis)
        battle_intelligence = generate_battle_intelligence(analysis, combat_intelligence)
        style_profile = generate_style_profile(battle_intelligence, combat_intelligence, preferences.style)
        cinematic_sequence = self._cinematic_engine.generate(combat_intelligence,style_profile,)
        battle_mode_intelligence = generate_battle_mode_intelligence(
            preferences.battle_mode, combat_intelligence, battle_intelligence
        )
        
        logger.info(
            "Mapped %d combat events (pace=%s, balance=%s); battle_arc=%s, combat_style=%s; "
            "style=%s; battle_mode=%s (scale=%s)",
            len(combat_intelligence.events),
            combat_intelligence.profile.battle_pace,
            combat_intelligence.profile.fighter_balance,
            battle_intelligence.battle_arc.value,
            battle_intelligence.combat_style.value,
            style_profile.style.value,
            battle_mode_intelligence.mode.value,
            battle_mode_intelligence.scale,
        )

        fight_story = await self._narrative_generator.generate(
            analysis, combat_intelligence, battle_intelligence, style_profile, battle_mode_intelligence
        )

        video_placeholder = VideoPlaceholder(
            status="not_generated",
            message="Video rendering is not implemented yet — this is analysis + narrative only.",
            estimated_duration_seconds=_parse_seconds(fight_story.estimated_length),
        )

        return GenerateResponse(
            status="completed",
            game_analysis=analysis,
            fight_story=fight_story,
            video_placeholder=video_placeholder,
            game_metadata=analysis.metadata,
            combat_intelligence=combat_intelligence,
            battle_intelligence=battle_intelligence,
            style_profile=style_profile,
            battle_mode_intelligence=battle_mode_intelligence,
            cinematic_sequence=cinematic_sequence,
        )


def _parse_seconds(estimated_length: str) -> int:
    """Extracts a representative second count from either a single
    value ("12 sec") or a range ("25-40 sec", the format
    narrative_generator now returns) — a range averages to its
    midpoint. v1.4 fix: the previous digit-concatenation approach
    ("25-40 sec" -> "2540" -> 2540) would have silently produced a
    nonsensical estimated_duration_seconds once estimated_length
    became a range; this is the smallest change that avoids that
    without touching VideoPlaceholder's schema."""
    numbers = [int(n) for n in re.findall(r"\d+", estimated_length)]
    if not numbers:
        return 15
    return round(sum(numbers) / len(numbers))
