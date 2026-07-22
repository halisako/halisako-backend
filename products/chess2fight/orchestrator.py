"""FightOrchestrator: the single entry point tying analysis and
narrative generation together. This is the class api/chess2fight.py
calls — it has no HTTP or FastAPI awareness of its own, so it's
trivially reusable (a CLI script, a batch job, a test) without dragging
in the web layer."""

from __future__ import annotations

import logging

from core.ai_router import AIProvider
from products.chess2fight.narrative_generator import NarrativeGenerator
from products.chess2fight.pgn_analyzer import analyze_game
from products.chess2fight.schemas import GenerateResponse, VideoPlaceholder

logger = logging.getLogger(__name__)


class FightOrchestrator:
    """
    Flow: receive PGN -> analyze chess game -> identify important
    moments -> generate cinematic battle narrative -> return structured
    result.
    """

    def __init__(self, ai_provider: AIProvider):
        self._narrative_generator = NarrativeGenerator(ai_provider)

    async def generate_fight(self, pgn: str, style: str) -> GenerateResponse:
        analysis = analyze_game(pgn)  # raises InvalidPGNError — let the API layer map it to 400
        logger.info(
            "Analyzed game: %s vs %s, %d moves, winner=%s",
            analysis.white_player,
            analysis.black_player,
            analysis.num_moves,
            analysis.winner,
        )

        fight_story = await self._narrative_generator.generate(analysis, style)

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
        )


def _parse_seconds(estimated_length: str) -> int:
    digits = "".join(ch for ch in estimated_length if ch.isdigit())
    return int(digits) if digits else 15
