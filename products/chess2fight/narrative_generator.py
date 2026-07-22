"""
Turns a GameAnalysis into a FightStory.

Deliberate split: winner / opening / best_move / turning_point /
estimated_length are FACTS, computed straight from GameAnalysis — never
handed to an AI provider to "decide," since an LLM has no business
overriding what the game notation already proves. Only fight_style,
battle_summary, and the cinematic `prompt` text are creative, and those
go through AIProvider.generate_text(). If the provider fails for any
reason, we fall back to TemplateProvider rather than failing the whole
request — the deterministic fields are correct either way.
"""

from __future__ import annotations

import logging
import re

from core.ai_router import AIProvider, TemplateProvider
from core.exceptions import AIProviderError
from products.chess2fight.schemas import FightStory, GameAnalysis

logger = logging.getLogger(__name__)

_RESPONSE_RE = re.compile(
    r"FIGHT_STYLE:\s*(?P<style>.+?)\s*"
    r"BATTLE_SUMMARY:\s*(?P<summary>.+?)\s*"
    r"SCENE_PROMPT:\s*(?P<prompt>.+)",
    re.DOTALL,
)


def _describe_winner(analysis: GameAnalysis) -> str:
    if analysis.winner == "draw":
        return "Draw"
    if analysis.winner == "unknown":
        return "Result unresolved"
    label = "White" if analysis.winner == "white" else "Black"
    return f"{label} wins by checkmate" if analysis.is_checkmate else f"{label} wins by resignation"


def _describe_best_move(analysis: GameAnalysis) -> str:
    # Prefer the most materially significant tactical moment; fall back
    # to the checkmating move; fall back to "no standout tactic."
    candidates = [m for m in analysis.tactical_moments if "Captures" in m.description]
    moment = candidates[-1] if candidates else (
        analysis.tactical_moments[-1] if analysis.tactical_moments else None
    )
    if moment is None:
        return "No standout tactic — a quiet, positional game."
    return f"{moment.move_label} — {moment.description.lower()}"


def _describe_turning_point(analysis: GameAnalysis) -> str:
    if not analysis.turning_points:
        return "No clear turning point — the game stayed balanced throughout."
    # Prefer the earliest turning point as the narrative pivot; the
    # checkmate itself is the climax, not the turn.
    point = analysis.turning_points[0]
    return f"{point.move_label} — {point.description}"


def _estimate_length(analysis: GameAnalysis) -> str:
    seconds = 8 + analysis.num_moves * 0.6 + len(analysis.tactical_moments) * 1.5
    seconds = max(10, min(30, round(seconds)))
    return f"{seconds} sec"


def _build_prompt(analysis: GameAnalysis, style: str) -> str:
    """Builds the text sent to AIProvider.generate_text(). The
    'ANALYSIS FACTS' block is deliberately machine-parseable — see
    TemplateProvider, which regexes these same lines back out."""
    tactical_lines = "\n".join(f"- {m.move_label}: {m.description}" for m in analysis.tactical_moments) or "- none"

    return (
        "You are writing a short cinematic fight-scene brief based on a real "
        "chess game. Use the facts below — do not invent moves or outcomes.\n\n"
        "ANALYSIS FACTS\n"
        f"White player: {analysis.white_player}\n"
        f"Black player: {analysis.black_player}\n"
        f"Opening: {analysis.opening}\n"
        f"Total moves: {analysis.num_moves}\n"
        f"Captures: {len(analysis.captures)}\n"
        f"Ends in checkmate: {'yes' if analysis.is_checkmate else 'no'}\n"
        f"Winner: {'White' if analysis.winner == 'white' else 'Black' if analysis.winner == 'black' else 'Draw'}\n"
        f"Requested style: {style}\n"
        "Tactical moments:\n"
        f"{tactical_lines}\n\n"
        "Respond in exactly this format and nothing else:\n"
        "FIGHT_STYLE: <a short 2-4 word evocative label for how this game 'felt'>\n"
        "BATTLE_SUMMARY: <2-3 cinematic sentences dramatizing the game, staying "
        "true to the facts above>\n"
        "SCENE_PROMPT: <a short scene-direction block: setting, fighters, a few "
        "beats, and the finish>"
    )


class NarrativeGenerator:
    def __init__(self, ai_provider: AIProvider):
        self._provider = ai_provider
        self._fallback = TemplateProvider()

    async def generate(self, analysis: GameAnalysis, style: str) -> FightStory:
        prompt = _build_prompt(analysis, style)

        raw = await self._generate_with_fallback(prompt)
        fight_style, battle_summary, scene_prompt = self._parse(raw, analysis)

        return FightStory(
            winner=_describe_winner(analysis),
            opening=analysis.opening,
            fight_style=fight_style,
            best_move=_describe_best_move(analysis),
            turning_point=_describe_turning_point(analysis),
            battle_summary=battle_summary,
            prompt=scene_prompt,
            estimated_length=_estimate_length(analysis),
        )

    async def _generate_with_fallback(self, prompt: str) -> str:
        try:
            return await self._provider.generate_text(prompt)
        except AIProviderError as exc:
            logger.warning("AI provider failed (%s) — using template fallback.", exc)
            return await self._fallback.generate_text(prompt)

    @staticmethod
    def _parse(raw: str, analysis: GameAnalysis) -> tuple[str, str, str]:
        match = _RESPONSE_RE.search(raw)
        if not match:
            # The model didn't follow the format — degrade gracefully
            # rather than 500ing: use the raw text as the summary and
            # fill the rest from the template provider's own format.
            logger.warning("Could not parse AI response into sections; using raw text as summary.")
            return "Unclassified", raw.strip()[:400], "Scene details unavailable."

        return (
            match.group("style").strip(),
            match.group("summary").strip(),
            match.group("prompt").strip(),
        )
