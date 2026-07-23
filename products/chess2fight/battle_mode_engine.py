"""
Battle Mode Interpreter.

Converts BattleMode + CombatIntelligence + BattleIntelligence into
battle presentation metadata (scale, unit mapping, environment, combat
focus). Presentation only, like style_engine.py — never touches chess
facts, never generates narrative prose (that stays
narrative_generator's job).

Rule 2 from this task's brief, made structural rather than just
documented: battle mode is NOT a visual style. This module never
imports StyleProfile or anything from style_engine.py — it only
consumes CombatIntelligence and BattleIntelligence, the same two
inputs style_engine.py consumes, so the two presentation dimensions
(mode and genre) are computed completely independently and combined
only in narrative_generator.py. Duel + Anime and Army + Fantasy are
both valid combinations precisely because neither module knows the
other exists.

Pure, deterministic, no AI provider, no randomness — same discipline
as combat_mapper.py, battle_director.py, and style_engine.py.
"""

from __future__ import annotations

from products.chess2fight.schemas import (
    BattleArc,
    BattleIntelligence,
    BattleMode,
    BattleModeIntelligence,
    CombatIntelligence,
    CombatStyle,
)

# Test 3's exact required mapping.
_ARMY_UNIT_MAPPING: dict[str, str] = {
    "pawn": "infantry",
    "knight": "cavalry",
    "bishop": "mage",
    "rook": "siege",
    "queen": "commander",
    "king": "fortress",
}

# A duel-appropriate equivalent: pieces read as a single fighter's
# techniques/stances rather than army units.
_DUEL_UNIT_MAPPING: dict[str, str] = {
    "pawn": "basic strike",
    "knight": "flanking strike",
    "bishop": "ranged technique",
    "rook": "defensive guard",
    "queen": "signature technique",
    "king": "final stand",
}

_ARMY_FOCUS_BY_STYLE: dict[CombatStyle, list[str]] = {
    CombatStyle.AGGRESSIVE: ["cavalry charges", "frontal assault", "overwhelming numbers"],
    CombatStyle.DEFENSIVE: ["fortified positions", "siege defense", "attrition warfare"],
    CombatStyle.BALANCED: ["formations", "territorial conquest", "mass combat"],
    CombatStyle.CALCULATED: ["coordinated maneuvers", "territorial control", "decisive flanking"],
    CombatStyle.CHAOTIC: ["broken formations", "close-quarters melee", "shifting front lines"],
    CombatStyle.PATIENT: ["siege warfare", "supply lines", "gradual encirclement"],
    CombatStyle.DESPERATE: ["last stand", "desperate reinforcement", "collapsing formation"],
    CombatStyle.OVERWHELMING: ["total assault", "rapid conquest", "overwhelming force"],
}

_DUEL_FOCUS_BY_STYLE: dict[CombatStyle, list[str]] = {
    CombatStyle.AGGRESSIVE: ["relentless offense", "overwhelming pressure", "decisive strikes"],
    CombatStyle.DEFENSIVE: ["patient defense", "counter-attacks", "calculated openings"],
    CombatStyle.BALANCED: ["personal skill", "even exchanges", "one-on-one mastery"],
    CombatStyle.CALCULATED: ["precise technique", "measured exchanges", "tactical mastery"],
    CombatStyle.CHAOTIC: ["unpredictable exchanges", "improvised counters", "scrambling defense"],
    CombatStyle.PATIENT: ["patient positioning", "drawn-out exchanges", "the long game"],
    CombatStyle.DESPERATE: ["a desperate stand", "last-resort counters", "fighting from behind"],
    CombatStyle.OVERWHELMING: ["total domination", "one-sided mastery", "an overwhelming finish"],
}

_ARMY_ENVIRONMENT_BY_ARC: dict[BattleArc, str] = {
    BattleArc.BLITZ_EXECUTION: "a rapidly shifting front line",
    BattleArc.TACTICAL_AMBUSH: "a concealed battlefield flank",
    BattleArc.WAR_OF_ATTRITION: "a sprawling, contested battlefield",
    BattleArc.COMEBACK: "a battlefield reclaimed under pressure",
    BattleArc.SIEGE: "a besieged fortress perimeter",
    BattleArc.GAMBIT_ASSAULT: "an exposed forward position",
    BattleArc.FINAL_DUEL: "the battlefield's final stand",
}
_DUEL_ENVIRONMENT_BY_ARC: dict[BattleArc, str] = {
    BattleArc.BLITZ_EXECUTION: "a lightning-fast dueling arena",
    BattleArc.TACTICAL_AMBUSH: "a shadowed ambush ground",
    BattleArc.WAR_OF_ATTRITION: "a grueling dueling arena",
    BattleArc.COMEBACK: "an arena reclaimed under pressure",
    BattleArc.SIEGE: "a fortified one-on-one arena",
    BattleArc.GAMBIT_ASSAULT: "an exposed, high-risk arena",
    BattleArc.FINAL_DUEL: "a fateful final arena",
}


def _army_scale(combat: CombatIntelligence) -> str:
    """Army mode's "scale" reflects battlefield size — driven by
    engagement volume (combat event count), since that's the closest
    proxy this pipeline has to "how big did this fight get."""
    count = len(combat.events)
    if count <= 8:
        return "skirmish"
    if count <= 20:
        return "battle"
    return "large-scale war"


def _duel_scale(battle: BattleIntelligence) -> str:
    """Duel mode's "scale" reflects intensity rather than size — a
    1v1 duel doesn't have a battlefield footprint to measure."""
    if battle.combat_style in (CombatStyle.OVERWHELMING, CombatStyle.AGGRESSIVE):
        return "explosive duel"
    if battle.combat_style in (CombatStyle.DEFENSIVE, CombatStyle.PATIENT, CombatStyle.DESPERATE):
        return "measured duel"
    return "intense duel"


def generate_battle_mode_intelligence(
    mode: BattleMode, combat: CombatIntelligence, battle: BattleIntelligence
) -> BattleModeIntelligence:
    """Single public entry point: BattleMode + CombatIntelligence +
    BattleIntelligence in, BattleModeIntelligence out. Pure function —
    no AI provider, no I/O, deterministic."""
    if mode == BattleMode.ARMY:
        return BattleModeIntelligence(
            mode=mode,
            scale=_army_scale(combat),
            unit_mapping=dict(_ARMY_UNIT_MAPPING),
            combat_focus=_ARMY_FOCUS_BY_STYLE.get(battle.combat_style, ["formations", "mass combat"]),
            environment=_ARMY_ENVIRONMENT_BY_ARC.get(battle.battle_arc, "an open battlefield"),
        )

    return BattleModeIntelligence(
        mode=mode,
        scale=_duel_scale(battle),
        unit_mapping=dict(_DUEL_UNIT_MAPPING),
        combat_focus=_DUEL_FOCUS_BY_STYLE.get(battle.combat_style, ["personal skill", "decisive strikes"]),
        environment=_DUEL_ENVIRONMENT_BY_ARC.get(battle.battle_arc, "a dueling arena"),
    )
