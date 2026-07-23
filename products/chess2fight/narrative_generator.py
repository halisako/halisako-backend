"""
Battle Screenplay Generator (formerly a thinner "Narrative Generator").

v1.4: this module no longer thinks like a chess commentator restating
GameAnalysis — it thinks like a director working from a completed
intelligence brief. It now consumes GameAnalysis, CombatIntelligence,
BattleIntelligence, and StyleProfile (all already computed upstream)
to build every fight_story field, instead of deriving everything from
GameAnalysis alone. GameAnalysis remains available as supporting fact
— the deterministic "what actually happened" ground truth every other
field is checked against — but the wording now comes from the richer
upstream intelligence.

v1.5: also consumes BattleModeIntelligence (duel vs. army). This is a
presentation dimension completely independent of StyleProfile's genre
dimension — see battle_mode_engine.py's module docstring — so the two
are combined here for the first time, in the CAMERA framing, the
FIGHTERS/FORCES section, and combat choreography's subject noun (a
style-genre weapon for duel, a battle_mode unit for army).

DETERMINISM: this module no longer calls an AI provider. A live LLM
call cannot guarantee bit-identical output between calls (even at low
temperature), which would directly break "same PGN + same style ->
identical narrative, always." Every field below is built by pure,
deterministic template selection over already-computed, deterministic
inputs — no random numbers, no timestamps, no UUIDs. `AIProvider` is
still accepted by the constructor (unused) purely so
`FightOrchestrator`'s existing construction of this class doesn't need
to change — see the engineering review for why the AIProvider
abstraction itself (OpenAI/Anthropic/Gemini/Local) is being left
in place but dormant, rather than removed.

Fields improved in this revision, per the brief — fight_style,
best_move, turning_point, battle_summary, prompt, estimated_length.
`winner` and `opening` are untouched: they were already pure,
well-supported facts, and the brief only asked to improve the other
six.

On hedged language: `best_move` and the screenplay's combat
choreography describe what CombatIntelligence already classified
(e.g. a `calculated_sacrifice` event) — they never assert intent
beyond what that classification already implies, matching the same
caution combat_mapper.py and battle_director.py already document.
"""

from __future__ import annotations

from core.ai_router import AIProvider
from products.chess2fight.schemas import (
    BattleArc,
    BattleIntelligence,
    BattleMode,
    BattleModeIntelligence,
    CombatEventType,
    CombatIntelligence,
    CombatStyle,
    FightStory,
    GameAnalysis,
    StyleId,
    StyleProfile,
)

# ======================================================================
# 1. fight_style — battle arc + combat style + selected visual style,
#    fused into a short, genre-shaped phrase. Personality-led for
#    anime/fantasy (matching how those genres typically name a fighter's
#    signature performance); abstract doctrine-style naming for
#    modern_warfare/scifi/superhero (matching how those genres typically
#    name an operation/engagement rather than a person).
# ======================================================================

_ANIME_COMBAT_VERB: dict[CombatStyle, str] = {
    CombatStyle.AGGRESSIVE: "storms through every exchange",
    CombatStyle.DEFENSIVE: "holds an unbreakable guard",
    CombatStyle.BALANCED: "trades blow for blow",
    CombatStyle.CALCULATED: "dismantles the opponent with precision",
    CombatStyle.CHAOTIC: "turns the duel into chaos",
    CombatStyle.PATIENT: "wears down every defense",
    CombatStyle.DESPERATE: "claws back from the brink",
    CombatStyle.OVERWHELMING: "overwhelms the battlefield",
}

_FANTASY_ARC_PHRASE: dict[BattleArc, str] = {
    BattleArc.BLITZ_EXECUTION: "a lightning execution",
    BattleArc.TACTICAL_AMBUSH: "a cunning ambush",
    BattleArc.WAR_OF_ATTRITION: "a grinding war of attrition",
    BattleArc.COMEBACK: "an against-the-odds uprising",
    BattleArc.SIEGE: "a patient siege",
    BattleArc.GAMBIT_ASSAULT: "a reckless gambit",
    BattleArc.FINAL_DUEL: "a fateful final duel",
}

_MW_ARC_NOUN: dict[BattleArc, str] = {
    BattleArc.BLITZ_EXECUTION: "Rapid Strike",
    BattleArc.TACTICAL_AMBUSH: "Ambush",
    BattleArc.WAR_OF_ATTRITION: "Siege",
    BattleArc.COMEBACK: "Counteroffensive",
    BattleArc.SIEGE: "Blockade",
    BattleArc.GAMBIT_ASSAULT: "Shock",
    BattleArc.FINAL_DUEL: "Standoff",
}
_MW_STYLE_ADJ: dict[CombatStyle, str] = {
    CombatStyle.AGGRESSIVE: "Urban",
    CombatStyle.DEFENSIVE: "Fortified",
    CombatStyle.BALANCED: "Combined-Arms",
    CombatStyle.CALCULATED: "Precision",
    CombatStyle.CHAOTIC: "Close-Quarters",
    CombatStyle.PATIENT: "Sustained",
    CombatStyle.DESPERATE: "Last-Stand",
    CombatStyle.OVERWHELMING: "Overwhelming",
}

_SCIFI_ARC_NOUN: dict[BattleArc, str] = {
    BattleArc.BLITZ_EXECUTION: "Strike",
    BattleArc.TACTICAL_AMBUSH: "Ambush",
    BattleArc.WAR_OF_ATTRITION: "Siege",
    BattleArc.COMEBACK: "Reversal",
    BattleArc.SIEGE: "Blockade",
    BattleArc.GAMBIT_ASSAULT: "Gambit",
    BattleArc.FINAL_DUEL: "Duel",
}
_SCIFI_STYLE_ADJ: dict[CombatStyle, str] = {
    CombatStyle.AGGRESSIVE: "Orbital",
    CombatStyle.DEFENSIVE: "Shielded",
    CombatStyle.BALANCED: "Fleet",
    CombatStyle.CALCULATED: "Precision",
    CombatStyle.CHAOTIC: "Void",
    CombatStyle.PATIENT: "Deep-Space",
    CombatStyle.DESPERATE: "Last-Ditch",
    CombatStyle.OVERWHELMING: "Total",
}

_HERO_ARC_NOUN: dict[BattleArc, str] = {
    BattleArc.BLITZ_EXECUTION: "Takedown",
    BattleArc.TACTICAL_AMBUSH: "Ambush",
    BattleArc.WAR_OF_ATTRITION: "War",
    BattleArc.COMEBACK: "Comeback",
    BattleArc.SIEGE: "Siege",
    BattleArc.GAMBIT_ASSAULT: "Gambit",
    BattleArc.FINAL_DUEL: "Duel of Titans",
}
_HERO_STYLE_ADJ: dict[CombatStyle, str] = {
    CombatStyle.AGGRESSIVE: "Cosmic",
    CombatStyle.DEFENSIVE: "Unbreakable",
    CombatStyle.BALANCED: "Titanic",
    CombatStyle.CALCULATED: "Masterful",
    CombatStyle.CHAOTIC: "Chaotic",
    CombatStyle.PATIENT: "Enduring",
    CombatStyle.DESPERATE: "Last-Stand",
    CombatStyle.OVERWHELMING: "Cataclysmic",
}


def _winner_personality_label(battle: BattleIntelligence, analysis: GameAnalysis) -> str:
    if analysis.winner == "black":
        return battle.fighter_personality.black.label
    return battle.fighter_personality.white.label  # white, draw, or unknown all default to White's


def _fight_style(analysis: GameAnalysis, battle: BattleIntelligence, style_profile: StyleProfile) -> str:
    personality = _winner_personality_label(battle, analysis)

    if style_profile.style == StyleId.ANIME:
        return f"{personality} {_ANIME_COMBAT_VERB[battle.combat_style]}"
    if style_profile.style == StyleId.FANTASY:
        return f"{personality} wages {_FANTASY_ARC_PHRASE[battle.battle_arc]}"
    if style_profile.style == StyleId.MODERN_WARFARE:
        return f"{_MW_STYLE_ADJ[battle.combat_style]} {_MW_ARC_NOUN[battle.battle_arc]} Assault"
    if style_profile.style == StyleId.SCIFI:
        return f"{_SCIFI_STYLE_ADJ[battle.combat_style]} {_SCIFI_ARC_NOUN[battle.battle_arc]} Doctrine"
    return f"{_HERO_STYLE_ADJ[battle.combat_style]} {_HERO_ARC_NOUN[battle.battle_arc]}"  # superhero


# ======================================================================
# 2 & 3. best_move / turning_point — explain WHY a move mattered rather
#    than restating its notation, using whichever CombatEvent matches
#    it (by move_label) to pick a style- and event-type-aware template.
#    The underlying move/fact never changes — only the framing does.
# ======================================================================

_PIECE_NOUN: dict[StyleId, dict[str, str]] = {
    StyleId.ANIME: {
        "queen": "strongest warrior", "rook": "guardian tower", "bishop": "shadow blade",
        "knight": "cavalry fighter", "pawn": "foot soldier", "king": "commander",
    },
    StyleId.FANTASY: {
        "queen": "Queen's guardian", "rook": "fortress tower", "bishop": "court mage",
        "knight": "mounted knight", "pawn": "footman", "king": "sovereign",
    },
    StyleId.MODERN_WARFARE: {
        "queen": "command vehicle", "rook": "armored position", "bishop": "support unit",
        "knight": "strike team", "pawn": "infantry unit", "king": "command post",
    },
    StyleId.SCIFI: {
        "queen": "flagship", "rook": "defense platform", "bishop": "support cruiser",
        "knight": "interceptor", "pawn": "drone unit", "king": "command station",
    },
    StyleId.SUPERHERO: {
        "queen": "strongest defender", "rook": "fortress ally", "bishop": "support hero",
        "knight": "strike-team hero", "pawn": "civilian defender", "king": "team leader",
    },
}

_BEST_MOVE_TEMPLATE: dict[StyleId, dict[CombatEventType, str]] = {
    StyleId.ANIME: {
        CombatEventType.CALCULATED_SACRIFICE: "{side} appears to sacrifice ground, forcing a tactical sequence the opponent can't escape.",
        CombatEventType.BREAKTHROUGH_ATTACK: "{side} disarms the opponent's {piece}, shifting the momentum of the duel.",
        CombatEventType.FINISHING_STRIKE: "{side} lands the ultimate technique, ending the duel outright.",
        CombatEventType.CRITICAL_THREAT: "{side} presses forward with a direct threat the opponent cannot ignore.",
        CombatEventType.ATTACK_LANDED: "{side} lands a clean strike on the opponent's {piece}.",
    },
    StyleId.FANTASY: {
        CombatEventType.CALCULATED_SACRIFICE: "{side} appears to offer up ground, luring the enemy into a forced trap.",
        CombatEventType.BREAKTHROUGH_ATTACK: "{side} destroys the {piece}, leaving the kingdom exposed.",
        CombatEventType.FINISHING_STRIKE: "{side} delivers the final blow, ending the battle for the realm.",
        CombatEventType.CRITICAL_THREAT: "{side} presses the attack, threatening the enemy stronghold directly.",
        CombatEventType.ATTACK_LANDED: "{side} strikes down the enemy's {piece}.",
    },
    StyleId.MODERN_WARFARE: {
        CombatEventType.CALCULATED_SACRIFICE: "{side} appears to accept a costly forward position, setting up a forced counter.",
        CombatEventType.BREAKTHROUGH_ATTACK: "Heavy fire disables the enemy's {piece}, leaving the formation vulnerable.",
        CombatEventType.FINISHING_STRIKE: "{side} executes the decisive strike that ends the engagement.",
        CombatEventType.CRITICAL_THREAT: "{side} closes in with a direct threat the enemy cannot ignore.",
        CombatEventType.ATTACK_LANDED: "A precision hit takes out the enemy's {piece}.",
    },
    StyleId.SCIFI: {
        CombatEventType.CALCULATED_SACRIFICE: "{side} appears to sacrifice position, forcing a compromising tactical sequence.",
        CombatEventType.BREAKTHROUGH_ATTACK: "Heavy plasma fire disables the {piece}, leaving the fleet vulnerable.",
        CombatEventType.FINISHING_STRIKE: "{side} fires the decisive shot that ends the confrontation.",
        CombatEventType.CRITICAL_THREAT: "{side} closes in with a direct threat the enemy cannot evade.",
        CombatEventType.ATTACK_LANDED: "A direct hit disables the enemy's {piece}.",
    },
    StyleId.SUPERHERO: {
        CombatEventType.CALCULATED_SACRIFICE: "{side} appears to leave an opening, baiting the opponent into a forced exchange.",
        CombatEventType.BREAKTHROUGH_ATTACK: "A devastating blast removes the opponent's {piece} from the fight.",
        CombatEventType.FINISHING_STRIKE: "{side} lands the finishing blow that ends the confrontation.",
        CombatEventType.CRITICAL_THREAT: "{side} presses forward with overwhelming, undeniable pressure.",
        CombatEventType.ATTACK_LANDED: "{side} lands a solid hit on the opponent's {piece}.",
    },
}

_DEFAULT_MOVE_TEMPLATE = "{side} finds the key exchange that defines the duel."


def _side_label(color: str) -> str:
    return "White" if color == "white" else "Black"


def _find_event_for_move(combat: CombatIntelligence, move_label: str):
    for event in combat.events:
        if event.move_label == move_label:
            return event
    return None


def _piece_noun(style: StyleId, piece: str | None) -> str:
    if not piece:
        return "position"
    return _PIECE_NOUN.get(style, {}).get(piece, piece)


def _best_move(analysis: GameAnalysis, combat: CombatIntelligence, style_profile: StyleProfile) -> str:
    # Same candidate-selection logic as before: prefer the most
    # materially significant tactical moment, fall back to the
    # checkmating move, fall back to "no standout tactic."
    candidates = [m for m in analysis.tactical_moments if "Captures" in m.description]
    moment = candidates[-1] if candidates else (
        analysis.tactical_moments[-1] if analysis.tactical_moments else None
    )
    if moment is None:
        return "No standout tactic emerges — a quiet, positional battle throughout."

    event = _find_event_for_move(combat, moment.move_label)
    templates = _BEST_MOVE_TEMPLATE.get(style_profile.style, {})
    template = templates.get(event.event_type, _DEFAULT_MOVE_TEMPLATE) if event else _DEFAULT_MOVE_TEMPLATE

    captured = next((c.captured_piece for c in analysis.captures if c.move_label == moment.move_label), None)
    side = _side_label(event.attacker) if event else ""
    piece = _piece_noun(style_profile.style, captured)

    return f"{moment.move_label} — {template.format(side=side, piece=piece)}"


_TURNING_POINT_TEMPLATE: dict[StyleId, dict[CombatEventType, str]] = {
    StyleId.ANIME: {
        CombatEventType.CALCULATED_SACRIFICE: "The forced tactical sequence left one fighter's guard broken beyond repair.",
        CombatEventType.BREAKTHROUGH_ATTACK: "The decisive breakthrough shattered the opponent's defenses.",
        CombatEventType.FINISHING_STRIKE: "The coordinated strike left the opponent with no escape.",
        CombatEventType.CRITICAL_THREAT: "The relentless pressure finally exposed an opening.",
    },
    StyleId.FANTASY: {
        CombatEventType.CALCULATED_SACRIFICE: "The bait was taken, and the trap closed with brutal precision.",
        CombatEventType.BREAKTHROUGH_ATTACK: "The fortress wall was breached, and there was no holding the line after.",
        CombatEventType.FINISHING_STRIKE: "The final assault left no ground left to defend.",
        CombatEventType.CRITICAL_THREAT: "The siege finally found the crack in the kingdom's defenses.",
    },
    StyleId.MODERN_WARFARE: {
        CombatEventType.CALCULATED_SACRIFICE: "The exposed position turned out to be a forced setup, not a mistake.",
        CombatEventType.BREAKTHROUGH_ATTACK: "The failed defense exposed the command position entirely.",
        CombatEventType.FINISHING_STRIKE: "The coordinated assault left the enemy with no route to retreat.",
        CombatEventType.CRITICAL_THREAT: "Sustained pressure finally broke the defensive line.",
    },
    StyleId.SCIFI: {
        CombatEventType.CALCULATED_SACRIFICE: "The apparent opening was a forced sequence, not an error in judgment.",
        CombatEventType.BREAKTHROUGH_ATTACK: "The shield failure exposed the fleet's command ship entirely.",
        CombatEventType.FINISHING_STRIKE: "The coordinated strike left the enemy fleet with no vector to escape.",
        CombatEventType.CRITICAL_THREAT: "Sustained fire finally breached the outer defenses.",
    },
    StyleId.SUPERHERO: {
        CombatEventType.CALCULATED_SACRIFICE: "The opening was a forced gambit, not a lapse in defense.",
        CombatEventType.BREAKTHROUGH_ATTACK: "The failed block shattered the defender's guard completely.",
        CombatEventType.FINISHING_STRIKE: "The coordinated assault left no opening for a counter.",
        CombatEventType.CRITICAL_THREAT: "The relentless assault finally broke through the defense.",
    },
}
_DEFAULT_TURNING_POINT = "The balance of the battle shifted decisively at this moment."


def _turning_point(
    analysis: GameAnalysis, combat: CombatIntelligence, style_profile: StyleProfile
) -> str:
    if not analysis.turning_points:
        return "No clear turning point emerges — the battle stayed balanced throughout."
    point = analysis.turning_points[0]
    event = _find_event_for_move(combat, point.move_label)
    templates = _TURNING_POINT_TEMPLATE.get(style_profile.style, {})
    text = templates.get(event.event_type, _DEFAULT_TURNING_POINT) if event else _DEFAULT_TURNING_POINT
    return f"{point.move_label} — {text}"


# ======================================================================
# 4. battle_summary — a movie-poster-style synopsis weaving together
#    opening, battle arc, combat style, both personalities, winner, and
#    ending type.
# ======================================================================

_ENDING_PHRASE = {
    "checkmate": "a decisive finishing strike",
    "resignation": "a hard-fought concession",
    "draw": "a hard-fought standstill",
    "time_forfeit": "a race against the clock",
    "unknown": "an inconclusive finish",
}


_SUMMARY_CLASH_VERB: dict[StyleId, str] = {
    StyleId.ANIME: "clashed",
    StyleId.FANTASY: "did battle",
    StyleId.MODERN_WARFARE: "engaged",
    StyleId.SCIFI: "engaged in orbit",
    StyleId.SUPERHERO: "collided",
}

_COMBAT_STYLE_ADVERB: dict[CombatStyle, str] = {
    CombatStyle.AGGRESSIVE: "aggressively",
    CombatStyle.DEFENSIVE: "cautiously",
    CombatStyle.BALANCED: "evenly",
    CombatStyle.CALCULATED: "with cold precision",
    CombatStyle.CHAOTIC: "chaotically",
    CombatStyle.PATIENT: "patiently",
    CombatStyle.DESPERATE: "desperately",
    CombatStyle.OVERWHELMING: "overwhelmingly",
}


def _battle_summary(
    analysis: GameAnalysis, battle: BattleIntelligence, style_profile: StyleProfile
) -> str:
    winner_label = _side_label(analysis.winner) if analysis.winner in ("white", "black") else None
    winner_personality = _winner_personality_label(battle, analysis)
    other_personality = (
        battle.fighter_personality.white.label
        if analysis.winner == "black"
        else battle.fighter_personality.black.label
    )
    arc_words = battle.battle_arc.value.replace("_", " ")
    style_adverb = _COMBAT_STYLE_ADVERB.get(battle.combat_style, "evenly")
    ending_phrase = _ENDING_PHRASE.get(_ending_type_key(analysis, style_profile), "an inconclusive finish")
    clash_verb = _SUMMARY_CLASH_VERB.get(style_profile.style, "clashed")

    if winner_label is None:
        return (
            f"Out of the {analysis.opening}, in the {style_profile.environment}, "
            f"{other_personality} and {winner_personality} {clash_verb} {style_adverb} through "
            f"{arc_words} that ended in {ending_phrase}."
        )

    return (
        f"Out of the {analysis.opening}, in the {style_profile.environment}, "
        f"{other_personality} and {winner_personality} {clash_verb} {style_adverb} through "
        f"{arc_words} before {winner_label} closed it out with {ending_phrase}."
    )


def _ending_type_key(analysis: GameAnalysis, style_profile: StyleProfile) -> str:
    if analysis.winner == "draw":
        return "draw"
    return "checkmate" if analysis.is_checkmate else "resignation"


# ======================================================================
# 5. prompt — a structured cinematic screenplay, replacing the old
#    5-line SCENE/FIGHTERS/BEATS/FINISH/STYLE sketch. Combat
#    choreography is derived from every CombatEvent — see
#    _CHOREOGRAPHY_NEUTRAL (movement-only events, style-independent)
#    and _choreography_line (impact events, drawn from style_profile's
#    own weapons/powers so choreography never invents vocabulary the
#    style profile didn't already select).
# ======================================================================

_CHOREOGRAPHY_NEUTRAL: dict[CombatEventType, str] = {
    CombatEventType.TERRITORIAL_ADVANCE: "circling and advancing, testing the range",
    CombatEventType.TACTICAL_SETUP: "a stance change, preparing the next exchange",
    CombatEventType.DEFENSIVE_REPOSITIONING: "a retreat into a fortified stance",
    CombatEventType.STRATEGIC_POSITIONING: "repositioning under pressure",
    CombatEventType.POWER_DEPLOYMENT: "committing their strongest asset to the field",
}

_ARMY_CHOREO_VERB: dict[CombatEventType, str] = {
    CombatEventType.TERRITORIAL_ADVANCE: "advances into position",
    CombatEventType.TACTICAL_SETUP: "maneuvers into formation",
    CombatEventType.ATTACK_LANDED: "engages the enemy line",
    CombatEventType.COORDINATED_ASSAULT: "presses a coordinated assault",
    CombatEventType.CRITICAL_THREAT: "threatens a breakthrough",
    CombatEventType.CALCULATED_SACRIFICE: "is deliberately exposed as bait",
    CombatEventType.DEFENSIVE_REPOSITIONING: "falls back to a fortified position",
    CombatEventType.BREAKTHROUGH_ATTACK: "breaks through the enemy line",
    CombatEventType.FINISHING_STRIKE: "delivers the decisive blow that ends the battle",
    CombatEventType.POWER_DEPLOYMENT: "commits to the field",
    CombatEventType.STRATEGIC_POSITIONING: "repositions under pressure",
}


def _piece_for_event(analysis: GameAnalysis, event) -> str | None:
    for move in analysis.moves:
        if move.move_label == event.move_label:
            return move.piece_moved
    return None


def _choreography_line(
    event, style_profile: StyleProfile, battle_mode: BattleModeIntelligence, piece: str | None
) -> str:
    if battle_mode.mode == BattleMode.ARMY:
        unit = battle_mode.unit_mapping.get(piece or "", "unit")
        verb = _ARMY_CHOREO_VERB.get(event.event_type, "continues the engagement")
        return f"{event.move_label}: the {unit} {verb}."

    neutral = _CHOREOGRAPHY_NEUTRAL.get(event.event_type)
    if neutral is not None:
        return f"{event.move_label}: {neutral}."

    weapon = style_profile.weapons[0] if style_profile.weapons else "a weapon"
    power = style_profile.powers[0] if style_profile.powers else "a technique"

    if event.event_type == CombatEventType.ATTACK_LANDED:
        return f"{event.move_label}: a {weapon} strike connects."
    if event.event_type == CombatEventType.COORDINATED_ASSAULT:
        return f"{event.move_label}: a rapid combo of strikes presses the advantage."
    if event.event_type == CombatEventType.CRITICAL_THREAT:
        return f"{event.move_label}: overwhelming pressure with {power}, forcing a response."
    if event.event_type == CombatEventType.CALCULATED_SACRIFICE:
        return f"{event.move_label}: a deliberate opening, baiting a counter-opportunity."
    if event.event_type == CombatEventType.BREAKTHROUGH_ATTACK:
        return f"{event.move_label}: guard shattered — the {weapon} breaks straight through."
    if event.event_type == CombatEventType.FINISHING_STRIKE:
        return f"{event.move_label}: the finishing blow — {power} lands a final, cinematic impact."
    return f"{event.move_label}: the exchange continues."


def _build_prompt(
    analysis: GameAnalysis,
    combat: CombatIntelligence,
    battle: BattleIntelligence,
    style_profile: StyleProfile,
    battle_mode: BattleModeIntelligence,
) -> str:
    choreography = "\n".join(
        f"- {_choreography_line(e, style_profile, battle_mode, _piece_for_event(analysis, e))}"
        for e in combat.events
    ) or "- (no moves)"
    weapons = ", ".join(style_profile.weapons) or "none"
    powers = ", ".join(style_profile.powers) or "none"
    effects = ", ".join(style_profile.visual_effects) or "none"
    winner_label = _side_label(analysis.winner) if analysis.winner in ("white", "black") else "Neither side"

    is_army = battle_mode.mode == BattleMode.ARMY
    unit_header = "FORCES" if is_army else "FIGHTERS"
    white_sub = "White Force" if is_army else "White Fighter"
    black_sub = "Black Force" if is_army else "Black Fighter"

    return (
        "STYLE\n"
        f"{style_profile.style.value}\n\n"
        "SETTING\n"
        f"{analysis.opening}\n\n"
        "ENVIRONMENT\n"
        f"{style_profile.environment} — {battle_mode.environment}\n\n"
        "LIGHTING\n"
        f"{_lighting_for(battle)}\n\n"
        "CAMERA\n"
        f"{_camera_for(battle, battle_mode)}\n\n"
        f"{unit_header}\n"
        f"{white_sub}\n"
        f"{battle.fighter_personality.white.label} — {battle.fighter_personality.white.rationale}\n\n"
        f"{black_sub}\n"
        f"{battle.fighter_personality.black.label} — {battle.fighter_personality.black.rationale}\n\n"
        "PERSONALITIES\n"
        f"White: {battle.fighter_personality.white.label}\n"
        f"Black: {battle.fighter_personality.black.label}\n\n"
        "BATTLE SCALE\n"
        f"{battle_mode.scale} — focus: {', '.join(battle_mode.combat_focus)}\n\n"
        "VISUAL STYLE\n"
        f"{style_profile.style.value}\n\n"
        "WEAPONS\n"
        f"{weapons}\n\n"
        "POWERS\n"
        f"{powers}\n\n"
        "VISUAL EFFECTS\n"
        f"{effects}\n\n"
        "COMBAT CHOREOGRAPHY\n"
        f"{choreography}\n\n"
        "ENDING\n"
        f"{winner_label} — {style_profile.finisher}\n\n"
        "FINAL SHOT\n"
        f"{_final_shot_for(battle, style_profile)}"
    )


def _lighting_for(battle: BattleIntelligence) -> str:
    if battle.combat_style in (CombatStyle.OVERWHELMING, CombatStyle.AGGRESSIVE, CombatStyle.CHAOTIC):
        return "harsh, high-contrast lighting with sudden flares on impact"
    if battle.combat_style in (CombatStyle.DEFENSIVE, CombatStyle.PATIENT, CombatStyle.DESPERATE):
        return "dim, tense lighting that slowly brightens toward the finish"
    return "even, dramatic lighting throughout"


def _camera_for(battle: BattleIntelligence, battle_mode: BattleModeIntelligence) -> str:
    if battle_mode.mode == BattleMode.ARMY:
        if battle.battle_arc in (BattleArc.BLITZ_EXECUTION, BattleArc.GAMBIT_ASSAULT):
            return "wide battlefield shots with rapid cuts between flanks"
        if battle.battle_arc in (BattleArc.WAR_OF_ATTRITION, BattleArc.SIEGE):
            return "wide, sweeping battlefield shots that slowly close in as the siege wears on"
        return "wide battlefield shots establishing the full scale of the conflict"

    if battle.battle_arc in (BattleArc.BLITZ_EXECUTION, BattleArc.GAMBIT_ASSAULT):
        return "close combat shots, fast cuts on impact frames"
    if battle.battle_arc in (BattleArc.WAR_OF_ATTRITION, BattleArc.SIEGE):
        return "close combat shots, wide and patient, tightening as the duel wears on"
    return "close combat shots, a mix of wide stance shots and tight reaction shots"


def _final_shot_for(battle: BattleIntelligence, style_profile: StyleProfile) -> str:
    return f"Hold on the aftermath in the {style_profile.environment} as {style_profile.finisher.lower()}"


# ======================================================================
# 6. estimated_length — battle pace, arc, combat event volume, tactical
#    density, ending type, and finishing intensity, not move count.
#    Returned as a range (matching the brief's own examples), rounded
#    to a clean 5-second bucket at each end. orchestrator.py's
#    _parse_seconds was updated to average a range like "25-40 sec" —
#    see that file's docstring note.
#
#    battle_arc is the primary driver rather than battle_pace: pace is
#    inferred upstream (combat_mapper.py, not modifiable here) from
#    time control or tactical density, and a short, decisive game with
#    no time-control header and few tactical flashes can get
#    classified "strategic" pace despite obviously being a blitz
#    execution — arc doesn't have that failure mode, since
#    blitz_execution requires both checkmate AND a short move count by
#    construction (see battle_director.py).
# ======================================================================

_ARC_BASE_RANGE: dict[BattleArc, tuple[int, int]] = {
    BattleArc.BLITZ_EXECUTION: (10, 15),
    BattleArc.GAMBIT_ASSAULT: (18, 28),
    BattleArc.TACTICAL_AMBUSH: (22, 32),
    BattleArc.FINAL_DUEL: (28, 40),
    BattleArc.COMEBACK: (35, 50),
    BattleArc.SIEGE: (45, 65),
    BattleArc.WAR_OF_ATTRITION: (55, 80),
}
_PACE_NUDGE = {"fast": -0.05, "moderate": 0.0, "strategic": 0.08}


def _estimated_length(
    analysis: GameAnalysis, combat: CombatIntelligence, battle: BattleIntelligence
) -> str:
    low, high = _ARC_BASE_RANGE.get(battle.battle_arc, (20, 35))

    # Small, bounded nudges on top of the arc-driven base — enough to
    # reflect tactical density/pace/finishing intensity without a
    # pace misclassification swamping the arc signal.
    pace_nudge = _PACE_NUDGE.get(combat.profile.battle_pace, 0.0)
    tactical_nudge = min(0.15, len(analysis.tactical_moments) * 0.03)
    finishing_intensity = combat.events[-1].intensity if combat.events else 5
    intensity_nudge = 0.05 if (combat.profile.ending_type == "checkmate" and finishing_intensity >= 9) else 0.0

    multiplier = 1.0 + pace_nudge + tactical_nudge + intensity_nudge
    low = max(10, min(90, round(low * multiplier / 5) * 5))
    high = max(10, min(90, round(high * multiplier / 5) * 5))
    if high <= low:
        high = min(90, low + 5)
    return f"{low}-{high} sec"


# ======================================================================
# Facts — unchanged from the previous revision.
# ======================================================================


def _describe_winner(analysis: GameAnalysis) -> str:
    if analysis.winner == "draw":
        return "Draw"
    if analysis.winner == "unknown":
        return "Result unresolved"
    label = "White" if analysis.winner == "white" else "Black"
    return f"{label} wins by checkmate" if analysis.is_checkmate else f"{label} wins by resignation"


class NarrativeGenerator:
    """`ai_provider` is accepted and stored but never called — kept
    only so FightOrchestrator's existing construction of this class
    doesn't need to change. See module docstring."""

    def __init__(self, ai_provider: AIProvider):
        self._ai_provider = ai_provider  # unused; see module docstring

    async def generate(
        self,
        analysis: GameAnalysis,
        combat: CombatIntelligence,
        battle: BattleIntelligence,
        style_profile: StyleProfile,
        battle_mode: BattleModeIntelligence,
    ) -> FightStory:
        return FightStory(
            winner=_describe_winner(analysis),
            opening=analysis.opening,
            fight_style=_fight_style(analysis, battle, style_profile),
            best_move=_best_move(analysis, combat, style_profile),
            turning_point=_turning_point(analysis, combat, style_profile),
            battle_summary=_battle_summary(analysis, battle, style_profile),
            prompt=_build_prompt(analysis, combat, battle, style_profile, battle_mode),
            estimated_length=_estimated_length(analysis, combat, battle),
        )
