"""
Style Engine — translates the SAME battle into a chosen visual universe
(anime, fantasy, modern_warfare, superhero, scifi). Presentation only:
nothing here computes, corrects, or reinterprets chess facts. It reads
BattleIntelligence and CombatIntelligence (already-computed
interpretations) and picks vocabulary — it never looks at a PGN or
GameAnalysis directly, and it can't change what actually happened in
the game.

Pipeline position:

    PGN -> Metadata Normalizer -> Analysis -> Combat Mapper ->
    Battle Director -> Style Engine (here) -> Narrative Generator

v1.4 note: the brief for this revision said "Do NOT modify Style
Engine" in one place and "expand style vocabulary significantly, to
10-20 items per category" in another — a direct contradiction within
the same document. Resolved as: expand the vocabulary DATA only.
`generate_style_profile`'s signature, `StyleProfile`'s schema, and the
three original selection axes (intensity/arc/pace/finisher-flavor) are
completely unchanged — existing callers see identical behavior in
kind, just drawing from larger, still-deterministic pools. Flagged
explicitly in the accompanying engineering review rather than resolved
silently.

Content safety: every word bank below uses generic genre archetypes,
not references to any specific copyrighted franchise, character, or
work. See tests/test_style_engine.py's
test_no_copyrighted_franchise_terms_anywhere for the concrete check.

Selection is deterministic, never random:

- intensity axis ("aggressive" | "defensive" | "balanced"), from
  BattleIntelligence.combat_style — drives weapons + powers together.
- arc axis ("dynamic" | "fortified" | "neutral"), from
  BattleIntelligence.battle_arc — drives the environment.
- pace axis ("fast" | "slow"), from CombatIntelligence.profile.battle_pace
  — drives visual_effects.
- finisher flavor ("decisive" | "grinding" | "open"), from
  CombatIntelligence.profile.ending_type + battle_arc.

New in v1.4: each axis bucket now selects from a much larger shared
pool (10-20 items) via a deterministic seed derived from
already-computed CombatIntelligence/BattleIntelligence signals (event
count, event intensities, personality label lengths) — never a random
number, timestamp, or UUID. The same battle always produces the same
seed, so the same PGN + style always produces the same profile; two
different battles landing in the same bucket (e.g. both "aggressive")
will typically draw different items from the pool, so the expanded
vocabulary actually gets used rather than only its first few entries.
"""

from __future__ import annotations

from products.chess2fight.schemas import (
    BattleArc,
    BattleIntelligence,
    CombatIntelligence,
    CombatStyle,
    StyleId,
    StyleProfile,
)

_AGGRESSIVE_STYLES = {CombatStyle.AGGRESSIVE, CombatStyle.OVERWHELMING, CombatStyle.CHAOTIC}
_DEFENSIVE_STYLES = {CombatStyle.DEFENSIVE, CombatStyle.PATIENT, CombatStyle.DESPERATE}

_DYNAMIC_ARCS = {BattleArc.BLITZ_EXECUTION, BattleArc.GAMBIT_ASSAULT, BattleArc.TACTICAL_AMBUSH}
_FORTIFIED_ARCS = {BattleArc.WAR_OF_ATTRITION, BattleArc.SIEGE}

# Accepts common variants of the 5 canonical style ids without changing
# GenerateRequest.style's type (it stays a free-form str, exactly as
# before) — anything unrecognized falls back to "anime", matching that
# field's own existing default.
_STYLE_ALIASES: dict[str, StyleId] = {
    "anime": StyleId.ANIME,
    "fantasy": StyleId.FANTASY,
    "modern_warfare": StyleId.MODERN_WARFARE,
    "modern-warfare": StyleId.MODERN_WARFARE,
    "modern warfare": StyleId.MODERN_WARFARE,
    "superhero": StyleId.SUPERHERO,
    "super_hero": StyleId.SUPERHERO,
    "scifi": StyleId.SCIFI,
    "sci_fi": StyleId.SCIFI,
    "sci-fi": StyleId.SCIFI,
}

# Bucket offsets: a small, fixed stride added to the seed before
# indexing into a style's shared pool, so different buckets (e.g.
# "aggressive" vs "defensive") land on different parts of the same
# pool rather than all rotating through it identically.
_BUCKET_OFFSET = {
    "aggressive": 0, "defensive": 4, "balanced": 8,
    "dynamic": 0, "fortified": 3, "neutral": 6,
    "fast": 0, "slow": 5,
}

STYLE_VOCAB: dict[StyleId, dict] = {
    StyleId.ANIME: {
        "weapons": [
            "katana", "dual blades", "chain scythe", "spirit spear", "giant sword",
            "chakra staff", "energy kunai", "demon blade", "shadow claws", "dragon halberd",
            "twin daggers", "iron war fan", "spirit chain", "war naginata",
        ],
        "powers": [
            "burning spirit energy", "energy aura", "chi burst", "phantom afterimage",
            "deflecting barrier", "focused inner power", "explosive energy burst",
            "spirit resonance", "elemental infusion", "shadow-step technique",
            "overwhelming presence", "silent focus",
        ],
        "environments": [
            "storm-swept rooftop", "mountain temple", "sakura-lined courtyard",
            "shattered dojo", "moonlit shrine", "cliffside arena", "abandoned bridge",
            "bamboo forest clearing", "temple ruins", "frozen waterfall",
        ],
        "visual_effects": [
            "speed lines", "afterimage trails", "rising energy glow",
            "swirling wind particles", "cracking ground shockline", "screen-flash impact frame",
            "trailing energy arcs", "drifting petal debris", "lens-flare burst",
            "slow-motion clash frame",
        ],
        "finishers": {
            "decisive": [
                "A single decisive sword strike ends the exchange in a flash.",
                "One perfectly timed technique closes the duel instantly.",
                "A blinding strike ends the confrontation before the echo fades.",
                "The finishing blow lands too fast for any counter.",
                "A single strike settles the duel outright.",
            ],
            "grinding": [
                "After a long, relentless exchange, one final technique finally breaks through.",
                "Following a drawn-out duel, a last decisive blow finally lands.",
                "After trading blow after blow, one technique finally proves decisive.",
                "The exhausting exchange ends with one final, hard-won strike.",
                "A prolonged duel is finally settled by one clean technique.",
            ],
            "open": [
                "The duel fades out without a single, clean finishing blow.",
                "Neither fighter lands a final decisive strike before the bout ends.",
                "The exchange winds down with no clear finishing technique.",
                "The confrontation ends unresolved, without a defining blow.",
                "The duel closes quietly, with no dramatic final strike.",
            ],
        },
    },
    StyleId.FANTASY: {
        "weapons": [
            "longsword", "battle axe", "war hammer", "enchanted blade", "great hammer",
            "longbow", "arcane staff", "holy sword", "ice spear", "rune axe",
            "crystal lance", "dragon blade", "elven bow", "war mace",
        ],
        "powers": [
            "ancient magic", "elemental channeling", "wild arcane fire", "berserker's rage",
            "warding runes", "protective ward", "divine blessing", "storm calling",
            "shadow binding", "healing light", "earthshaker's strength", "frost conjuring",
        ],
        "environments": [
            "ruined castle", "burning siege camp", "ancient stone keep", "misty battlefield",
            "sunken temple ruins", "frozen mountain pass", "enchanted forest clearing",
            "collapsed watchtower", "royal throne hall", "cursed battlefield",
        ],
        "visual_effects": [
            "flaring magical runes", "sparks of arcane energy", "swirling mist",
            "glowing sigils rising", "cracking stone debris", "shimmering ward light",
            "drifting embers", "banners torn by wind", "glinting blade reflections",
            "rolling storm clouds",
        ],
        "finishers": {
            "decisive": [
                "A single divine execution ends the duel in an instant.",
                "One masterful strike ends the confrontation outright.",
                "A final blow lands before any counter can be raised.",
                "The duel is settled in a single, decisive stroke.",
                "One strike, perfectly placed, ends the battle.",
            ],
            "grinding": [
                "After a grinding siege of blows, one final strike breaks the stalemate.",
                "Following a long and bitter struggle, one blow finally decides it.",
                "After a relentless exchange, a last strike finally breaks through.",
                "The prolonged struggle ends with one hard-won final blow.",
                "A drawn-out contest is finally settled by one decisive strike.",
            ],
            "open": [
                "The battle fades without either side landing a true final blow.",
                "Neither warrior delivers a clean, decisive strike before it ends.",
                "The confrontation winds down without a defining blow.",
                "The struggle ends unresolved, with no dramatic final stroke.",
                "The duel closes quietly, without a clear finishing blow.",
            ],
        },
    },
    StyleId.MODERN_WARFARE: {
        "weapons": [
            "assault rifle", "designated marksman rifle", "sniper rifle", "combat knife",
            "rocket launcher", "ballistic shield", "submachine gun", "sidearm",
            "breaching charge", "heavy machine gun", "riot shield", "fortified cover position",
        ],
        "powers": [
            "suppressing fire support", "adrenaline-fueled reflexes", "coordinated fire support",
            "battlefield awareness", "tactical night vision", "fortified overwatch position",
            "rapid extraction protocol", "squad coordination", "precision targeting system",
            "reinforced perimeter defense",
        ],
        "environments": [
            "urban battlefield", "collapsing overpass", "fortified compound",
            "trench-lined battlefield", "abandoned industrial complex", "rubble-strewn street",
            "besieged checkpoint", "smoke-covered courtyard", "war-torn city block",
            "fortified command outpost",
        ],
        "visual_effects": [
            "tracer rounds", "muzzle flashes", "drifting smoke grenades",
            "dust kicked up by impacts", "shell casings scattering", "flashbang bursts",
            "debris from explosions", "flickering emergency lighting", "rain-slicked reflections",
            "distant flare illumination",
        ],
        "finishers": {
            "decisive": [
                "A precision strike ends the engagement in a single shot.",
                "One well-placed shot ends the confrontation instantly.",
                "The engagement ends in a single, decisive shot.",
                "A single strike neutralizes the threat outright.",
                "One shot ends the firefight before a response is possible.",
            ],
            "grinding": [
                "After a sustained firefight, one final volley finally breaks through.",
                "Following a prolonged exchange, one last shot decides the engagement.",
                "After a drawn-out firefight, a final volley proves decisive.",
                "The extended engagement ends with one hard-won final shot.",
                "A long firefight is finally settled by one decisive volley.",
            ],
            "open": [
                "The engagement winds down without a clean, decisive shot.",
                "Neither side lands a clear finishing shot before it ends.",
                "The firefight fades out without a defining final shot.",
                "The engagement ends unresolved, with no decisive volley.",
                "The exchange closes quietly, without a clear finishing blow.",
            ],
        },
    },
    StyleId.SCIFI: {
        "weapons": [
            "plasma rifle", "energy blade", "railgun", "photon cannon", "drone swarm",
            "ion spear", "gravity hammer", "nano blade", "pulse carbine", "laser pike",
            "particle cannon", "energy shield", "deflector barrier",
        ],
        "powers": [
            "energy shields", "targeting matrix", "overcharged reactor burst",
            "kinetic amplifiers", "phase-shift evasion", "neural targeting link",
            "gravity well projection", "cloaking field", "shield overcharge",
            "adaptive armor plating",
        ],
        "environments": [
            "orbital station", "collapsing space station", "zero-gravity platform",
            "derelict star cruiser", "asteroid mining outpost", "fortified starbase",
            "planetary defense grid", "abandoned research station", "shattered space dock",
            "deep-space observation post",
        ],
        "visual_effects": [
            "holographic particles", "streaking energy trails", "pulsing energy fields",
            "drifting starlight", "sparking circuitry", "shimmering shield impacts",
            "floating debris fragments", "flickering hologram static", "ionized air distortion",
            "reflective hull glare",
        ],
        "finishers": {
            "decisive": [
                "An antimatter blast ends the confrontation in an instant.",
                "One precise shot ends the confrontation outright.",
                "A single decisive blast neutralizes the target instantly.",
                "The confrontation ends in a single, overwhelming strike.",
                "One shot ends the engagement before a response is possible.",
            ],
            "grinding": [
                "After a prolonged exchange, one final barrage finally breaks through.",
                "Following a sustained engagement, one last strike decides it.",
                "After a drawn-out battle, a final barrage proves decisive.",
                "The extended confrontation ends with one hard-won final blast.",
                "A long engagement is finally settled by one decisive strike.",
            ],
            "open": [
                "The confrontation fades without a single, decisive blast.",
                "Neither side lands a clear finishing blow before it ends.",
                "The engagement winds down without a defining final strike.",
                "The confrontation ends unresolved, with no decisive blast.",
                "The exchange closes quietly, without a clear finishing shot.",
            ],
        },
    },
    StyleId.SUPERHERO: {
        "weapons": [
            "reinforced gauntlets", "kinetic strike gloves", "impact-resistant armor",
            "energy barrier generator", "reinforced armor plating", "shock-absorbing boots",
            "power-channeling gauntlets", "defensive energy plating", "force-field bracers",
            "impact dampening suit",
        ],
        "powers": [
            "enhanced strength", "kinetic energy channeling", "energy manipulation",
            "lightning generation", "flight", "telekinesis", "gravity control",
            "cosmic beam projection", "ice generation", "sonic shockwave",
            "time-slowing perception", "photon shield generation",
        ],
        "environments": [
            "metropolitan skyline", "collapsing skyscraper", "crowded highway overpass",
            "downtown financial district", "harborside industrial zone", "rooftop helipad",
            "elevated train platform", "city bridge crossing", "underground transit tunnel",
            "stadium under construction",
        ],
        "visual_effects": [
            "shockwaves", "cracking pavement bursts", "glowing energy buildup",
            "settling dust and debris", "shattering glass fragments", "rippling air distortion",
            "flickering streetlights", "wind-blown debris trails", "afterglow impact rings",
            "collapsing structural debris",
        ],
        "finishers": {
            "decisive": [
                "A single finishing smash ends the confrontation instantly.",
                "One decisive blow ends the confrontation outright.",
                "A single overwhelming strike ends it before a response is possible.",
                "The confrontation ends in one perfectly landed blow.",
                "One strike settles the confrontation outright.",
            ],
            "grinding": [
                "After trading blow after blow, one final strike finally breaks through.",
                "Following a prolonged clash, one last blow finally decides it.",
                "After a drawn-out confrontation, a final strike proves decisive.",
                "The extended clash ends with one hard-won final blow.",
                "A long confrontation is finally settled by one decisive strike.",
            ],
            "open": [
                "The confrontation winds down without a single, decisive blow.",
                "Neither side lands a clear finishing strike before it ends.",
                "The clash fades out without a defining final blow.",
                "The confrontation ends unresolved, with no decisive strike.",
                "The exchange closes quietly, without a clear finishing blow.",
            ],
        },
    },
}


def _resolve_style(requested: str) -> StyleId:
    normalized = (requested or "").strip().lower()
    if normalized in _STYLE_ALIASES:
        return _STYLE_ALIASES[normalized]
    try:
        return StyleId(normalized.replace(" ", "_").replace("-", "_"))
    except ValueError:
        return StyleId.ANIME  # same safe default as GenerateRequest.style


def resolve_style_id(requested: str) -> StyleId:
    """Public entry point for the same lenient string->StyleId
    resolution `generate_style_profile` already does internally — used
    by api/chess2fight.py to build a default BattlePreferences from the
    legacy top-level `style` string, so an unrecognized style name
    degrades exactly the same way it always has (falls back to anime,
    never a validation error)."""
    return _resolve_style(requested)


def _intensity_axis(combat_style: CombatStyle) -> str:
    if combat_style in _AGGRESSIVE_STYLES:
        return "aggressive"
    if combat_style in _DEFENSIVE_STYLES:
        return "defensive"
    return "balanced"


def _arc_axis(battle_arc: BattleArc) -> str:
    if battle_arc in _DYNAMIC_ARCS:
        return "dynamic"
    if battle_arc in _FORTIFIED_ARCS:
        return "fortified"
    return "neutral"


def _pace_axis(battle_pace: str) -> str:
    return "fast" if battle_pace == "fast" else "slow"


def _finisher_flavor(battle: BattleIntelligence, combat: CombatIntelligence) -> str:
    if combat.profile.ending_type != "checkmate":
        # Nothing on the board was actually a final decisive blow —
        # a resignation, draw, or time-forfeit shouldn't be narrated
        # as one. See module docstring.
        return "open"
    if battle.battle_arc in (BattleArc.WAR_OF_ATTRITION, BattleArc.SIEGE):
        return "grinding"
    return "decisive"


def _intelligence_seed(battle: BattleIntelligence, combat: CombatIntelligence) -> int:
    """A small, fully deterministic integer derived from
    already-computed CombatIntelligence/BattleIntelligence signals —
    never random, never time-based, never a UUID. The same battle
    always yields the same seed; different battles typically yield
    different seeds, spreading selection across the expanded pools."""
    return (
        len(combat.events)
        + sum(e.intensity for e in combat.events)
        + len(battle.fighter_personality.white.label)
        + len(battle.fighter_personality.black.label)
    )


def _pick_many(pool: list[str], seed: int, bucket: str, count: int = 3) -> list[str]:
    if not pool:
        return []
    start = (seed + _BUCKET_OFFSET.get(bucket, 0)) % len(pool)
    return [pool[(start + i) % len(pool)] for i in range(min(count, len(pool)))]


def _pick_one(pool: list[str], seed: int, bucket: str = "") -> str:
    if not pool:
        return ""
    index = (seed + _BUCKET_OFFSET.get(bucket, 0)) % len(pool)
    return pool[index]


def generate_style_profile(
    battle: BattleIntelligence, combat: CombatIntelligence, style: str
) -> StyleProfile:
    """Single public entry point: BattleIntelligence + CombatIntelligence
    + the requested style string in, StyleProfile out. Pure function —
    no AI provider, no I/O, deterministic (same inputs always produce
    the same profile)."""
    style_id = _resolve_style(style)
    vocab = STYLE_VOCAB[style_id]
    seed = _intelligence_seed(battle, combat)

    intensity = _intensity_axis(battle.combat_style)
    arc_axis = _arc_axis(battle.battle_arc)
    pace = _pace_axis(combat.profile.battle_pace)
    finisher_flavor = _finisher_flavor(battle, combat)

    return StyleProfile(
        style=style_id,
        weapons=_pick_many(vocab["weapons"], seed, intensity),
        powers=_pick_many(vocab["powers"], seed, intensity),
        environment=_pick_one(vocab["environments"], seed, arc_axis),
        visual_effects=_pick_many(vocab["visual_effects"], seed, pace),
        finisher=_pick_one(vocab["finishers"][finisher_flavor], seed),
    )
