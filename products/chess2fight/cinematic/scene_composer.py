"""Scene Composer: ShotTimeline + BattleIntelligence + StyleProfile +
BattleModeIntelligence -> ComposedTimeline.

Runs immediately after the Timeline Engine:

    Timeline Engine -> Scene Composer

Its one job is visual continuity: derive a single, persistent
description of both fighters' appearance, the arena, the weather, the
time of day, the art style, and the color palette — once per battle —
and attach that same description to every shot. Nothing here generates
a prompt string or an image; every output is structured, reusable
data. A future renderer reading any single EnrichedShot in isolation
(out of order, or without the rest of the timeline) has everything it
needs to keep that frame consistent with every other frame in the
scene, because the `scene` field on every shot is the exact same
value, not independently regenerated per shot.

Arena layout deliberately reuses BattleModeIntelligence's own
`environment` field rather than deriving a second description from
battle_arc — see `compose_scene`'s docstring for why: continuity means
staying consistent with the rest of the response too, not just across
this module's own shots.

Deterministic throughout: same BattleIntelligence + StyleProfile
always produce the same SceneContinuity, using the same
seed-from-already-computed-signals approach style_engine.py
established, so two different battles landing in the same intensity
bucket still typically draw different (but each individually
reproducible) appearance/setting choices.
"""

from __future__ import annotations

from products.chess2fight.cinematic.schemas import (
    ArenaContinuity,
    ComposedTimeline,
    EnrichedShot,
    FighterAppearance,
    SceneContinuity,
    ShotTimeline,
)
from products.chess2fight.schemas import (
    BattleIntelligence,
    BattleModeIntelligence,
    CombatStyle,
    StyleId,
    StyleProfile,
)

_AGGRESSIVE_STYLES = {CombatStyle.AGGRESSIVE, CombatStyle.OVERWHELMING, CombatStyle.CHAOTIC}
_DEFENSIVE_STYLES = {CombatStyle.DEFENSIVE, CombatStyle.PATIENT, CombatStyle.DESPERATE}

# --- Appearance vocabulary, per style, per intensity bucket -----------------
# Each style gets a shared pool per attribute; the intensity axis picks
# a different rotation of the SAME pool, matching the pattern
# style_engine.py already established for weapons/powers.

_APPEARANCE_VOCAB: dict[StyleId, dict[str, list[str]]] = {
    StyleId.ANIME: {
        "hair": ["spiky jet-black hair", "long silver hair tied back", "windswept crimson hair", "short, sharply cropped hair"],
        "facial_features": ["a determined, narrow-eyed gaze", "a calm, focused expression", "a sharp jawline with a faint scar", "bright, intense eyes"],
        "clothing": ["a flowing high-collared coat", "a fitted training gi", "a battle-worn tunic with trailing sash", "layered combat wraps"],
        "armor": ["light shoulder plating", "a reinforced chest guard", "minimal wrist and shin guards", "an ornate segmented breastplate"],
    },
    StyleId.FANTASY: {
        "hair": ["long braided hair", "a wild, unkempt mane", "hair pulled back beneath a circlet", "short-cropped, weathered hair"],
        "facial_features": ["a weathered, battle-worn face", "sharp, noble features", "a stern brow and steady gaze", "a scarred cheek and firm jaw"],
        "clothing": ["a hooded traveling cloak", "layered leather and cloth", "a tabard bearing a house crest", "simple, worn traveling clothes"],
        "armor": ["a full plate cuirass", "chainmail beneath a surcoat", "banded leather armor", "ornate engraved vambraces"],
    },
    StyleId.MODERN_WARFARE: {
        "hair": ["a tight military buzz cut", "hair pulled back under a helmet", "short, practical hair", "a close-cropped tactical cut"],
        "facial_features": ["a hardened, focused expression", "tired but alert eyes", "a stubbled jaw", "camouflage face paint"],
        "clothing": ["a tactical combat uniform", "a weathered field jacket", "layered tactical fatigues", "a reinforced combat vest over fatigues"],
        "armor": ["a plate carrier with pouches", "a ballistic helmet and vest", "modular tactical armor plating", "a lightweight tactical harness"],
    },
    StyleId.SCIFI: {
        "hair": ["a sleek, close-cropped cut", "hair concealed beneath a helmet visor", "short hair with a faint neon undercut", "tightly bound hair beneath a neural interface"],
        "facial_features": ["a calm, augmented gaze", "faint bio-luminescent markings", "a focused, visor-lit expression", "sharp features under HUD glow"],
        "clothing": ["a sleek pressure suit", "an insulated flight uniform", "a form-fitting combat exosuit", "layered thermal undersuit and plating"],
        "armor": ["a segmented energy-shielded exosuit", "reactive plating with glowing seams", "a lightweight deflector harness", "reinforced composite plating"],
    },
    StyleId.SUPERHERO: {
        "hair": ["a slicked-back heroic style", "windswept, dynamic hair", "short, clean-cut hair", "a bold, gravity-defying style"],
        "facial_features": ["a square, determined jaw", "bright, confident eyes", "a stoic, unwavering expression", "a faint determined scowl"],
        "clothing": ["a form-fitting emblemed suit", "a flowing hero's cape over armor", "a sleek tactical bodysuit", "a reinforced civilian-styled uniform"],
        "armor": ["energy-channeling gauntlets and boots", "a chest emblem plate with shoulder guards", "reinforced impact-resistant plating", "lightweight kinetic-dampening armor"],
    },
}

# --- Weather, time-of-day, art style, palette --------------------------------
#
# Arena layout is deliberately NOT computed here from battle_arc — see
# `compose_scene`'s docstring for why it reuses BattleModeIntelligence's
# own environment description instead of deriving a second, potentially
# inconsistent one.

_WEATHER_BY_COMBAT_STYLE: dict[CombatStyle, str] = {
    CombatStyle.AGGRESSIVE: "a gathering storm, wind picking up",
    CombatStyle.DEFENSIVE: "still, heavy air",
    CombatStyle.BALANCED: "clear, calm conditions",
    CombatStyle.CALCULATED: "cool, still air with a light haze",
    CombatStyle.CHAOTIC: "swirling wind with debris in the air",
    CombatStyle.PATIENT: "a slow, settling mist",
    CombatStyle.DESPERATE: "a cold wind cutting through the scene",
    CombatStyle.OVERWHELMING: "a full storm breaking overhead",
}

_TIME_OF_DAY_BY_COMBAT_STYLE: dict[CombatStyle, str] = {
    CombatStyle.AGGRESSIVE: "high noon, harsh overhead sun",
    CombatStyle.DEFENSIVE: "dusk, fading light",
    CombatStyle.BALANCED: "late afternoon, golden hour",
    CombatStyle.CALCULATED: "early morning, flat even light",
    CombatStyle.CHAOTIC: "an unstable, flickering half-light",
    CombatStyle.PATIENT: "twilight, slow-fading light",
    CombatStyle.DESPERATE: "deep dusk, nearly night",
    CombatStyle.OVERWHELMING: "a storm-darkened midday",
}

_LIGHTING_CONTINUITY_BY_COMBAT_STYLE: dict[CombatStyle, str] = {
    CombatStyle.AGGRESSIVE: "A single hard key light is held from the same direction throughout, "
        "with sudden flares timed to impacts — the harshness never softens.",
    CombatStyle.DEFENSIVE: "A dim, even key light is held constant throughout, brightening only "
        "slightly as the scene resolves.",
    CombatStyle.BALANCED: "A neutral, evenly balanced three-point setup is held constant across "
        "every shot, with no dramatic shifts.",
    CombatStyle.CALCULATED: "A controlled, low-contrast key light is held constant, emphasizing "
        "clarity over drama in every shot.",
    CombatStyle.CHAOTIC: "Lighting direction shifts unpredictably shot to shot, but color "
        "temperature and overall exposure are held constant so the chaos reads as intentional.",
    CombatStyle.PATIENT: "A soft, low-intensity key light is held constant throughout, with only "
        "a slow, gradual brightening toward the finish.",
    CombatStyle.DESPERATE: "A dim, cool key light is held constant, with practical highlights "
        "consistently placed on whichever fighter is under pressure.",
    CombatStyle.OVERWHELMING: "A harsh, high-contrast key light is held constant from a single "
        "source, with escalating flare intensity building toward the finish.",
}

_ART_STYLE_BY_STYLE: dict[StyleId, str] = {
    StyleId.ANIME: "cel-shaded anime with dynamic speed lines and dramatic framing",
    StyleId.FANTASY: "painterly high-fantasy realism with rich, textured detail",
    StyleId.MODERN_WARFARE: "gritty, desaturated tactical realism",
    StyleId.SCIFI: "sleek sci-fi realism with holographic overlays and chrome highlights",
    StyleId.SUPERHERO: "bold, high-contrast comic-inspired realism",
}

_PALETTE_BY_STYLE_AND_INTENSITY: dict[StyleId, dict[str, list[str]]] = {
    StyleId.ANIME: {
        "aggressive": ["crimson red", "charcoal black", "electric white"],
        "defensive": ["deep indigo", "slate gray", "pale gold"],
        "balanced": ["crimson red", "deep indigo", "warm gold"],
    },
    StyleId.FANTASY: {
        "aggressive": ["blood crimson", "burnished bronze", "ash gray"],
        "defensive": ["forest green", "iron gray", "muted silver"],
        "balanced": ["deep emerald", "burnished gold", "royal purple"],
    },
    StyleId.MODERN_WARFARE: {
        "aggressive": ["rust orange", "gunmetal gray", "ash black"],
        "defensive": ["olive drab", "steel blue", "concrete gray"],
        "balanced": ["desert tan", "gunmetal gray", "faded olive"],
    },
    StyleId.SCIFI: {
        "aggressive": ["neon crimson", "void black", "chrome silver"],
        "defensive": ["deep cyan", "cool slate", "chrome silver"],
        "balanced": ["neon cyan", "chrome silver", "void black"],
    },
    StyleId.SUPERHERO: {
        "aggressive": ["bold crimson", "deep navy", "solar gold"],
        "defensive": ["steel blue", "slate gray", "muted silver"],
        "balanced": ["bold crimson", "deep navy", "solar gold"],
    },
}


def _intensity_axis(combat_style: CombatStyle) -> str:
    if combat_style in _AGGRESSIVE_STYLES:
        return "aggressive"
    if combat_style in _DEFENSIVE_STYLES:
        return "defensive"
    return "balanced"


def _seed(battle: BattleIntelligence) -> int:
    """A small, fully deterministic integer derived from
    already-computed BattleIntelligence signals — never random. Same
    battle always yields the same seed; different battles typically
    yield different seeds, spreading appearance selection across the
    vocabulary pools rather than always landing on the first entry."""
    return (
        len(battle.fighter_personality.white.label)
        + len(battle.fighter_personality.white.rationale)
        + len(battle.fighter_personality.black.label)
        + len(battle.fighter_personality.black.rationale)
    )


def _pick(pool: list[str], seed: int, offset: int) -> str:
    if not pool:
        return "unspecified"
    return pool[(seed + offset) % len(pool)]


def _fighter_appearance(style_id: StyleId, style_profile: StyleProfile, seed: int, offset: int) -> FighterAppearance:
    vocab = _APPEARANCE_VOCAB[style_id]
    weapons = style_profile.weapons or ["an unarmed stance"]
    return FighterAppearance(
        hair=_pick(vocab["hair"], seed, offset),
        facial_features=_pick(vocab["facial_features"], seed, offset),
        clothing=_pick(vocab["clothing"], seed, offset),
        armor=_pick(vocab["armor"], seed, offset),
        weapon=weapons[offset % len(weapons)],
    )


def compose_scene(
    timeline: ShotTimeline,
    battle: BattleIntelligence,
    style_profile: StyleProfile,
    battle_mode: BattleModeIntelligence,
) -> ComposedTimeline:
    """Enriches every shot in a ShotTimeline with persistent scene
    continuity data.

    Args:
        timeline: The Timeline Engine's output — the shots to enrich.
        battle: Drives weather, time of day, lighting continuity, and
            appearance-selection intensity.
        style_profile: Drives weapons, art style, and color palette.
        battle_mode: Its own `environment` field already describes the
            arena consistently with how the rest of the response
            frames the scene (a duel arena vs. an army battlefield,
            already varied by battle_arc in battle_mode_engine.py) —
            `arena.layout` reuses that value directly rather than
            deriving a second, independent description that could
            drift out of sync with it (e.g. describing "a symmetric
            dueling ground" for a response whose screenplay already
            says "FORCES" and a battlefield).

    Returns:
        A ComposedTimeline whose every shot carries the exact same
        SceneContinuity value — the continuity guarantee is structural,
        not something a consumer has to check for.
    """
    seed = _seed(battle)
    intensity = _intensity_axis(battle.combat_style)

    white = _fighter_appearance(style_profile.style, style_profile, seed, offset=0)
    black = _fighter_appearance(style_profile.style, style_profile, seed, offset=1)

    arena = ArenaContinuity(
        layout=battle_mode.environment,
        weather=_WEATHER_BY_COMBAT_STYLE.get(battle.combat_style, "clear, calm conditions"),
        time_of_day=_TIME_OF_DAY_BY_COMBAT_STYLE.get(battle.combat_style, "late afternoon, even light"),
    )

    palette_pool = _PALETTE_BY_STYLE_AND_INTENSITY[style_profile.style]
    color_palette = palette_pool.get(intensity, palette_pool["balanced"])

    continuity = SceneContinuity(
        white_fighter=white,
        black_fighter=black,
        arena=arena,
        lighting_continuity=_LIGHTING_CONTINUITY_BY_COMBAT_STYLE.get(
            battle.combat_style, "A neutral, evenly balanced lighting setup is held constant throughout."
        ),
        cinematic_art_style=_ART_STYLE_BY_STYLE[style_profile.style],
        color_palette=list(color_palette),
    )

    enriched_shots = [
        EnrichedShot(**shot.model_dump(), scene=continuity) for shot in timeline.shots
    ]

    return ComposedTimeline(
        shots=enriched_shots,
        total_duration_seconds=timeline.total_duration_seconds,
        shot_count=timeline.shot_count,
        scene_continuity=continuity,
    )
