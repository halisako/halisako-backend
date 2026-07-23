"""Tests for style_engine.

Covers the brief's explicit requirement (same PGN -> different
style_profile per style), the content-safety constraint (no
copyrighted franchise references), and the individual axis-selection
logic via hand-built fixtures."""

from products.chess2fight.battle_director import generate_battle_intelligence
from products.chess2fight.combat_mapper import generate_combat_intelligence
from products.chess2fight.pgn_analyzer import analyze_game
from products.chess2fight.style_engine import (
    STYLE_VOCAB,
    _intelligence_seed,
    _pick_many,
    _pick_one,
    generate_style_profile,
)
from products.chess2fight.schemas import (
    BattleArc,
    BattleIntelligence,
    CombatProfile,
    CombatIntelligence,
    CombatStyle,
    FighterPersonality,
    PersonalityProfile,
    StyleId,
)

SCHOLARS_MATE = """[Event "Example"]
[White "Halisako"]
[Black "Guest"]
[Result "1-0"]

1. e4 e5 2. Qh5 Nc6 3. Bc4 Nf6 4. Qxf7# 1-0"""

ALL_STYLES = ["anime", "fantasy", "modern_warfare", "superhero", "scifi"]

BANNED_TERMS = ["naruto", "marvel", "dc", "star wars", "demon slayer"]


def _run(pgn: str, style: str):
    analysis = analyze_game(pgn)
    combat = generate_combat_intelligence(analysis)
    battle = generate_battle_intelligence(analysis, combat)
    profile = generate_style_profile(battle, combat, style)
    return profile


# --- Required test: same PGN, different style outputs -------------------


def test_same_pgn_produces_different_profile_per_style():
    profiles = {style: _run(SCHOLARS_MATE, style) for style in ALL_STYLES}

    # Each profile reports back the style it was asked for.
    for style, profile in profiles.items():
        assert profile.style.value == style

    # The profiles are genuinely different from each other, not just
    # relabeled copies — check across every text-bearing field.
    environments = {p.environment for p in profiles.values()}
    finishers = {p.finisher for p in profiles.values()}
    weapons = {tuple(p.weapons) for p in profiles.values()}
    assert len(environments) == len(ALL_STYLES)
    assert len(finishers) == len(ALL_STYLES)
    assert len(weapons) == len(ALL_STYLES)


def test_all_five_styles_are_covered_in_the_vocab_table():
    assert set(STYLE_VOCAB.keys()) == {
        StyleId.ANIME, StyleId.FANTASY, StyleId.MODERN_WARFARE,
        StyleId.SUPERHERO, StyleId.SCIFI,
    }


# --- Content safety -------------------------------------------------------


def test_no_copyrighted_franchise_terms_anywhere():
    """Flat-scan the entire vocabulary table (not just one generated
    profile) so this catches a banned term even if no test PGN happens
    to select the branch it's hiding in."""
    blob: list[str] = []
    for style_data in STYLE_VOCAB.values():
        for category, value in style_data.items():
            if category == "finishers":
                for flavor_list in value.values():
                    blob.extend(flavor_list)
            elif isinstance(value, list):
                blob.extend(value)
            else:
                blob.append(value)
    full_text = " ".join(blob).lower()
    for banned in BANNED_TERMS:
        assert banned not in full_text, f"banned term '{banned}' found in style vocabulary"


# --- Fallback / graceful handling of the existing free-form style field --


def test_unknown_style_string_falls_back_to_anime():
    profile = _run(SCHOLARS_MATE, "noir detective")
    assert profile.style == StyleId.ANIME


def test_style_string_variants_are_normalized():
    assert _run(SCHOLARS_MATE, "Sci-Fi").style == StyleId.SCIFI
    assert _run(SCHOLARS_MATE, "MODERN WARFARE").style == StyleId.MODERN_WARFARE
    assert _run(SCHOLARS_MATE, "  anime  ").style == StyleId.ANIME


# --- Axis selection, via hand-built fixtures ------------------------------


def _battle(arc: BattleArc, combat_style: CombatStyle) -> BattleIntelligence:
    profile = PersonalityProfile(label="The Contender", rationale="test fixture")
    return BattleIntelligence(
        battle_arc=arc, combat_style=combat_style,
        fighter_personality=FighterPersonality(white=profile, black=profile),
    )


def _combat(pace: str = "fast", ending_type: str = "checkmate") -> CombatIntelligence:
    return CombatIntelligence(
        events=[],
        profile=CombatProfile(
            battle_pace=pace, fighter_balance="even", ending_type=ending_type, winner="white"
        ),
    )


def test_aggressive_combat_style_selects_from_aggressive_bucket():
    battle = _battle(BattleArc.FINAL_DUEL, CombatStyle.AGGRESSIVE)
    combat = _combat()
    profile = generate_style_profile(battle, combat, "anime")
    seed = _intelligence_seed(battle, combat)
    expected = _pick_many(STYLE_VOCAB[StyleId.ANIME]["weapons"], seed, "aggressive")
    assert profile.weapons == expected
    # And it's actually different from what the defensive bucket would pick.
    defensive_battle = _battle(BattleArc.FINAL_DUEL, CombatStyle.DEFENSIVE)
    defensive_profile = generate_style_profile(defensive_battle, combat, "anime")
    assert defensive_profile.weapons != profile.weapons


def test_defensive_combat_style_selects_from_defensive_bucket():
    battle = _battle(BattleArc.FINAL_DUEL, CombatStyle.DEFENSIVE)
    combat = _combat()
    profile = generate_style_profile(battle, combat, "fantasy")
    seed = _intelligence_seed(battle, combat)
    expected = _pick_many(STYLE_VOCAB[StyleId.FANTASY]["powers"], seed, "defensive")
    assert profile.powers == expected


def test_war_of_attrition_selects_from_fortified_bucket():
    battle = _battle(BattleArc.WAR_OF_ATTRITION, CombatStyle.CALCULATED)
    combat = _combat()
    profile = generate_style_profile(battle, combat, "scifi")
    seed = _intelligence_seed(battle, combat)
    expected = _pick_one(STYLE_VOCAB[StyleId.SCIFI]["environments"], seed, "fortified")
    assert profile.environment == expected


def test_non_checkmate_ending_never_gets_a_decisive_finisher():
    """A resignation shouldn't be narrated as an on-screen final blow —
    see module docstring."""
    battle = _battle(BattleArc.BLITZ_EXECUTION, CombatStyle.OVERWHELMING)
    combat = _combat(ending_type="resignation")
    profile = generate_style_profile(battle, combat, "superhero")
    seed = _intelligence_seed(battle, combat)
    assert profile.finisher in STYLE_VOCAB[StyleId.SUPERHERO]["finishers"]["open"]
    assert profile.finisher == _pick_one(STYLE_VOCAB[StyleId.SUPERHERO]["finishers"]["open"], seed)


def test_war_of_attrition_checkmate_gets_grinding_finisher():
    battle = _battle(BattleArc.WAR_OF_ATTRITION, CombatStyle.CALCULATED)
    combat = _combat(ending_type="checkmate")
    profile = generate_style_profile(battle, combat, "modern_warfare")
    assert profile.finisher in STYLE_VOCAB[StyleId.MODERN_WARFARE]["finishers"]["grinding"]


def test_vocab_pools_meet_the_expanded_size_target():
    """Section 7 of the brief: ~10-20 options per category per style."""
    for style_id, vocab in STYLE_VOCAB.items():
        assert 10 <= len(vocab["weapons"]) <= 20, style_id
        assert 10 <= len(vocab["powers"]) <= 20, style_id
        assert 10 <= len(vocab["environments"]) <= 20, style_id
        assert 10 <= len(vocab["visual_effects"]) <= 20, style_id


def test_different_battles_draw_different_items_from_the_same_bucket():
    """The whole point of the seed mechanism: two different battles
    landing in the same intensity bucket should usually surface
    different vocabulary, not always the pool's first N entries."""
    combat_a = _combat()
    combat_b = CombatIntelligence(
        events=[], profile=CombatProfile(
            battle_pace="fast", fighter_balance="even", ending_type="checkmate", winner="white"
        ),
    )
    battle_a = _battle(BattleArc.FINAL_DUEL, CombatStyle.AGGRESSIVE)
    profile_a = PersonalityProfile(label="The Cornered Defender", rationale="x")
    battle_b = BattleIntelligence(
        battle_arc=BattleArc.FINAL_DUEL, combat_style=CombatStyle.AGGRESSIVE,
        fighter_personality=FighterPersonality(white=profile_a, black=profile_a),
    )
    a = generate_style_profile(battle_a, combat_a, "anime")
    b = generate_style_profile(battle_b, combat_b, "anime")
    assert a.weapons != b.weapons
