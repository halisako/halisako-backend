"""Integration tests: the Scene Composer's output as it actually
appears in the real HTTP API response."""

from fastapi.testclient import TestClient

from main import app

client = TestClient(app)

SCHOLARS_MATE_PGN = (
    '[Event "Example"]\n[White "Halisako"]\n[Black "Guest"]\n[Result "1-0"]\n\n'
    "1. e4 e5 2. Qh5 Nc6 3. Bc4 Nf6 4. Qxf7# 1-0"
)


def test_generate_response_includes_scene_composition():
    res = client.post("/api/v1/chess2fight/generate", json={"pgn": SCHOLARS_MATE_PGN, "style": "anime"})
    assert res.status_code == 200
    assert "scene_composition" in res.json()


def test_scene_composition_has_correct_shape():
    res = client.post("/api/v1/chess2fight/generate", json={"pgn": SCHOLARS_MATE_PGN, "style": "anime"})
    composition = res.json()["scene_composition"]
    assert set(composition.keys()) == {"shots", "total_duration_seconds", "shot_count", "scene_continuity"}

    continuity = composition["scene_continuity"]
    assert set(continuity.keys()) == {
        "white_fighter", "black_fighter", "arena", "lighting_continuity",
        "cinematic_art_style", "color_palette",
    }
    for fighter_key in ("white_fighter", "black_fighter"):
        assert set(continuity[fighter_key].keys()) == {"hair", "facial_features", "clothing", "armor", "weapon"}
    assert set(continuity["arena"].keys()) == {"layout", "weather", "time_of_day"}


def test_every_shot_in_the_response_carries_the_same_scene_continuity():
    res = client.post("/api/v1/chess2fight/generate", json={"pgn": SCHOLARS_MATE_PGN, "style": "fantasy"})
    composition = res.json()["scene_composition"]
    for shot in composition["shots"]:
        assert shot["scene"] == composition["scene_continuity"]


def test_shot_timeline_still_present_alongside_scene_composition():
    """The Scene Composer is additive — shot_timeline (Sprint 2) must
    still be present and unaffected."""
    res = client.post("/api/v1/chess2fight/generate", json={"pgn": SCHOLARS_MATE_PGN, "style": "anime"})
    data = res.json()
    assert "shot_timeline" in data
    assert "scene_composition" in data
    # Same underlying shots (minus the added `scene` field).
    for original, enriched in zip(data["shot_timeline"]["shots"], data["scene_composition"]["shots"]):
        enriched_without_scene = {k: v for k, v in enriched.items() if k != "scene"}
        assert original == enriched_without_scene


def test_every_pre_existing_response_field_still_present():
    res = client.post("/api/v1/chess2fight/generate", json={"pgn": SCHOLARS_MATE_PGN, "style": "anime"})
    data = res.json()
    for field in (
        "status", "game_analysis", "fight_story", "video_placeholder", "game_metadata",
        "combat_intelligence", "battle_intelligence", "style_profile", "battle_mode_intelligence",
    ):
        assert field in data


def test_determinism_across_repeated_requests():
    responses = [
        client.post("/api/v1/chess2fight/generate", json={"pgn": SCHOLARS_MATE_PGN, "style": "scifi"}).json()["scene_composition"]
        for _ in range(5)
    ]
    assert all(r == responses[0] for r in responses)


def test_army_mode_arena_layout_matches_battle_mode_intelligence_in_the_same_response():
    res = client.post(
        "/api/v1/chess2fight/generate",
        json={"pgn": SCHOLARS_MATE_PGN, "preferences": {"battle_mode": "army"}},
    )
    data = res.json()
    assert data["scene_composition"]["scene_continuity"]["arena"]["layout"] == data["battle_mode_intelligence"]["environment"]


def test_invalid_pgn_still_returns_400():
    res = client.post("/api/v1/chess2fight/generate", json={"pgn": "garbage"})
    assert res.status_code == 400
