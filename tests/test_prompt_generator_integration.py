"""Integration tests: the Prompt Generator's output as it actually
appears in the real HTTP API response."""

from fastapi.testclient import TestClient

from main import app

client = TestClient(app)

SCHOLARS_MATE_PGN = (
    '[Event "Example"]\n[White "Halisako"]\n[Black "Guest"]\n[Result "1-0"]\n\n'
    "1. e4 e5 2. Qh5 Nc6 3. Bc4 Nf6 4. Qxf7# 1-0"
)


def test_generate_response_includes_prompted_timeline():
    res = client.post("/api/v1/chess2fight/generate", json={"pgn": SCHOLARS_MATE_PGN, "style": "anime"})
    assert res.status_code == 200
    assert "prompted_timeline" in res.json()


def test_prompted_timeline_has_correct_shape():
    res = client.post("/api/v1/chess2fight/generate", json={"pgn": SCHOLARS_MATE_PGN, "style": "anime"})
    prompted = res.json()["prompted_timeline"]
    assert set(prompted.keys()) == {"shots", "total_duration_seconds", "shot_count", "scene_continuity"}
    for shot in prompted["shots"]:
        assert "image_prompt" in shot
        assert isinstance(shot["image_prompt"], str)
        assert len(shot["image_prompt"]) > 0


def test_every_shot_has_a_non_empty_unique_prompt():
    """Each shot's prompt should differ from the others (different
    camera, action, mood per shot), not be a copy-pasted template."""
    res = client.post("/api/v1/chess2fight/generate", json={"pgn": SCHOLARS_MATE_PGN, "style": "fantasy"})
    shots = res.json()["prompted_timeline"]["shots"]
    prompts = [s["image_prompt"] for s in shots]
    assert len(set(prompts)) == len(prompts)


def test_shot_timeline_and_scene_composition_still_present_alongside_prompted_timeline():
    """The Prompt Generator is additive — earlier Sprint 3 fields must
    still be present and unaffected."""
    res = client.post("/api/v1/chess2fight/generate", json={"pgn": SCHOLARS_MATE_PGN, "style": "anime"})
    data = res.json()
    assert "shot_timeline" in data
    assert "scene_composition" in data
    assert "prompted_timeline" in data
    # Same underlying shots (minus the added `image_prompt` field).
    for enriched, prompted_shot in zip(data["scene_composition"]["shots"], data["prompted_timeline"]["shots"]):
        prompted_without_prompt = {k: v for k, v in prompted_shot.items() if k != "image_prompt"}
        assert enriched == prompted_without_prompt


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
        client.post("/api/v1/chess2fight/generate", json={"pgn": SCHOLARS_MATE_PGN, "style": "scifi"}).json()["prompted_timeline"]
        for _ in range(5)
    ]
    assert all(r == responses[0] for r in responses)


def test_style_selection_changes_the_generated_prompts():
    anime_res = client.post("/api/v1/chess2fight/generate", json={"pgn": SCHOLARS_MATE_PGN, "style": "anime"}).json()
    scifi_res = client.post("/api/v1/chess2fight/generate", json={"pgn": SCHOLARS_MATE_PGN, "style": "scifi"}).json()
    anime_prompt = anime_res["prompted_timeline"]["shots"][0]["image_prompt"]
    scifi_prompt = scifi_res["prompted_timeline"]["shots"][0]["image_prompt"]
    assert anime_prompt != scifi_prompt
    assert "anime" not in scifi_prompt.lower()


def test_invalid_pgn_still_returns_400():
    res = client.post("/api/v1/chess2fight/generate", json={"pgn": "garbage"})
    assert res.status_code == 400
