"""Tests for RenderPipeline.

Uses the real pipeline through the Prompt Generator wherever possible
— RenderPipeline's whole job is turning real PromptedTimeline output
into files on disk, so testing against a hand-built fixture would
prove much less than testing against what the rest of the pipeline
actually produces.
"""

import asyncio
import json

from core.ai_router import TemplateProvider
from core.image_router import ImageProviderRegistry, ImageRouter, MockImageProvider
from products.chess2fight.battle_director import generate_battle_intelligence
from products.chess2fight.battle_mode_engine import generate_battle_mode_intelligence
from products.chess2fight.cinematic.prompt_generator import generate_prompts
from products.chess2fight.cinematic.scene_composer import compose_scene
from products.chess2fight.cinematic.timeline_engine import generate_shot_timeline
from products.chess2fight.combat_mapper import generate_combat_intelligence
from products.chess2fight.narrative_generator import NarrativeGenerator
from products.chess2fight.pgn_analyzer import analyze_game
from products.chess2fight.rendering.asset_manager import AssetManager
from products.chess2fight.rendering.render_pipeline import RenderPipeline, _derive_seed
from products.chess2fight.schemas import BattleMode
from products.chess2fight.style_engine import generate_style_profile

SCHOLARS_MATE = (
    '[Event "Example"]\n[White "Halisako"]\n[Black "Guest"]\n[Result "1-0"]\n\n'
    "1. e4 e5 2. Qh5 Nc6 3. Bc4 Nf6 4. Qxf7# 1-0"
)
LEGALL_TRAP = (
    '[Result "1-0"]\n\n'
    "1. e4 e5 2. Nf3 d6 3. Bc4 Bg4 4. Nc3 g6 5. Nxe5 Bxd1 6. Bxf7+ Ke7 7. Nd5# 1-0"
)


async def _build_prompted(pgn: str, style: str, mode: BattleMode = BattleMode.DUEL):
    analysis = analyze_game(pgn)
    combat = generate_combat_intelligence(analysis)
    battle = generate_battle_intelligence(analysis, combat)
    style_profile = generate_style_profile(battle, combat, style)
    battle_mode = generate_battle_mode_intelligence(mode, combat, battle)
    story = await NarrativeGenerator(TemplateProvider()).generate(
        analysis, combat, battle, style_profile, battle_mode
    )
    timeline = generate_shot_timeline(battle, story)
    composed = compose_scene(timeline, battle, style_profile, battle_mode)
    return generate_prompts(composed)


def _pipeline(image_dir, storage_dir) -> RenderPipeline:
    registry = ImageProviderRegistry()
    registry.register("mock", lambda: MockImageProvider(output_dir=str(image_dir)))
    router = ImageRouter(registry=registry)
    from core import config

    config.settings.image_provider = "mock"
    asset_manager = AssetManager(storage_root=str(storage_dir))
    return RenderPipeline(image_router=router, asset_manager=asset_manager)


# --- Core rendering behavior -------------------------------------------------


def test_renders_exactly_one_frame_per_shot(tmp_path):
    prompted = asyncio.run(_build_prompted(SCHOLARS_MATE, "anime"))
    pipeline = _pipeline(tmp_path / "images", tmp_path / "storage")
    output = asyncio.run(pipeline.render(prompted, fight_id="fight_1"))
    assert output.frame_count == prompted.shot_count
    assert len(output.frames) == prompted.shot_count


def test_frame_numbers_match_shot_sequence_order_exactly(tmp_path):
    prompted = asyncio.run(_build_prompted(LEGALL_TRAP, "fantasy"))
    pipeline = _pipeline(tmp_path / "images", tmp_path / "storage")
    output = asyncio.run(pipeline.render(prompted, fight_id="fight_1"))
    for shot, frame in zip(prompted.shots, output.frames):
        assert frame.frame_number == shot.sequence_order


def test_frame_files_actually_exist_on_disk_as_valid_images(tmp_path):
    from PIL import Image

    prompted = asyncio.run(_build_prompted(SCHOLARS_MATE, "anime"))
    pipeline = _pipeline(tmp_path / "images", tmp_path / "storage")
    output = asyncio.run(pipeline.render(prompted, fight_id="fight_1"))

    for frame in output.frames:
        image = Image.open(frame.frame_path)
        assert image.format == "PNG"


def test_no_video_files_are_ever_produced(tmp_path):
    """Hard requirement: only frames, never a video."""
    prompted = asyncio.run(_build_prompted(SCHOLARS_MATE, "anime"))
    pipeline = _pipeline(tmp_path / "images", tmp_path / "storage")
    output = asyncio.run(pipeline.render(prompted, fight_id="fight_1"))

    from pathlib import Path

    video_extensions = (".mp4", ".mov", ".avi", ".webm", ".mkv", ".gif")
    for entry in Path(output.output_dir).iterdir():
        assert entry.suffix.lower() not in video_extensions


# --- Metadata completeness --------------------------------------------------------


def test_every_required_metadata_field_present_for_every_frame(tmp_path):
    prompted = asyncio.run(_build_prompted(LEGALL_TRAP, "scifi"))
    pipeline = _pipeline(tmp_path / "images", tmp_path / "storage")
    output = asyncio.run(pipeline.render(prompted, fight_id="fight_1"))

    for shot, frame in zip(prompted.shots, output.frames):
        meta = frame.metadata
        assert meta.frame_number == shot.sequence_order
        assert meta.prompt == shot.image_prompt
        assert meta.camera_angle == shot.camera_angle.value
        assert meta.camera_motion == shot.camera_motion.value
        assert meta.shot_id == shot.shot_id
        assert meta.shot_type == shot.shot_type.value
        assert meta.source_moves == shot.source_moves
        assert meta.timestamp
        assert isinstance(meta.generation_seed, int)


def test_manifest_written_to_disk_matches_the_returned_output(tmp_path):
    prompted = asyncio.run(_build_prompted(SCHOLARS_MATE, "anime"))
    pipeline = _pipeline(tmp_path / "images", tmp_path / "storage")
    output = asyncio.run(pipeline.render(prompted, fight_id="fight_1"))

    with open(output.manifest_path) as f:
        manifest = json.load(f)
    assert manifest["fight_id"] == "fight_1"
    assert manifest["frame_count"] == output.frame_count
    assert len(manifest["frames"]) == output.frame_count
    assert [f["frame_number"] for f in manifest["frames"]] == [
        frame.frame_number for frame in output.frames
    ]


def test_asset_manager_can_read_back_the_manifest_the_pipeline_wrote(tmp_path):
    prompted = asyncio.run(_build_prompted(SCHOLARS_MATE, "anime"))
    asset_manager = AssetManager(storage_root=str(tmp_path / "storage"))
    registry = ImageProviderRegistry()
    registry.register("mock", lambda: MockImageProvider(output_dir=str(tmp_path / "images")))
    from core import config

    config.settings.image_provider = "mock"
    pipeline = RenderPipeline(image_router=ImageRouter(registry=registry), asset_manager=asset_manager)

    asyncio.run(pipeline.render(prompted, fight_id="fight_readback"))
    reloaded = asset_manager.read_manifest("fight_readback")
    assert reloaded.fight_id == "fight_readback"
    assert reloaded.frame_count == prompted.shot_count


# --- Determinism (of what's expected to be deterministic) ------------------------


def test_rendering_the_same_timeline_twice_produces_identical_files(tmp_path):
    prompted = asyncio.run(_build_prompted(SCHOLARS_MATE, "anime"))
    pipeline = _pipeline(tmp_path / "images", tmp_path / "storage")

    output1 = asyncio.run(pipeline.render(prompted, fight_id="repeat_test"))
    output2 = asyncio.run(pipeline.render(prompted, fight_id="repeat_test"))

    for frame1, frame2 in zip(output1.frames, output2.frames):
        assert frame1.metadata.generation_seed == frame2.metadata.generation_seed
        assert open(frame1.frame_path, "rb").read() == open(frame2.frame_path, "rb").read()


def test_generation_seed_is_deterministic_for_the_same_prompt():
    assert _derive_seed("some prompt text") == _derive_seed("some prompt text")


def test_generation_seed_differs_for_different_prompts():
    assert _derive_seed("prompt A") != _derive_seed("prompt B")


# --- Fight isolation --------------------------------------------------------------


def test_different_fight_ids_produce_separate_storage_directories(tmp_path):
    prompted = asyncio.run(_build_prompted(SCHOLARS_MATE, "anime"))
    pipeline = _pipeline(tmp_path / "images", tmp_path / "storage")

    output_a = asyncio.run(pipeline.render(prompted, fight_id="fight_a"))
    output_b = asyncio.run(pipeline.render(prompted, fight_id="fight_b"))

    assert output_a.output_dir != output_b.output_dir
    assert output_a.fight_id == "fight_a"
    assert output_b.fight_id == "fight_b"


# --- Different styles / modes render through correctly ----------------------------


def test_different_styles_produce_different_prompts_in_the_metadata(tmp_path):
    anime_prompted = asyncio.run(_build_prompted(SCHOLARS_MATE, "anime"))
    fantasy_prompted = asyncio.run(_build_prompted(SCHOLARS_MATE, "fantasy"))
    pipeline = _pipeline(tmp_path / "images", tmp_path / "storage")

    anime_output = asyncio.run(pipeline.render(anime_prompted, fight_id="anime_fight"))
    fantasy_output = asyncio.run(pipeline.render(fantasy_prompted, fight_id="fantasy_fight"))

    assert anime_output.frames[0].metadata.prompt != fantasy_output.frames[0].metadata.prompt


def test_army_mode_renders_correctly(tmp_path):
    prompted = asyncio.run(_build_prompted(SCHOLARS_MATE, "anime", BattleMode.ARMY))
    pipeline = _pipeline(tmp_path / "images", tmp_path / "storage")
    output = asyncio.run(pipeline.render(prompted, fight_id="army_fight"))
    assert output.frame_count == prompted.shot_count


def test_draw_game_renders_correctly(tmp_path):
    draw_pgn = '[Result "1/2-1/2"]\n\n1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 1/2-1/2'
    prompted = asyncio.run(_build_prompted(draw_pgn, "fantasy"))
    pipeline = _pipeline(tmp_path / "images", tmp_path / "storage")
    output = asyncio.run(pipeline.render(prompted, fight_id="draw_fight"))
    assert output.frame_count == prompted.shot_count


# --- Injection / isolation ---------------------------------------------------------


def test_render_pipeline_never_imports_a_concrete_image_provider_by_name():
    """Structural check on this module's own source, not just its
    behavior: RenderPipeline must depend only on ImageRouter, never
    reference MockImageProvider (or any other concrete provider)."""
    import inspect

    from products.chess2fight.rendering import render_pipeline as module

    source = inspect.getsource(module)
    assert "MockImageProvider" not in source
    assert "ImageProviderRegistry" not in source
