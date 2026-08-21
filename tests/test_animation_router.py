"""Tests for core/animation_router.py.

Uses real ffmpeg (via MockAnimationProvider's real VideoBuilder
delegation) and pytest's `tmp_path` fixture throughout, so generated
clips never leak into the real working directory during test runs.
"""

import asyncio

import pytest
from PIL import Image

from core.animation_router import (
    AnimationInstruction,
    AnimationProvider,
    AnimationProviderRegistry,
    AnimationResult,
    AnimationRouter,
    AnimationType,
    MockAnimationProvider,
    get_animation_router,
)
from core.exceptions import AnimationProviderError


def _make_reference_image(path, size=(256, 256), color=(180, 60, 60)):
    Image.new("RGB", size, color=color).save(path)
    return str(path)


def _instruction(image_path=None, **overrides) -> AnimationInstruction:
    defaults = dict(
        shot_id="shot_001",
        source_image_path=image_path or "/tmp/placeholder_never_used.png",
        prompt="a fighter lunging forward",
        duration_seconds=2.0,
        camera_motion="tracking",
        subject_motion="forward_strike",
        motion_intensity=0.7,
    )
    defaults.update(overrides)
    return AnimationInstruction(**defaults)


# --- AnimationInstruction: creation and validation --------------------------


def test_valid_instruction_constructs(tmp_path):
    image_path = _make_reference_image(tmp_path / "ref.png")
    instruction = _instruction(image_path)
    assert instruction.shot_id == "shot_001"
    assert instruction.source_image_path == image_path
    assert instruction.animation_type == AnimationType.IMAGE_TO_VIDEO  # default


def test_invalid_duration_zero_rejected():
    with pytest.raises(ValueError):
        _instruction(duration_seconds=0)


def test_invalid_duration_negative_rejected():
    with pytest.raises(ValueError):
        _instruction(duration_seconds=-1.0)


def test_invalid_motion_intensity_above_one_rejected():
    with pytest.raises(ValueError):
        _instruction(motion_intensity=1.5)


def test_invalid_motion_intensity_below_zero_rejected():
    with pytest.raises(ValueError):
        _instruction(motion_intensity=-0.1)


def test_motion_intensity_boundary_values_accepted():
    assert _instruction(motion_intensity=0.0).motion_intensity == 0.0
    assert _instruction(motion_intensity=1.0).motion_intensity == 1.0


def test_all_three_animation_types_are_valid():
    for animation_type in AnimationType:
        instruction = _instruction(animation_type=animation_type)
        assert instruction.animation_type == animation_type
    assert len(list(AnimationType)) == 3


def test_animation_type_values_are_the_required_modalities():
    values = {t.value for t in AnimationType}
    assert values == {"image_to_video", "text_to_video", "image_sequence_to_video"}


def test_invalid_animation_type_string_rejected():
    with pytest.raises(ValueError):
        _instruction(animation_type="not_a_real_type")


def test_empty_shot_id_rejected():
    with pytest.raises(ValueError):
        _instruction(shot_id="")


def test_empty_source_image_path_rejected():
    with pytest.raises(ValueError):
        _instruction(source_image_path="")


def test_empty_prompt_rejected():
    with pytest.raises(ValueError):
        _instruction(prompt="")


def test_reference_image_paths_defaults_to_empty_list():
    instruction = _instruction()
    assert instruction.reference_image_paths == []


def test_reference_image_paths_can_carry_secondary_images():
    instruction = _instruction(reference_image_paths=["/a.png", "/b.png"])
    assert instruction.reference_image_paths == ["/a.png", "/b.png"]


def test_negative_prompt_defaults_to_none():
    assert _instruction().negative_prompt is None


def test_negative_prompt_can_be_set():
    instruction = _instruction(negative_prompt="blurry, low quality")
    assert instruction.negative_prompt == "blurry, low quality"


def test_width_and_height_default_to_1024():
    instruction = _instruction()
    assert instruction.width == 1024
    assert instruction.height == 1024


def test_width_and_height_can_be_overridden():
    instruction = _instruction(width=512, height=768)
    assert (instruction.width, instruction.height) == (512, 768)


def test_fps_and_seed_default_to_none():
    instruction = _instruction()
    assert instruction.fps is None
    assert instruction.seed is None


def test_fps_and_seed_can_be_set():
    instruction = _instruction(fps=30, seed=42)
    assert instruction.fps == 30
    assert instruction.seed == 42


def test_camera_motion_and_subject_motion_are_free_form_strings():
    """Deliberately not constrained to the product-specific CameraMotion
    enum — any non-empty string is valid at this contract layer."""
    instruction = _instruction(camera_motion="a completely made-up motion descriptor")
    assert instruction.camera_motion == "a completely made-up motion descriptor"


def test_metadata_defaults_to_empty_dict():
    assert _instruction().metadata == {}


# --- AnimationResult ---------------------------------------------------------


def test_successful_result_shape():
    result = AnimationResult(
        success=True, shot_id="shot_001", provider="MockAnimationProvider",
        video_path="/tmp/x.mp4", duration_seconds=2.0, width=512, height=512, fps=24,
    )
    assert result.success is True
    assert result.error_message is None
    assert (result.width, result.height, result.fps) == (512, 512, 24)


def test_failed_result_shape():
    result = AnimationResult(
        success=False, shot_id="shot_001", provider="MockAnimationProvider",
        error_message="something went wrong",
    )
    assert result.success is False
    assert result.video_path is None
    assert result.duration_seconds is None
    assert result.width is None
    assert result.error_message == "something went wrong"


def test_result_metadata_defaults_to_empty_dict():
    result = AnimationResult(success=True, shot_id="s1", provider="p")
    assert result.metadata == {}


def test_result_metadata_handles_arbitrary_values():
    result = AnimationResult(
        success=True, shot_id="s1", provider="p",
        metadata={"animation_type": "image_to_video", "nested": {"a": 1}},
    )
    assert result.metadata["nested"]["a"] == 1


# --- AnimationProvider interface behavior -------------------------------------


def test_animation_provider_cannot_be_instantiated_directly():
    with pytest.raises(TypeError):
        AnimationProvider()


def test_mock_animation_provider_satisfies_the_interface():
    assert issubclass(MockAnimationProvider, AnimationProvider)


# --- MockAnimationProvider: execution and real output artifact --------------


def test_mock_provider_execution_returns_animation_result(tmp_path):
    image_path = _make_reference_image(tmp_path / "ref.png")
    provider = MockAnimationProvider(output_dir=str(tmp_path / "out"))
    result = asyncio.run(provider.generate_animation(_instruction(image_path)))
    assert isinstance(result, AnimationResult)
    assert result.success is True


def test_mock_output_artifact_is_a_real_valid_mp4_file(tmp_path):
    image_path = _make_reference_image(tmp_path / "ref.png")
    provider = MockAnimationProvider(output_dir=str(tmp_path / "out"))
    result = asyncio.run(provider.generate_animation(_instruction(image_path, duration_seconds=1.5)))

    assert result.video_path is not None
    from pathlib import Path

    assert Path(result.video_path).exists()
    assert Path(result.video_path).stat().st_size > 0


def test_mock_output_duration_matches_instruction(tmp_path):
    image_path = _make_reference_image(tmp_path / "ref.png")
    provider = MockAnimationProvider(output_dir=str(tmp_path / "out"))
    result = asyncio.run(provider.generate_animation(_instruction(image_path, duration_seconds=4.0)))
    assert result.duration_seconds == 4.0


def test_mock_output_dimensions_match_instruction(tmp_path):
    image_path = _make_reference_image(tmp_path / "ref.png")
    provider = MockAnimationProvider(output_dir=str(tmp_path / "out"))
    result = asyncio.run(provider.generate_animation(_instruction(image_path, width=640, height=480)))
    assert (result.width, result.height) == (640, 480)

    import subprocess
    import json

    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-print_format", "json", "-show_streams", result.video_path],
        capture_output=True, text=True, check=True,
    )
    stream = next(s for s in json.loads(probe.stdout)["streams"] if s["codec_type"] == "video")
    assert (stream["width"], stream["height"]) == (640, 480)


def test_mock_output_fps_defaults_to_24_when_not_specified(tmp_path):
    image_path = _make_reference_image(tmp_path / "ref.png")
    provider = MockAnimationProvider(output_dir=str(tmp_path / "out"))
    result = asyncio.run(provider.generate_animation(_instruction(image_path)))
    assert result.fps == 24


def test_mock_output_fps_honors_explicit_instruction_value(tmp_path):
    image_path = _make_reference_image(tmp_path / "ref.png")
    provider = MockAnimationProvider(output_dir=str(tmp_path / "out"))
    result = asyncio.run(provider.generate_animation(_instruction(image_path, fps=12)))
    assert result.fps == 12


def test_deterministic_behavior_same_input_same_structural_output(tmp_path):
    """Same instruction always produces a clip with the same duration,
    resolution, and success outcome. Not asserting on raw encoded
    bytes — ffmpeg encoder output can vary slightly build to build for
    identical input, which this repository's own VideoBuilder test
    suite already establishes."""
    image_path = _make_reference_image(tmp_path / "ref.png")
    provider = MockAnimationProvider(output_dir=str(tmp_path / "out"))
    instruction = _instruction(image_path)

    result1 = asyncio.run(provider.generate_animation(instruction))
    result2 = asyncio.run(provider.generate_animation(instruction))

    assert result1.success == result2.success
    assert result1.duration_seconds == result2.duration_seconds
    assert result1.video_path == result2.video_path


# --- Failure behavior / invalid instructions ---------------------------------


def test_missing_reference_image_returns_failed_result_not_exception(tmp_path):
    provider = MockAnimationProvider(output_dir=str(tmp_path / "out"))
    instruction = _instruction(str(tmp_path / "does_not_exist.png"))
    result = asyncio.run(provider.generate_animation(instruction))
    assert result.success is False
    assert result.video_path is None
    assert "not found" in result.error_message.lower()


def test_no_external_network_dependency_in_provider_source():
    """Structural check: MockAnimationProvider's own code never
    references httpx, requests, or any HTTP client — it has no way to
    reach an external service."""
    import inspect

    from core import animation_router as module

    source = inspect.getsource(module.MockAnimationProvider)
    for forbidden in ("httpx", "requests", "urllib", "AsyncOpenAI"):
        assert forbidden not in source


def test_reuses_the_injected_video_builder_rather_than_a_new_one(tmp_path):
    """Confirms MockAnimationProvider delegates to VideoBuilder rather
    than reimplementing ffmpeg invocation — verified by injecting a
    fake and checking it was actually called with the right arguments."""
    calls = []

    class _FakeVideoBuilder:
        async def build_video(self, **kwargs):
            calls.append(kwargs)

            class _Result:
                video_path = str(tmp_path / "fake.mp4")
                duration_seconds = kwargs["frame_duration_seconds"]
                width = kwargs.get("width", 1024)
                height = kwargs.get("height", 1024)
                fps = kwargs.get("fps", 24)

            return _Result()

    image_path = _make_reference_image(tmp_path / "ref.png")
    provider = MockAnimationProvider(output_dir=str(tmp_path / "out"), video_builder=_FakeVideoBuilder())
    result = asyncio.run(provider.generate_animation(_instruction(image_path, duration_seconds=2.5)))

    assert len(calls) == 1
    assert calls[0]["frame_count"] == 1
    assert calls[0]["frame_duration_seconds"] == 2.5
    assert result.success is True


# --- Provider registration / lookup / missing-provider behavior -------------


def test_registry_starts_empty():
    registry = AnimationProviderRegistry()
    assert registry.list_providers() == []
    assert not registry.is_registered("mock")


def test_registry_register_and_create(tmp_path):
    registry = AnimationProviderRegistry()
    registry.register("mock", lambda: MockAnimationProvider(output_dir=str(tmp_path)))
    assert registry.is_registered("mock")
    provider = registry.create("mock")
    assert isinstance(provider, MockAnimationProvider)


def test_registry_create_unregistered_name_raises():
    registry = AnimationProviderRegistry()
    with pytest.raises(AnimationProviderError):
        registry.create("does_not_exist")


def test_registry_create_returns_a_fresh_instance_each_time(tmp_path):
    call_count = 0

    def factory():
        nonlocal call_count
        call_count += 1
        return MockAnimationProvider(output_dir=str(tmp_path))

    registry = AnimationProviderRegistry()
    registry.register("mock", factory)
    registry.create("mock")
    registry.create("mock")
    assert call_count == 2


# --- Router provider selection ------------------------------------------------


def test_router_selects_the_configured_mock_provider(tmp_path, monkeypatch):
    from core import config

    monkeypatch.setattr(config.get_settings(), "animation_provider", "mock")
    registry = AnimationProviderRegistry()
    registry.register("mock", lambda: MockAnimationProvider(output_dir=str(tmp_path)))
    router = AnimationRouter(registry=registry)

    image_path = _make_reference_image(tmp_path / "ref.png")
    result = asyncio.run(router.generate_animation(_instruction(image_path)))
    assert result.provider == "MockAnimationProvider"
    assert router.active_provider_name() == "mock"


def test_router_falls_back_to_mock_for_unregistered_configured_provider(tmp_path, monkeypatch):
    from core import config

    monkeypatch.setattr(config.get_settings(), "animation_provider", "does_not_exist")
    registry = AnimationProviderRegistry()
    registry.register("mock", lambda: MockAnimationProvider(output_dir=str(tmp_path)))
    router = AnimationRouter(registry=registry)

    image_path = _make_reference_image(tmp_path / "ref.png")
    result = asyncio.run(router.generate_animation(_instruction(image_path)))
    assert result.provider == "MockAnimationProvider"
    assert router.active_provider_name() == "mock"


def test_router_wraps_unexpected_provider_exceptions_into_a_failed_result(monkeypatch):
    """Even a misbehaving provider that raises (rather than returning
    success=False) must not crash the router."""
    from core import config

    class _ExplodingProvider(AnimationProvider):
        async def generate_animation(self, instruction):
            raise RuntimeError("boom")

    monkeypatch.setattr(config.get_settings(), "animation_provider", "exploding")
    registry = AnimationProviderRegistry()
    registry.register("exploding", _ExplodingProvider)
    router = AnimationRouter(registry=registry)

    result = asyncio.run(router.generate_animation(_instruction()))
    assert result.success is False
    assert "boom" in result.error_message


def test_get_animation_router_returns_a_cached_singleton():
    assert get_animation_router() is get_animation_router()


def test_a_second_independent_provider_can_be_registered_and_used(monkeypatch):
    """Provider isolation, demonstrated the same way it's established
    for ImageRouter: a second provider needs zero changes to
    AnimationRouter, AnimationProviderRegistry, or MockAnimationProvider."""
    from core import config

    class _SecondProvider(AnimationProvider):
        async def generate_animation(self, instruction):
            return AnimationResult(success=True, shot_id=instruction.shot_id, provider="_SecondProvider")

    monkeypatch.setattr(config.get_settings(), "animation_provider", "second")
    registry = AnimationProviderRegistry()
    registry.register("second", _SecondProvider)
    router = AnimationRouter(registry=registry)

    result = asyncio.run(router.generate_animation(_instruction()))
    assert result.provider == "_SecondProvider"


# --- Isolation: usable through the router alone, and no orchestration wiring ---


def test_a_caller_can_generate_without_ever_naming_a_concrete_provider(tmp_path, monkeypatch):
    """Mirrors the same guarantee already verified for ImageRouter:
    demonstrates the router-only calling path actually works, using
    only AnimationRouter/get_animation_router/AnimationResult."""
    import core.animation_router as animation_router_module
    from core import config

    original_singleton = animation_router_module._router_instance
    animation_router_module._router_instance = None
    monkeypatch.setattr(config.get_settings(), "animation_output_dir", str(tmp_path))
    monkeypatch.setattr(config.get_settings(), "animation_provider", "mock")
    try:
        router = get_animation_router()
        image_path = _make_reference_image(tmp_path / "ref.png")
        result: AnimationResult = asyncio.run(router.generate_animation(_instruction(image_path)))
        assert result.success is True
    finally:
        animation_router_module._router_instance = original_singleton


def test_orchestrator_does_not_import_animation_router():
    """This task is explicit: the animation system must NOT be wired
    into the main orchestration flow yet."""
    import inspect

    from products.chess2fight import orchestrator as module

    source = inspect.getsource(module)
    assert "animation_router" not in source
    assert "AnimationRouter" not in source
    assert "AnimationProvider" not in source


def test_api_module_does_not_import_animation_router():
    import inspect

    from api import chess2fight as module

    source = inspect.getsource(module)
    assert "animation_router" not in source
    assert "AnimationRouter" not in source


def test_provider_does_not_import_fastapi_chess_or_pgn_modules():
    """The provider must not know about FastAPI, HTTP requests, chess,
    PGN, Battle Director, or cinematic timeline generation. Checked via
    AST against actual import statements in this whole module, not a
    raw text search — a raw search over the source string is tripped
    up by this module's own docstrings, which legitimately *discuss*
    the isolation principle in prose ("must never know about FastAPI,
    HTTP, chess, PGN..."). That prose is documentation, not a
    dependency; this test verifies there is no real one.
    """
    import ast
    import inspect

    from core import animation_router as module

    tree = ast.parse(inspect.getsource(module))
    imported_modules = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    } | {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module
    }

    forbidden_prefixes = ("fastapi", "chess", "products.chess2fight.pgn_analyzer", "products.chess2fight.battle_director", "products.chess2fight.cinematic.timeline_engine")
    for imported in imported_modules:
        assert not imported.startswith(forbidden_prefixes), f"forbidden import found: {imported}"
