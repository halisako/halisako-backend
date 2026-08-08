"""Tests for core/image_router.py.

Uses pytest's `tmp_path` fixture throughout so generated placeholder
images never leak into the real working directory during test runs.
"""

import asyncio

import pytest
from PIL import Image

from core.exceptions import ImageProviderError
from core.image_router import (
    ImageGenerationResult,
    ImageProvider,
    ImageProviderRegistry,
    ImageRouter,
    MockImageProvider,
    get_image_router,
)


# --- ImageProvider is a real interface, not instantiable directly ----------


def test_image_provider_cannot_be_instantiated_directly():
    with pytest.raises(TypeError):
        ImageProvider()


def test_mock_image_provider_satisfies_the_interface():
    assert issubclass(MockImageProvider, ImageProvider)


# --- MockImageProvider generates real, valid placeholder files -------------


def test_generates_a_real_valid_png_file(tmp_path):
    provider = MockImageProvider(output_dir=str(tmp_path))
    result = asyncio.run(provider.generate_image("a warrior in golden armor"))

    assert (tmp_path / result_filename(result)).exists()
    image = Image.open(result.image_path)
    assert image.format == "PNG"
    assert image.size == (result.width, result.height)


def result_filename(result: ImageGenerationResult) -> str:
    from pathlib import Path

    return Path(result.image_path).name


def test_respects_requested_width_and_height(tmp_path):
    provider = MockImageProvider(output_dir=str(tmp_path))
    result = asyncio.run(provider.generate_image("test prompt", width=512, height=256))
    assert result.width == 512
    assert result.height == 256
    image = Image.open(result.image_path)
    assert image.size == (512, 256)


def test_result_has_correct_provider_name_and_prompt(tmp_path):
    provider = MockImageProvider(output_dir=str(tmp_path))
    result = asyncio.run(provider.generate_image("a specific test prompt"))
    assert result.provider == "MockImageProvider"
    assert result.prompt == "a specific test prompt"


def test_generation_time_is_a_real_non_negative_measurement(tmp_path):
    provider = MockImageProvider(output_dir=str(tmp_path))
    result = asyncio.run(provider.generate_image("timing test"))
    assert result.generation_time_seconds >= 0


# --- Determinism: same prompt -> identical file, byte for byte -------------


def test_same_prompt_produces_identical_path_and_bytes(tmp_path):
    provider = MockImageProvider(output_dir=str(tmp_path))
    r1 = asyncio.run(provider.generate_image("a warrior in golden armor"))
    r2 = asyncio.run(provider.generate_image("a warrior in golden armor"))
    assert r1.image_path == r2.image_path
    assert open(r1.image_path, "rb").read() == open(r2.image_path, "rb").read()


def test_different_prompts_produce_different_paths(tmp_path):
    provider = MockImageProvider(output_dir=str(tmp_path))
    r1 = asyncio.run(provider.generate_image("a warrior in golden armor"))
    r2 = asyncio.run(provider.generate_image("a mage in blue robes"))
    assert r1.image_path != r2.image_path


def test_output_directory_is_created_if_missing(tmp_path):
    nested = tmp_path / "does" / "not" / "exist" / "yet"
    assert not nested.exists()
    MockImageProvider(output_dir=str(nested))
    assert nested.exists()


# --- ImageProviderRegistry ---------------------------------------------------


def test_registry_starts_empty():
    registry = ImageProviderRegistry()
    assert registry.list_providers() == []
    assert not registry.is_registered("mock")


def test_registry_register_and_create(tmp_path):
    registry = ImageProviderRegistry()
    registry.register("mock", lambda: MockImageProvider(output_dir=str(tmp_path)))
    assert registry.is_registered("mock")
    provider = registry.create("mock")
    assert isinstance(provider, MockImageProvider)


def test_registry_create_unregistered_name_raises():
    registry = ImageProviderRegistry()
    with pytest.raises(ImageProviderError):
        registry.create("does_not_exist")


def test_registry_list_providers_reflects_registrations(tmp_path):
    registry = ImageProviderRegistry()
    registry.register("mock", lambda: MockImageProvider(output_dir=str(tmp_path)))
    registry.register("another", lambda: MockImageProvider(output_dir=str(tmp_path)))
    assert set(registry.list_providers()) == {"mock", "another"}


def test_registry_create_returns_a_fresh_instance_each_time(tmp_path):
    """Factories, not pre-built instances — registering must not
    construct a provider until actually requested."""
    call_count = 0

    def factory():
        nonlocal call_count
        call_count += 1
        return MockImageProvider(output_dir=str(tmp_path))

    registry = ImageProviderRegistry()
    registry.register("mock", factory)
    assert call_count == 0
    registry.create("mock")
    assert call_count == 1
    registry.create("mock")
    assert call_count == 2


# --- Provider isolation: a new provider needs zero changes elsewhere --------


class _FakeProvider(ImageProvider):
    """A minimal second provider, used only to prove
    ImageProviderRegistry/ImageRouter work with any ImageProvider —
    written without importing or modifying MockImageProvider at all."""

    async def generate_image(self, prompt: str, width: int = 1024, height: int = 1024) -> ImageGenerationResult:
        return ImageGenerationResult(
            image_path="/fake/path.png", provider="_FakeProvider", prompt=prompt,
            width=width, height=height, generation_time_seconds=0.0,
        )


def test_a_second_independent_provider_can_be_registered_and_used():
    registry = ImageProviderRegistry()
    registry.register("fake", _FakeProvider)
    router = ImageRouter(registry=registry)

    from core import config

    original = config.settings.image_provider
    config.settings.image_provider = "fake"
    try:
        result = asyncio.run(router.generate_image("test"))
        assert result.provider == "_FakeProvider"
    finally:
        config.settings.image_provider = original


# --- ImageRouter: resolution, fallback, active_provider_name ----------------


def test_router_resolves_the_configured_provider(tmp_path):
    from core import config

    registry = ImageProviderRegistry()
    registry.register("mock", lambda: MockImageProvider(output_dir=str(tmp_path)))
    router = ImageRouter(registry=registry)

    original = config.settings.image_provider
    config.settings.image_provider = "mock"
    try:
        result = asyncio.run(router.generate_image("test"))
        assert result.provider == "MockImageProvider"
        assert router.active_provider_name() == "mock"
    finally:
        config.settings.image_provider = original


def test_router_falls_back_to_mock_for_an_unregistered_configured_provider(tmp_path):
    from core import config

    registry = ImageProviderRegistry()
    registry.register("mock", lambda: MockImageProvider(output_dir=str(tmp_path)))
    router = ImageRouter(registry=registry)

    original = config.settings.image_provider
    config.settings.image_provider = "does_not_exist"
    try:
        result = asyncio.run(router.generate_image("fallback test"))
        assert result.provider == "MockImageProvider"
        assert router.active_provider_name() == "mock"
    finally:
        config.settings.image_provider = original


def test_router_only_resolves_the_provider_once(tmp_path):
    """Confirms the resolved provider is cached rather than
    re-resolved (and re-constructed) on every call."""
    from core import config

    construction_count = 0

    def counting_factory():
        nonlocal construction_count
        construction_count += 1
        return MockImageProvider(output_dir=str(tmp_path))

    registry = ImageProviderRegistry()
    registry.register("mock", counting_factory)
    router = ImageRouter(registry=registry)

    original = config.settings.image_provider
    config.settings.image_provider = "mock"
    try:
        asyncio.run(router.generate_image("first"))
        asyncio.run(router.generate_image("second"))
        assert construction_count == 1
    finally:
        config.settings.image_provider = original


def test_get_image_router_returns_a_cached_singleton():
    assert get_image_router() is get_image_router()


# --- The core architectural guarantee: usable through the router alone -----


def test_a_caller_can_generate_a_real_image_without_ever_naming_a_concrete_provider(tmp_path, monkeypatch):
    """This test deliberately never references MockImageProvider (or
    any concrete provider) by name in the calling code — only
    ImageRouter / get_image_router / ImageGenerationResult are used,
    demonstrating the isolation the brief requires: a caller (a future
    "Director") only ever needs the router-level API to get a working
    result.

    Saves and restores the module-level cached singleton around the
    test: `get_image_router()` caches a real, shared ImageRouter for
    the whole process, and this test needs to force it to re-resolve
    against a temp directory — without a save/restore, that mutation
    would leak into any other test that calls `get_image_router()`
    afterward.
    """
    import core.image_router as image_router_module
    from core import config

    original_singleton = image_router_module._router_instance
    image_router_module._router_instance = None
    monkeypatch.setattr(config.settings, "image_output_dir", str(tmp_path))
    try:
        router = get_image_router()
        result: ImageGenerationResult = asyncio.run(
            router.generate_image("a caller that never names a concrete provider")
        )
        assert result.image_path
        image = Image.open(result.image_path)
        assert image.format == "PNG"
    finally:
        image_router_module._router_instance = original_singleton


# --- ImageGenerationResult schema -----------------------------------------------


def test_image_generation_result_requires_positive_dimensions():
    with pytest.raises(ValueError):
        ImageGenerationResult(
            image_path="/x.png", provider="p", prompt="x",
            width=0, height=100, generation_time_seconds=0.0,
        )


def test_image_generation_result_metadata_defaults_to_empty_dict():
    result = ImageGenerationResult(
        image_path="/x.png", provider="p", prompt="x",
        width=10, height=10, generation_time_seconds=0.0,
    )
    assert result.metadata == {}
