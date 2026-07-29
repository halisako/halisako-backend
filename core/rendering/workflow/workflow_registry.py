"""Registry of renderer targets the Workflow Builder can produce
workflows for.

This is deliberately not a registry of *adapters* — it holds no
renderer-specific serialization logic and imports nothing
renderer-specific (no ComfyUI, no anything). Being "registered" here
means only that `WorkflowBuilder.build()` will accept a given name as
a valid `renderer` argument. Turning the resulting generic
`RendererWorkflow` into a specific renderer's actual input format is
an adapter layer's job, downstream of this subpackage, per the
architecture's explicit separation between the Workflow Builder and
renderer adapters.

This registry exists so that adding a new renderer target is a
one-line `register()` call, not a change to `WorkflowBuilder` itself —
see section 17 of the specification ("Future Compatibility").
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from core.rendering.workflow.exceptions import UnsupportedRendererError


class RendererTarget(BaseModel):
    """Metadata about one renderer target known to the platform."""

    name: str = Field(..., min_length=1, description="Stable identifier, e.g. \"comfyui\".")
    display_name: str = Field(..., min_length=1, description="Human-readable name for logs/UI.")
    supports_video: bool = Field(default=True, description="Whether this target can produce video output.")
    supports_audio_nodes: bool = Field(
        default=True, description="Whether this target's adapter understands audio-intent nodes."
    )
    notes: str = Field(default="", description="Free-text notes, e.g. current integration status.")


class WorkflowRegistry:
    """Tracks which renderer targets are known to the platform.

    Pre-populated with the targets named in the specification
    (ComfyUI, RunPod, Modal, Replicate, Unreal Engine, Unity, and a
    generic "custom" slot) so `WorkflowBuilder.build()` can validate
    its `renderer` argument without any renderer-specific import.
    """

    def __init__(self) -> None:
        self._targets: dict[str, RendererTarget] = {}
        self._register_defaults()

    def register(self, target: RendererTarget) -> None:
        """Registers (or replaces) a renderer target."""
        self._targets[target.name] = target

    def is_registered(self, name: str) -> bool:
        """Returns True if `name` is a known renderer target."""
        return name in self._targets

    def get(self, name: str) -> RendererTarget:
        """Returns the RendererTarget for `name`.

        Raises:
            UnsupportedRendererError: If `name` is not registered.
        """
        target = self._targets.get(name)
        if target is None:
            raise UnsupportedRendererError(
                f"Renderer target {name!r} is not registered. "
                f"Known targets: {sorted(self._targets)}."
            )
        return target

    def list_targets(self) -> list[RendererTarget]:
        """Returns every currently registered renderer target."""
        return list(self._targets.values())

    def _register_defaults(self) -> None:
        """Registers the renderer targets named in the specification.
        None of these imply a working adapter exists yet — see the
        module docstring."""
        defaults = [
            RendererTarget(
                name="comfyui", display_name="ComfyUI",
                notes="First target renderer; adapter serializes this subpackage's generic graph "
                "into ComfyUI's node format.",
            ),
            RendererTarget(name="runpod", display_name="RunPod", notes="Adapter not yet implemented."),
            RendererTarget(name="modal", display_name="Modal", notes="Adapter not yet implemented."),
            RendererTarget(name="replicate", display_name="Replicate", notes="Adapter not yet implemented."),
            RendererTarget(
                name="unreal", display_name="Unreal Engine",
                supports_audio_nodes=False, notes="Adapter not yet implemented.",
            ),
            RendererTarget(
                name="unity", display_name="Unity",
                supports_audio_nodes=False, notes="Adapter not yet implemented.",
            ),
            RendererTarget(
                name="custom", display_name="Custom Renderer",
                notes="Generic slot for a bespoke renderer integration.",
            ),
        ]
        for target in defaults:
            self.register(target)
