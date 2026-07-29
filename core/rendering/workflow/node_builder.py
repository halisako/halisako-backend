"""Renderer-agnostic workflow graph nodes.

Every node in a workflow graph is constructed through `NodeBuilder`,
never by hand-assembling a dict — that's what keeps node construction
validated in one place and node IDs guaranteed unique. `WorkflowNode`
is a generic node representation: a type, a stable ID, its own
parameters, and named references to upstream nodes it depends on. It
carries no ComfyUI-specific (or any other renderer-specific) shape —
translating this generic graph into a specific renderer's exact
serialization format is an adapter's job, not this module's.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from core.rendering.workflow.exceptions import NodeConstructionError


class NodeType(str, Enum):
    """Every node type the Workflow Builder can emit."""

    IMAGE_LOADER = "image_loader"
    CHECKPOINT_LOADER = "checkpoint_loader"
    LORA_LOADER = "lora_loader"
    SAMPLER = "sampler"
    CONTROLNET = "controlnet"
    ANIMATEDIFF = "animatediff"
    VAE = "vae"
    VIDEO_COMBINE = "video_combine"
    AUDIO = "audio"
    SAVE_VIDEO = "save_video"
    PREVIEW = "preview"
    OUTPUT = "output"


class WorkflowNode(BaseModel):
    """One node in a renderer-agnostic workflow graph.

    `inputs` maps a named input slot on this node (e.g. "checkpoint",
    "reference") to the `node_id` of the upstream node that supplies
    it — that's the graph's edges. `parameters` holds everything about
    this node that isn't a graph edge (model names, numeric settings,
    flags).
    """

    node_id: str = Field(
        ..., min_length=1, description="Unique identifier for this node within its workflow."
    )
    node_type: NodeType = Field(..., description="Which kind of node this is.")
    parameters: dict[str, Any] = Field(default_factory=dict, description="This node's own configuration.")
    inputs: dict[str, str] = Field(
        default_factory=dict, description="Named input slot -> upstream node_id, forming the graph's edges."
    )


class NodeBuilder:
    """Constructs WorkflowNodes with guaranteed-unique IDs and
    validated parameters.

    One NodeBuilder instance should back one workflow build — its
    internal counter is what guarantees every node it produces has a
    unique ID, so two separate builds (each with their own
    NodeBuilder) never collide, but nodes from the *same* build never
    duplicate either.
    """

    def __init__(self, id_prefix: str = "node") -> None:
        """Initializes a NodeBuilder.

        Args:
            id_prefix: Prefix used when generating node IDs, useful
                for keeping IDs readable/traceable to a specific
                sequence or beat when multiple builders' output is
                later merged.
        """
        self._id_prefix = id_prefix
        self._counter = 0

    def image_loader(self, image_path: str) -> WorkflowNode:
        """Builds a node that loads a single image asset."""
        if not image_path:
            raise NodeConstructionError("image_loader requires a non-empty image_path.")
        return self._build(NodeType.IMAGE_LOADER, {"image_path": image_path})

    def checkpoint_loader(self, model: str) -> WorkflowNode:
        """Builds a node that loads a base checkpoint model."""
        if not model:
            raise NodeConstructionError("checkpoint_loader requires a non-empty model identifier.")
        return self._build(NodeType.CHECKPOINT_LOADER, {"model": model})

    def lora_loader(self, lora: str, checkpoint_node_id: str, strength: float = 1.0) -> WorkflowNode:
        """Builds a node that applies a LoRA on top of a checkpoint node."""
        if not lora:
            raise NodeConstructionError("lora_loader requires a non-empty lora identifier.")
        if not 0.0 <= strength <= 2.0:
            raise NodeConstructionError(f"lora strength must be within [0, 2], got {strength}.")
        return self._build(
            NodeType.LORA_LOADER,
            {"lora": lora, "strength": strength},
            inputs={"checkpoint": checkpoint_node_id},
        )

    def sampler(
        self,
        checkpoint_node_id: str,
        cfg_scale: float,
        steps: int,
        sampler_name: str,
        scheduler: str,
        seed: int,
    ) -> WorkflowNode:
        """Builds a sampling node that generates frames from a checkpoint."""
        if cfg_scale <= 0:
            raise NodeConstructionError(f"cfg_scale must be positive, got {cfg_scale}.")
        if steps <= 0:
            raise NodeConstructionError(f"steps must be positive, got {steps}.")
        return self._build(
            NodeType.SAMPLER,
            {
                "cfg_scale": cfg_scale, "steps": steps,
                "sampler_name": sampler_name, "scheduler": scheduler, "seed": seed,
            },
            inputs={"checkpoint": checkpoint_node_id},
        )

    def controlnet(
        self, control_type: str, reference_node_id: str | None = None, strength: float = 1.0
    ) -> WorkflowNode:
        """Builds a ControlNet conditioning node, e.g. for pose or
        depth guidance from a reference image, or for a visual effect
        intensity signal with no upstream reference at all.

        Args:
            control_type: What this ControlNet conditions on, e.g.
                "pose" or "effect_fire".
            reference_node_id: The upstream node supplying the
                reference, if this ControlNet is conditioned on one.
                None for a conditioning signal with no reference image
                (e.g. a pure intensity-driven effect).
            strength: Conditioning strength.
        """
        if not control_type:
            raise NodeConstructionError("controlnet requires a non-empty control_type.")
        inputs = {"reference": reference_node_id} if reference_node_id else {}
        return self._build(
            NodeType.CONTROLNET, {"control_type": control_type, "strength": strength}, inputs=inputs
        )

    def animatediff(
        self,
        motion_clips: list[str],
        frame_count: int,
        context_length: int = 16,
        motion_strength: float = 0.5,
        camera_shake: float = 0.0,
        fov_degrees: float = 35.0,
    ) -> WorkflowNode:
        """Builds an AnimateDiff motion node driving frame-to-frame
        temporal coherence and camera movement.

        There is no dedicated "camera" node among the twelve node
        types this builder supports — camera intent (movement
        strength, shake, field of view) is realistically expressed as
        motion-module parameters in a diffusion-video pipeline, so the
        Camera Builder stage's output is carried here rather than as
        a separate node type. See workflow_builder.py's Camera Builder
        stage for how shot type and intensity are mapped into these
        values via ParameterMapper.

        `motion_clips` is a list, not a single joined string,
        specifically so each clip identifier a beat's actions resolve
        to stays individually traceable (e.g. for
        `validators.py`'s "required asset referenced" check) rather
        than being folded into one opaque string AnimateDiff would
        then need to re-parse.

        Args:
            motion_clips: One or more animation clip identifiers to
                chain/blend within this beat via AnimateDiff's context
                window.
            frame_count: Number of frames this node should drive.
            context_length: AnimateDiff context window length.
            motion_strength: 0-1 overall motion intensity.
            camera_shake: 0-1 camera shake amount.
            fov_degrees: Field of view, in degrees.
        """
        if not motion_clips:
            raise NodeConstructionError("animatediff requires at least one motion clip identifier.")
        if frame_count <= 0:
            raise NodeConstructionError(f"frame_count must be positive, got {frame_count}.")
        return self._build(
            NodeType.ANIMATEDIFF,
            {
                "motion_clips": list(motion_clips),
                "frame_count": frame_count,
                "context_length": context_length,
                "motion_strength": motion_strength,
                "camera_shake": camera_shake,
                "fov_degrees": fov_degrees,
            },
        )

    def vae(self, vae_model: str) -> WorkflowNode:
        """Builds a VAE node used to decode latents into pixels."""
        if not vae_model:
            raise NodeConstructionError("vae requires a non-empty vae_model identifier.")
        return self._build(NodeType.VAE, {"vae_model": vae_model})

    def video_combine(self, source_node_id: str, fps: int, frame_interpolation: bool) -> WorkflowNode:
        """Builds a node that combines rendered frames into a video."""
        if fps <= 0:
            raise NodeConstructionError(f"fps must be positive, got {fps}.")
        return self._build(
            NodeType.VIDEO_COMBINE,
            {"fps": fps, "frame_interpolation": frame_interpolation},
            inputs={"frames": source_node_id},
        )

    def audio(self, mood: str, impact_beats: int, voice_placeholder: bool) -> WorkflowNode:
        """Builds an audio-intent node. No audio is actually generated
        here — this only records intent for a future audio adapter."""
        if not mood:
            raise NodeConstructionError("audio requires a non-empty mood.")
        if impact_beats < 0:
            raise NodeConstructionError(f"impact_beats cannot be negative, got {impact_beats}.")
        return self._build(
            NodeType.AUDIO,
            {"mood": mood, "impact_beats": impact_beats, "voice_placeholder": voice_placeholder},
        )

    def save_video(self, source_node_id: str, filename_prefix: str) -> WorkflowNode:
        """Builds a node that persists a combined video to storage."""
        if not filename_prefix:
            raise NodeConstructionError("save_video requires a non-empty filename_prefix.")
        return self._build(
            NodeType.SAVE_VIDEO, {"filename_prefix": filename_prefix}, inputs={"video": source_node_id}
        )

    def preview(self, source_node_id: str) -> WorkflowNode:
        """Builds a node that exposes a preview of an upstream node's output."""
        return self._build(NodeType.PREVIEW, {}, inputs={"source": source_node_id})

    def output(self, source_node_id: str, output_name: str) -> WorkflowNode:
        """Builds a terminal node marking a graph output — see
        validators.py's "missing outputs" check, which looks for at
        least one of these."""
        if not output_name:
            raise NodeConstructionError("output requires a non-empty output_name.")
        return self._build(NodeType.OUTPUT, {"output_name": output_name}, inputs={"source": source_node_id})

    def _build(
        self, node_type: NodeType, parameters: dict[str, Any], inputs: dict[str, str] | None = None
    ) -> WorkflowNode:
        """Shared construction path for every node method — this is
        the only place `_counter` is incremented, which is what
        guarantees uniqueness."""
        self._counter += 1
        node_id = f"{self._id_prefix}_{node_type.value}_{self._counter}"
        return WorkflowNode(node_id=node_id, node_type=node_type, parameters=parameters, inputs=inputs or {})
