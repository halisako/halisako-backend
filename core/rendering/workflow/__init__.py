"""Halisako Workflow Builder.

Converts cinematic intent (a `CinematicSequence`, in a chosen
`StyleProfile`) into a renderer-agnostic `RendererWorkflow` graph.
Contains no rendering logic, no AI inference, and no dependency on any
specific renderer (ComfyUI or otherwise) — see `workflow_builder.py`'s
module docstring for the full architectural rationale, and
`workflow_templates.py`'s module docstring for an important caveat
about the `CinematicSequence` input contract this subpackage builds
against.

Import order below matches this subpackage's dependency graph, which
is intentionally acyclic:

    exceptions
        -> workflow_templates, node_builder, workflow_registry
            -> parameter_mapper, asset_resolver
                -> validators
                    -> workflow_builder
"""

from __future__ import annotations

from core.rendering.workflow.asset_resolver import AssetResolver, ResolvedAssets
from core.rendering.workflow.exceptions import (
    AssetResolutionError,
    InvalidCinematicSequenceError,
    NodeConstructionError,
    UnknownStyleError,
    UnsupportedRendererError,
    WorkflowBuilderError,
    WorkflowValidationError,
)
from core.rendering.workflow.node_builder import NodeBuilder, NodeType, WorkflowNode
from core.rendering.workflow.parameter_mapper import ParameterMapper
from core.rendering.workflow.validators import validate_workflow
from core.rendering.workflow.workflow_builder import RendererWorkflow, WorkflowBuilder
from core.rendering.workflow.workflow_registry import RendererTarget, WorkflowRegistry
from core.rendering.workflow.workflow_templates import (
    ActorAction,
    ActorActionType,
    AudioDirection,
    CameraDefaults,
    CameraDirection,
    CameraShotType,
    CinematicActor,
    CinematicBeat,
    CinematicSequence,
    IntensityLevel,
    LightingDefaults,
    RenderDefaults,
    RenderStyleId,
    StyleProfile,
    VisualEffect,
    VisualEffectType,
    WorkflowTemplate,
    get_template,
    registered_styles,
)

__all__ = [
    # Main entry point
    "WorkflowBuilder",
    "RendererWorkflow",
    # Cinematic input contract (proposed — see workflow_templates.py)
    "CinematicSequence",
    "CinematicBeat",
    "CinematicActor",
    "CameraDirection",
    "CameraShotType",
    "VisualEffect",
    "VisualEffectType",
    "ActorAction",
    "ActorActionType",
    "AudioDirection",
    "IntensityLevel",
    # Style
    "StyleProfile",
    "RenderStyleId",
    "WorkflowTemplate",
    "LightingDefaults",
    "CameraDefaults",
    "RenderDefaults",
    "get_template",
    "registered_styles",
    # Building blocks
    "NodeBuilder",
    "WorkflowNode",
    "NodeType",
    "ParameterMapper",
    "AssetResolver",
    "ResolvedAssets",
    "validate_workflow",
    "WorkflowRegistry",
    "RendererTarget",
    # Exceptions
    "WorkflowBuilderError",
    "UnsupportedRendererError",
    "UnknownStyleError",
    "AssetResolutionError",
    "NodeConstructionError",
    "WorkflowValidationError",
    "InvalidCinematicSequenceError",
]
