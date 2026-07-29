"""Exception hierarchy for the Workflow Builder.

Every error the Workflow Builder can raise is a subclass of
`WorkflowBuilderError`, so callers can catch broadly (`except
WorkflowBuilderError`) or narrowly (`except AssetResolutionError`) as
their situation requires. No exception here carries renderer-specific
detail — that stays out of this layer entirely, per the architecture's
separation between the renderer-agnostic Workflow Builder and whatever
adapter eventually serializes its output for a specific renderer.
"""

from __future__ import annotations


class WorkflowBuilderError(Exception):
    """Base class for every error raised by the Workflow Builder."""


class UnsupportedRendererError(WorkflowBuilderError):
    """Raised when a workflow is requested for a renderer target that
    has not been registered with the WorkflowRegistry."""


class UnknownStyleError(WorkflowBuilderError):
    """Raised when a StyleProfile references a style that has no
    registered WorkflowTemplate."""


class AssetResolutionError(WorkflowBuilderError):
    """Raised when the AssetResolver cannot determine a required
    model, LoRA, VAE, ControlNet, IPAdapter, animation, texture, HDRI,
    weapon, or environment asset for the given style and sequence."""


class NodeConstructionError(WorkflowBuilderError):
    """Raised when the NodeBuilder is asked to construct a node with
    invalid or incomplete parameters."""


class WorkflowValidationError(WorkflowBuilderError):
    """Raised when a fully assembled workflow fails validation —
    missing assets or models, duplicate node IDs, a broken graph,
    missing outputs, unsupported parameters, or invalid timing.

    Carries the full list of validation failures, not just the first
    one, so a caller can report everything wrong in a single pass
    rather than fixing and re-running one error at a time.
    """

    def __init__(self, failures: list[str]) -> None:
        self.failures = list(failures)
        summary = "; ".join(self.failures) if self.failures else "unknown validation failure"
        super().__init__(f"Workflow validation failed: {summary}")


class InvalidCinematicSequenceError(WorkflowBuilderError):
    """Raised when a CinematicSequence is structurally invalid — e.g.
    a beat references an actor_id that isn't in the sequence's actor
    list, or timing is inconsistent."""
