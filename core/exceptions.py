"""Domain-specific exceptions, kept separate from Pydantic/FastAPI's own
so route handlers can catch them precisely and map them to the right
HTTP status code (see api/chess2fight.py)."""



class Chess2FightError(Exception):
    """Base class for all Chess2Fight domain errors."""


class InvalidPGNError(Chess2FightError):
    """Raised when the supplied PGN can't be parsed into a game."""


class AIProviderError(Chess2FightError):
    """Raised when an AI provider call fails unrecoverably.

    Callers (see products/chess2fight/narrative_generator.py) generally
    catch this internally and fall back to TemplateProvider rather than
    letting it bubble up — a flaky AI provider shouldn't take down the
    whole endpoint when a perfectly good deterministic fallback exists.
    """

class ImageProviderError(Chess2FightError):
    """Raised when a requested image provider is not registered, or a
    registered provider fails to generate an image."""

class VideoBuilderError(Chess2FightError):
    """Raised when FFmpeg cannot build the fight video."""