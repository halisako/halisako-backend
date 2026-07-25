"""
Chess2Fight product orchestration layer.
"""


from core.ai_router import AIProvider
from products.cinema.cinematic_engine import CinematicEngine


class FightOrchestrator:

    def __init__(self, ai_provider: AIProvider):
        self.ai_provider = ai_provider
        self._cinematic_engine = CinematicEngine()

