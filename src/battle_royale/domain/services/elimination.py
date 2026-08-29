import numpy as np

from battle_royale.domain.entities.agent import Agent
from battle_royale.domain.entities.arena import Arena


class EliminationService:
    @staticmethod
    def is_eliminated(agent: Agent, arena: Arena) -> bool:
        if not agent.alive:
            return True
        return float(np.linalg.norm(agent.position)) > arena.radius
