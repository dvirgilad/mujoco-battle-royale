import numpy as np
from battle_royale.domain.entities.agent import Agent
from battle_royale.domain.entities.arena import Arena

_NUM_NEIGHBORS = 3
_OBS_DIM = (
    17  # 2 (pos) + 2 (vel) + 1 (dist_boundary) + 3 * 4 (neighbor rel_pos + rel_vel)
)


class ObservationBuilder:
    @staticmethod
    def build(agent: Agent, all_agents: list[Agent], arena: Arena) -> np.ndarray:
        obs = np.zeros(_OBS_DIM, dtype=np.float32)
        obs[0:2] = agent.position
        obs[2:4] = agent.velocity
        obs[4] = arena.radius - float(np.linalg.norm(agent.position))

        others = [a for a in all_agents if a.id != agent.id and a.alive]
        others_sorted = sorted(
            others,
            key=lambda a: float(np.linalg.norm(a.position - agent.position)),
        )
        neighbors = others_sorted[:_NUM_NEIGHBORS]

        for i, neighbor in enumerate(neighbors):
            obs[5 + i * 2 : 7 + i * 2] = neighbor.position - agent.position
            obs[11 + i * 2 : 13 + i * 2] = neighbor.velocity - agent.velocity

        return obs
