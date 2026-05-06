from __future__ import annotations

import functools

import numpy as np
from gymnasium import spaces
from pettingzoo import ParallelEnv

from battle_royale.domain.interfaces.environment import IBattleRoyaleEnv
from battle_royale.domain.services.observation import ObservationBuilder
from battle_royale.domain.entities.arena import Arena
from battle_royale.infrastructure.config.yaml_loader import Config

_OBS_DIM = 17


class BattleRoyaleEnv(ParallelEnv):
    metadata = {"render_modes": [], "name": "battle_royale_v0"}

    def __init__(self, env: IBattleRoyaleEnv, config: Config) -> None:
        super().__init__()
        self._env = env
        self._config = config
        self._arena = Arena(radius=config.arena.radius)
        self._num_agents = config.training.num_agents
        self.render_mode = None
        self.possible_agents = [f"agent_{i}" for i in range(self._num_agents)]
        self.agents: list[str] = []
        self._agent_objects: list = []

    def reset(self, seed=None, options=None):
        self._num_agents = self._config.training.num_agents
        self.possible_agents = [f"agent_{i}" for i in range(self._num_agents)]
        agents_dict = self._env.reset(num_agents=self._num_agents)
        self.agents = list(agents_dict.keys())
        self._agent_objects = list(agents_dict.values())
        observations = self._build_observations()
        self.terminations = {aid: False for aid in self.agents}
        self.truncations = {aid: False for aid in self.agents}
        return observations, {aid: {} for aid in self.agents}

    def step(self, actions: dict[str, np.ndarray]):
        # Only pass actions for alive agents
        filtered = {
            aid: actions[aid]
            for aid in self.agents
            if not self.terminations.get(aid, False)
        }
        # Pad with zeros for terminated agents
        for aid in self.possible_agents:
            if aid not in filtered:
                filtered[aid] = np.zeros(2, dtype=np.float32)

        agents_dict, rewards, terminations, truncations = self._env.step(filtered)
        self._agent_objects = list(agents_dict.values())

        observations = self._build_observations()
        self.terminations = terminations
        self.truncations = truncations

        # Remove fully done agents from self.agents
        self.agents = [
            aid
            for aid in self.agents
            if not (terminations.get(aid, False) or truncations.get(aid, False))
        ]

        return (
            observations,
            rewards,
            terminations,
            truncations,
            {aid: {} for aid in observations},
        )

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent: str) -> spaces.Box:
        return spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(_OBS_DIM,),
            dtype=np.float32,
        )

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent: str) -> spaces.Box:
        return spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(2,),
            dtype=np.float32,
        )

    def _build_observations(self) -> dict[str, np.ndarray]:
        obs = {}
        for agent_obj in self._agent_objects:
            obs[agent_obj.id] = ObservationBuilder.build(
                agent_obj, self._agent_objects, self._arena
            )
        return obs
