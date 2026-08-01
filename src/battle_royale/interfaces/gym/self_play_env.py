"""Single-agent self-play environment.

SB3 controls a single *learner* agent (``agent_0``). Every other agent in the
arena is driven by a frozen opponent policy sampled from the snapshot pool at
the start of each episode. When the pool is empty (early training) opponents
act randomly, which is enough to bootstrap self-play.

Exposing the problem as a single-agent Gymnasium env is what makes the three
success targets measurable: each episode is one match of learner-vs-pool, so it
yields a clean win/loss for win-rate and Elo, and the same env instantiated at a
different ``num_agents`` gives the zero-shot generalization test.
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from battle_royale.application.training.snapshot_pool import SnapshotPool
from battle_royale.domain.entities.agent import Agent
from battle_royale.domain.entities.arena import Arena
from battle_royale.domain.services.observation import ObservationBuilder
from battle_royale.infrastructure.config.yaml_loader import Config

_OBS_DIM = 17
_ACT_DIM = 2


class SelfPlayEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        env,
        config: Config,
        snapshot_pool: SnapshotPool,
        learner_id: str = "agent_0",
        max_steps: int = 1000,
        randomize_learner: bool = False,
        random_opponent_prob: float = 0.0,
    ) -> None:
        super().__init__()
        self._env = env
        self._config = config
        self._pool = snapshot_pool
        self._learner_id = learner_id
        self._max_steps = max_steps
        # Fraction of episodes played against passive/random opponents instead of
        # pool snapshots. Self-play opponents are aggressive and come to the
        # learner, so a pure-self-play policy never learns to *hunt* non-moving
        # targets and loses to a random baseline. Mixing in random opponents
        # teaches robust ejection of passive play (needed for the dominance
        # demonstration: 1 trained vs N-1 untrained).
        self._random_opponent_prob = random_opponent_prob
        # When True the learner occupies a random slot each episode. Spawns are
        # symmetric on a circle, so a learner fixed to agent_0 always starts at
        # the same angle and overfits to its absolute start position (obs[0:2]),
        # collapsing when evaluated from other slots. Randomizing forces a
        # rotation-robust policy that matches the sweep protocol.
        self._randomize_learner = randomize_learner
        self._arena = Arena(radius=config.arena.radius)
        self._num_agents = config.training.num_agents

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(_OBS_DIM,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(_ACT_DIM,), dtype=np.float32
        )

        self._agents: dict[str, Agent] = {}
        self._opponent: Any = None
        self._step_count = 0

    # -- opponent management ------------------------------------------------
    def _load_opponent(self) -> Any:
        if self._pool.is_empty():
            return None
        path = self._pool.sample_path()
        if path is None:
            return None
        from stable_baselines3 import PPO  # noqa: PLC0415

        return PPO.load(str(path), device="cpu")

    # -- gym API ------------------------------------------------------------
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._num_agents = self._config.training.num_agents
        self._agents = self._env.reset(num_agents=self._num_agents)
        if self._randomize_learner:
            idx = int(self.np_random.integers(self._num_agents))
            self._learner_id = f"agent_{idx}"
        # With some probability play against passive/random opponents; otherwise
        # sample a frozen policy from the pool. (None => random actions.)
        if self.np_random.random() < self._random_opponent_prob:
            self._opponent = None
        else:
            self._opponent = self._load_opponent()
        self._step_count = 0
        return self._obs_for(self._learner_id), {}

    def step(self, action):
        actions: dict[str, np.ndarray] = {}
        for aid, agent in self._agents.items():
            if not agent.alive:
                actions[aid] = np.zeros(_ACT_DIM, dtype=np.float32)
            elif aid == self._learner_id:
                actions[aid] = np.asarray(action, dtype=np.float32)
            else:
                actions[aid] = self._opponent_action(aid)

        self._agents, rewards, _terminations, _truncations = self._env.step(actions)
        self._step_count += 1

        learner = self._agents[self._learner_id]
        alive_ids = [aid for aid, ag in self._agents.items() if ag.alive]
        won = learner.alive and len(alive_ids) == 1

        terminated = (not learner.alive) or won
        truncated = self._step_count >= self._max_steps

        info: dict[str, Any] = {}
        if terminated or truncated:
            eliminations = sum(
                1
                for aid, ag in self._agents.items()
                if aid != self._learner_id and not ag.alive
            )
            info["match_result"] = {
                "win": bool(won),
                "length": self._step_count,
                "eliminations": eliminations,
            }
            info["is_success"] = bool(won)

        return (
            self._obs_for(self._learner_id),
            float(rewards[self._learner_id]),
            terminated,
            truncated,
            info,
        )

    # -- helpers ------------------------------------------------------------
    def _opponent_action(self, agent_id: str) -> np.ndarray:
        if self._opponent is None:
            return self.np_random.uniform(-1.0, 1.0, _ACT_DIM).astype(np.float32)
        obs = self._obs_for(agent_id)
        act, _ = self._opponent.predict(obs, deterministic=True)
        return np.asarray(act, dtype=np.float32)

    def _obs_for(self, agent_id: str) -> np.ndarray:
        agent = self._agents[agent_id]
        # Reflect the (possibly shrinking) boundary in the observation so the
        # agent perceives the storm closing in.
        radius = getattr(self._env, "current_radius", self._arena.radius)
        arena = Arena(radius=radius)
        return ObservationBuilder.build(agent, list(self._agents.values()), arena)
