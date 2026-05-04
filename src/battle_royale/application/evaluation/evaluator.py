from __future__ import annotations

from typing import Any, Callable
import numpy as np

from battle_royale.application.training.snapshot_pool import SnapshotPool
from battle_royale.domain.interfaces.logger import ILogger


class Evaluator:
    def __init__(
        self,
        env_factory: Callable[[int], Any],
        snapshot_pool: SnapshotPool,
        logger: ILogger,
    ) -> None:
        self._env_factory = env_factory
        self._snapshot_pool = snapshot_pool
        self._logger = logger

    def _load_opponent(self, fallback) -> Any:
        if not self._snapshot_pool.is_empty():
            path = self._snapshot_pool.sample_path()
            if path is not None:
                from stable_baselines3 import PPO  # noqa: PLC0415

                return PPO.load(str(path))
        return fallback

    def evaluate(self, model, num_agents: int, num_episodes: int) -> dict[str, float]:
        env = self._env_factory(num_agents)
        opponent = self._load_opponent(model)
        eval_wins = 0
        episode_lengths: list[int] = []

        for ep_idx in range(num_episodes):
            eval_agent = f"agent_{ep_idx % num_agents}"
            observations, _ = env.reset()
            done = False
            step_count = 0
            while not done:
                actions = {}
                for agent in env.agents:
                    obs = observations.get(agent, np.zeros(17))
                    m = model if agent == eval_agent else opponent
                    action, _ = m.predict(obs)
                    actions[agent] = action
                observations, _, terminations, _, _ = env.step(actions)
                step_count += 1
                alive = [a for a in env.agents if not terminations.get(a, True)]
                done = len(alive) <= 1 or step_count >= 1000
                if len(alive) == 1 and alive[0] == eval_agent:
                    eval_wins += 1
            episode_lengths.append(step_count)

        mean_len = float(np.mean(episode_lengths)) if episode_lengths else 0.0
        metrics = {
            "win_rate": eval_wins / num_episodes,
            "mean_episode_length": mean_len,
        }
        self._logger.log(metrics, step=0)
        return metrics
