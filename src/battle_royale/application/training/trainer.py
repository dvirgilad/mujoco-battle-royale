from __future__ import annotations

from stable_baselines3 import PPO
from supersuit import pettingzoo_env_to_vec_env_v1

from battle_royale.application.metrics.tracker import MetricsTracker
from battle_royale.application.training.snapshot_pool import SnapshotPool
from battle_royale.domain.interfaces.logger import ILogger
from battle_royale.infrastructure.config.yaml_loader import Config


class Trainer:
    def __init__(
        self,
        env,
        logger: ILogger,
        snapshot_pool: SnapshotPool,
        tracker: MetricsTracker,
        config: Config,
    ) -> None:
        self._env = env
        self._logger = logger
        self._snapshot_pool = snapshot_pool
        self._tracker = tracker
        self._config = config

    def run(self) -> None:
        vec_env = pettingzoo_env_to_vec_env_v1(self._env)

        model = PPO(
            policy="MlpPolicy",
            env=vec_env,
            learning_rate=self._config.ppo.lr,
            n_steps=self._config.ppo.n_steps,
            batch_size=self._config.ppo.batch_size,
            clip_range=self._config.ppo.clip_range,
            n_epochs=self._config.ppo.n_epochs,
            verbose=1,
        )

        model.learn(
            total_timesteps=self._config.training.total_steps,
            callback=self._make_callback(model),
        )

    def _make_callback(self, model):
        from stable_baselines3.common.callbacks import BaseCallback

        config = self._config
        pool = self._snapshot_pool

        class SnapshotCallback(BaseCallback):
            def __init__(self):
                super().__init__()

            def _on_step(self) -> bool:
                if self.n_calls % config.training.snapshot_interval == 0:
                    pool.save(self.model, step=self.num_timesteps)
                return True

        return SnapshotCallback()
