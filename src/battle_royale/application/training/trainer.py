from __future__ import annotations

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from battle_royale.application.metrics.tracker import MetricsTracker
from battle_royale.application.training.snapshot_pool import SnapshotPool
from battle_royale.domain.interfaces.logger import ILogger
from battle_royale.infrastructure.config.yaml_loader import Config


class Trainer:
    """Self-play PPO trainer.

    The learner is trained as a single-agent policy inside ``SelfPlayEnv``; its
    opponents are frozen snapshots sampled from ``snapshot_pool``. A callback
    saves new snapshots (feeding the opponent pool) and records each match
    outcome into ``tracker`` for win-rate / Elo logging.
    """

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

    def run(self) -> PPO:
        vec_env = DummyVecEnv([lambda: self._env])

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
            callback=self._make_callback(),
        )

        # Persist the final policy as the newest pool member.
        self._snapshot_pool.save(model, step=self._config.training.total_steps)
        return model

    # Curriculum for the fraction of episodes played against passive / random
    # opponents. A pure-random WARMUP teaches the learner to *hunt* and eject
    # passive targets (self-play alone never requires this -> a defensive policy
    # that loses to a random baseline). After the warmup the random fraction
    # decays so training becomes mostly genuine self-play against the pool
    # (which is what produces decisive ~1/N outcomes among equal agents), while
    # keeping a residual random fraction so the hunting skill is not forgotten.
    _CURRICULUM_WARMUP_FRAC = 0.3  # first 30% of steps: 100% random opponents
    _CURRICULUM_START_PROB = 1.0
    _CURRICULUM_END_PROB = 0.35

    def _make_callback(self):
        from stable_baselines3.common.callbacks import BaseCallback

        config = self._config
        pool = self._snapshot_pool
        tracker = self._tracker
        env = self._env
        start_p = self._CURRICULUM_START_PROB
        end_p = self._CURRICULUM_END_PROB
        warmup = self._CURRICULUM_WARMUP_FRAC

        class SelfPlayCallback(BaseCallback):
            def _on_step(self) -> bool:
                frac = min(self.num_timesteps / max(config.training.total_steps, 1), 1.0)
                if frac < warmup:
                    env._random_opponent_prob = start_p
                else:
                    decay = (frac - warmup) / max(1.0 - warmup, 1e-9)
                    env._random_opponent_prob = start_p + (end_p - start_p) * decay
                for info in self.locals.get("infos", []):
                    result = info.get("match_result")
                    if result is not None:
                        tracker.record_match(
                            won=result["win"],
                            episode_length=result["length"],
                            eliminations=result["eliminations"],
                            step=self.num_timesteps,
                        )
                if self.n_calls % config.training.snapshot_interval == 0:
                    pool.save(self.model, step=self.num_timesteps)
                return True

        return SelfPlayCallback()
