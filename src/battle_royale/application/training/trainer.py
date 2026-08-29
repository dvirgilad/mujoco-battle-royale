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
            gamma=self._config.ppo.gamma,
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

    # Aggression-boost curriculum (OpenAI-sumo style: a dense shaping reward that
    # is strong early and annealed toward the true objective). Both aggression
    # terms (push + approach) are potential-based, so scaling them changes only
    # the learning-gradient strength, never the optimal policy -- we can crank
    # them up to FORCE pushing to survive the melee-risk transition (committing
    # to a shove at n>=3 exposes you to the others, so the sparse win reward
    # alone collapses to caution), then anneal back to 1.0 so the final policy is
    # optimised on the unshaped objective. The boost holds across the agent-count
    # ramp + early melee, then decays; before the ramp it stays at 1.0 (pushing
    # forms fine 1v1 on baseline shaping).
    _AGGR_BOOST_MAX = 2.5
    _AGGR_BOOST_HOLD_END = 0.9  # hold the boost until here, then anneal to 1.0

    # Agent-count curriculum (only active when config.training.
    # curriculum_start_agents > 0). Hold the start count for the first
    # _AGENT_RAMP_START of training so the pushing skill forms at low n (no melee
    # risk), then ramp integer-by-integer up to num_agents by _AGENT_RAMP_END,
    # then hold the full count. The start-count phase deliberately spans the
    # random-opponent warmup so hunting+pushing are both learned 1v1 first.
    _AGENT_RAMP_START = 0.4
    _AGENT_RAMP_END = 0.75

    @classmethod
    def _curriculum_agent_count(cls, frac: float, start_n: int, final_n: int) -> int:
        if start_n <= 0 or start_n >= final_n:
            return final_n
        if frac < cls._AGENT_RAMP_START:
            return start_n
        if frac >= cls._AGENT_RAMP_END:
            return final_n
        span = max(cls._AGENT_RAMP_END - cls._AGENT_RAMP_START, 1e-9)
        prog = (frac - cls._AGENT_RAMP_START) / span
        return int(round(start_n + (final_n - start_n) * prog))

    @classmethod
    def _random_opponent_prob(cls, frac: float) -> float:
        """Fraction of episodes played against random opponents at ``frac``.

        A warmup at ``start_prob`` (learn to hunt/eject a passive target) then a
        linear decay to ``end_prob`` over the rest of training, keeping a residual
        so the hunting skill is not forgotten. Extending random exposure beyond
        this residual was shown to erode melee pushing into turtling, so the melee
        robustness is instead handled by the aggression boost, not more random.
        """
        warmup = cls._CURRICULUM_WARMUP_FRAC
        start_p = cls._CURRICULUM_START_PROB
        end_p = cls._CURRICULUM_END_PROB
        if frac < warmup:
            return start_p
        decay = (frac - warmup) / max(1.0 - warmup, 1e-9)
        return start_p + (end_p - start_p) * decay

    @classmethod
    def _aggression_scale(cls, frac: float, start_n: int, final_n: int) -> float:
        """Multiplier on the push+approach shaping at ``frac``.

        Only boosts when the agent-count curriculum is active. Holds 1.0 until the
        ramp begins (pushing forms fine 1v1), rises to ``_AGGR_BOOST_MAX`` across
        the ramp + early melee to force pushing through the melee-risk transition,
        then anneals back to 1.0 so the final policy is optimised unshaped.
        """
        if not 0 < start_n < final_n:
            return 1.0
        boost = cls._AGGR_BOOST_MAX
        ramp_start = cls._AGENT_RAMP_START
        hold_end = cls._AGGR_BOOST_HOLD_END
        if frac < ramp_start:
            return 1.0
        if frac < hold_end:
            # Ramp up over the agent-count ramp, then hold at the max.
            ramp_span = max(cls._AGENT_RAMP_END - ramp_start, 1e-9)
            rise = min((frac - ramp_start) / ramp_span, 1.0)
            return 1.0 + (boost - 1.0) * rise
        # Anneal the boost back to baseline over the tail of training.
        decay = (frac - hold_end) / max(1.0 - hold_end, 1e-9)
        return boost + (1.0 - boost) * decay

    def _make_callback(self):
        from stable_baselines3.common.callbacks import BaseCallback

        config = self._config
        pool = self._snapshot_pool
        tracker = self._tracker
        env = self._env
        start_n = config.training.curriculum_start_agents
        final_n = config.training.num_agents
        agent_count = type(self)._curriculum_agent_count
        random_prob = type(self)._random_opponent_prob
        aggression_scale = type(self)._aggression_scale
        logger = self._logger

        class SelfPlayCallback(BaseCallback):
            _last_n: int | None = None

            def _on_step(self) -> bool:
                frac = min(self.num_timesteps / max(config.training.total_steps, 1), 1.0)
                env._random_opponent_prob = random_prob(frac)
                # Aggression boost feeds the underlying physics env's shaping.
                scale = aggression_scale(frac, start_n, final_n)
                env._env._aggression_scale = scale
                # Agent-count curriculum: set the override the env reads on reset.
                n = agent_count(frac, start_n, final_n)
                if n != self._last_n:
                    env._num_agents_override = n
                    self._last_n = n
                    logger.log(
                        {
                            "curriculum/num_agents": float(n),
                            "curriculum/random_opponent_prob": env._random_opponent_prob,
                            "curriculum/aggression_scale": scale,
                        },
                        self.num_timesteps,
                    )
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
