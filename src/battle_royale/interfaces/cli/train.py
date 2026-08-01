"""
Usage:
    python -m battle_royale.interfaces.cli.train --config config/experiments/4v4_baseline.yaml
    python -m battle_royale.interfaces.cli.train --config config/default.yaml --logger stdout
"""

from __future__ import annotations

import argparse
import os

from battle_royale.application.metrics.tracker import MetricsTracker
from battle_royale.application.training.snapshot_pool import SnapshotPool
from battle_royale.application.training.trainer import Trainer
from battle_royale.infrastructure.config.yaml_loader import load_config
from battle_royale.infrastructure.logging.console import NullLogger, StdoutLogger
from battle_royale.infrastructure.physics.mujoco_env import MuJoCoEnvironment
from battle_royale.interfaces.gym.self_play_env import SelfPlayEnv


def _make_logger(kind: str, run_dir: str, config_path: str):
    if kind == "null":
        return NullLogger()
    if kind == "stdout":
        return StdoutLogger()
    # Imported lazily so offline runs never require the wandb package/login.
    from battle_royale.infrastructure.logging.wandb_logger import WandBLogger

    return WandBLogger(
        project="battle-royale",
        run_name=os.path.basename(run_dir),
        config={"config_path": config_path},
    )


def main(config_path: str, run_dir: str, logger_kind: str = "wandb") -> None:
    config = load_config(config_path)
    os.makedirs(run_dir, exist_ok=True)

    mujoco_env = MuJoCoEnvironment(config=config)
    snapshot_pool = SnapshotPool(
        save_dir=os.path.join(run_dir, "snapshots"),
        max_size=config.training.snapshot_pool_size,
    )
    self_play_env = SelfPlayEnv(
        env=mujoco_env,
        config=config,
        snapshot_pool=snapshot_pool,
        # Episode cap (config-driven). Combined with the rebalanced reward the
        # short cap discourages turtling to the step limit.
        max_steps=config.training.episode_max_steps,
        # Learner occupies a random slot each episode so it can't overfit to
        # agent_0's fixed spawn angle; this matches the rotating eval protocol.
        randomize_learner=True,
        # A quarter of episodes are played against passive/random opponents so
        # the policy learns to hunt and eject non-cooperative targets (needed to
        # dominate an untrained baseline).
        random_opponent_prob=0.25,
    )

    logger = _make_logger(logger_kind, run_dir, config_path)
    tracker = MetricsTracker(logger=logger)

    trainer = Trainer(
        env=self_play_env,
        logger=logger,
        snapshot_pool=snapshot_pool,
        tracker=tracker,
        config=config,
    )
    trainer.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Battle Royale agents")
    parser.add_argument(
        "--config", default="config/default.yaml", help="Path to YAML config"
    )
    parser.add_argument(
        "--run-dir", default="runs/latest", help="Directory to save outputs"
    )
    parser.add_argument(
        "--logger",
        default="wandb",
        choices=["wandb", "stdout", "null"],
        help="Metrics sink: wandb (default), stdout (offline), or null",
    )
    args = parser.parse_args()
    main(config_path=args.config, run_dir=args.run_dir, logger_kind=args.logger)
