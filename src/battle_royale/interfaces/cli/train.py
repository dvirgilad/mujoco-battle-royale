"""
Usage:
    python -m battle_royale.interfaces.cli.train --config config/experiments/4v4_baseline.yaml
"""

from __future__ import annotations

import argparse
import os

from battle_royale.application.metrics.tracker import MetricsTracker
from battle_royale.application.training.snapshot_pool import SnapshotPool
from battle_royale.application.training.trainer import Trainer
from battle_royale.infrastructure.config.yaml_loader import load_config
from battle_royale.infrastructure.logging.wandb_logger import WandBLogger
from battle_royale.infrastructure.physics.mujoco_env import MuJoCoEnvironment
from battle_royale.interfaces.pettingzoo.env import BattleRoyaleEnv


def main(config_path: str, run_dir: str) -> None:
    config = load_config(config_path)
    os.makedirs(run_dir, exist_ok=True)

    mujoco_env = MuJoCoEnvironment(config=config)
    pz_env = BattleRoyaleEnv(env=mujoco_env, config=config)

    logger = WandBLogger(
        project="battle-royale",
        run_name=os.path.basename(run_dir),
        config={"config_path": config_path},
    )
    snapshot_pool = SnapshotPool(
        save_dir=os.path.join(run_dir, "snapshots"),
        max_size=config.training.snapshot_pool_size,
    )
    tracker = MetricsTracker(logger=logger)

    trainer = Trainer(
        env=pz_env,
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
    args = parser.parse_args()
    main(config_path=args.config, run_dir=args.run_dir)
