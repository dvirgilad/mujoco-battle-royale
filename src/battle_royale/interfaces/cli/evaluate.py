"""
Usage:
    python -m battle_royale.interfaces.cli.evaluate --checkpoint runs/checkpoint_100k --num-agents 6
"""
from __future__ import annotations

import argparse

from stable_baselines3 import PPO

from battle_royale.application.evaluation.evaluator import Evaluator
from battle_royale.application.training.snapshot_pool import SnapshotPool
from battle_royale.infrastructure.config.yaml_loader import load_config
from battle_royale.infrastructure.physics.mujoco_env import MuJoCoEnvironment
from battle_royale.interfaces.pettingzoo.env import BattleRoyaleEnv


def main(
    checkpoint_path: str, num_agents: int, config_path: str = "config/default.yaml"
) -> dict:
    config = load_config(config_path)
    config.training.num_agents = num_agents

    def env_factory(n: int):
        mujoco_env = MuJoCoEnvironment(config=config)
        return BattleRoyaleEnv(env=mujoco_env, config=config)

    model = PPO.load(checkpoint_path)
    pool = SnapshotPool(save_dir="/tmp/eval_snapshots")

    # Null logger for evaluation
    class _PrintLogger:
        def log(self, metrics, step):
            print(f"[eval step={step}] {metrics}")

        def save_artifact(self, path, name):
            pass

    evaluator = Evaluator(
        env_factory=env_factory, snapshot_pool=pool, logger=_PrintLogger()
    )
    metrics = evaluator.evaluate(model=model, num_agents=num_agents, num_episodes=10)
    print(metrics)
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Battle Royale agents")
    parser.add_argument(
        "--checkpoint", required=True, help="Path to SB3 model checkpoint"
    )
    parser.add_argument(
        "--num-agents", type=int, default=4, help="Number of agents to evaluate with"
    )
    parser.add_argument(
        "--config", default="config/default.yaml", help="Path to YAML config"
    )
    args = parser.parse_args()
    main(
        checkpoint_path=args.checkpoint,
        num_agents=args.num_agents,
        config_path=args.config,
    )
