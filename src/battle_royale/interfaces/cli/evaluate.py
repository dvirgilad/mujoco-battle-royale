"""
Usage:
    # single agent-count evaluation against a run's snapshot pool
    python -m battle_royale.interfaces.cli.evaluate \
        --checkpoint runs/4v4_baseline/snapshots/snapshot_1000000 \
        --run-dir runs/4v4_baseline --num-agents 6

    # generalization sweep (train=4 -> eval on 4/6/8), prints a table
    python -m battle_royale.interfaces.cli.evaluate \
        --checkpoint runs/4v4_baseline/snapshots/snapshot_1000000 \
        --run-dir runs/4v4_baseline --sweep
"""

from __future__ import annotations

import argparse

from stable_baselines3 import PPO

from battle_royale.application.evaluation.evaluator import Evaluator
from battle_royale.application.training.snapshot_pool import SnapshotPool
from battle_royale.infrastructure.config.yaml_loader import load_config
from battle_royale.infrastructure.logging.console import StdoutLogger
from battle_royale.infrastructure.physics.mujoco_env import MuJoCoEnvironment
from battle_royale.interfaces.pettingzoo.env import BattleRoyaleEnv


def _evaluate_one(model, config, pool, logger, num_agents, num_episodes):
    config.training.num_agents = num_agents

    def env_factory(_n):
        mujoco_env = MuJoCoEnvironment(config=config)
        return BattleRoyaleEnv(env=mujoco_env, config=config)

    evaluator = Evaluator(env_factory=env_factory, snapshot_pool=pool, logger=logger)
    return evaluator.evaluate(
        model=model, num_agents=num_agents, num_episodes=num_episodes
    )


def main(
    checkpoint_path: str,
    num_agents: int,
    config_path: str = "config/default.yaml",
    run_dir: str | None = None,
    sweep: bool = False,
) -> dict:
    config = load_config(config_path)
    num_episodes = config.evaluation.num_episodes

    model = PPO.load(checkpoint_path)

    # Opponents come from the trained run's snapshot pool when available.
    pool = SnapshotPool(
        save_dir=f"{run_dir}/snapshots" if run_dir else "runs/latest/snapshots"
    )
    found = pool.discover()
    logger = StdoutLogger()
    print(f"Loaded {found} opponent snapshot(s) from the pool.")

    counts = config.evaluation.generalization_agent_counts if sweep else [num_agents]

    results: dict[int, dict] = {}
    for n in counts:
        metrics = _evaluate_one(model, config, pool, logger, n, num_episodes)
        results[n] = metrics

    _print_table(results)
    return (
        results[num_agents] if num_agents in results else next(iter(results.values()))
    )


def _print_table(results: dict[int, dict]) -> None:
    print("\n=== Generalization results ===")
    print(f"{'num_agents':>10} | {'win_rate':>9} | {'mean_len':>9}")
    print("-" * 36)
    for n, m in sorted(results.items()):
        print(
            f"{n:>10} | {m.get('win_rate', 0.0):>9.3f} | "
            f"{m.get('mean_episode_length', 0.0):>9.1f}"
        )


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
    parser.add_argument(
        "--run-dir",
        default=None,
        help="Run directory containing snapshots/ for the opponent pool",
    )
    parser.add_argument(
        "--sweep",
        action="store_true",
        help="Evaluate across generalization_agent_counts and print a table",
    )
    args = parser.parse_args()
    main(
        checkpoint_path=args.checkpoint,
        num_agents=args.num_agents,
        config_path=args.config,
        run_dir=args.run_dir,
        sweep=args.sweep,
    )
