"""Measure the three headline KPIs (active pushing / balance / dominance) at
several agent counts, using the real BattleRoyaleEnv (same mechanics as
training: no storm, per-episode random rotation, episode cap =
training.episode_max_steps).

    poetry run python scripts/eval_kpis.py \
        --checkpoint runs/sumo_light/snapshots/snapshot_001500000.zip \
        --config config/sumo_light.yaml --counts 4,6,8 --episodes 100

Definitions
    active pushing : fraction of self-play games (all agents = trained policy)
                     that resolve to a single survivor. With no storm, a
                     resolution is a genuine push-out.
    balance        : per-slot win share among N identical copies (ideal 1/N).
    dominance edge : WIN - LOSS for one trained agent (rotating slot) vs N-1
                     random agents.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from stable_baselines3 import PPO  # noqa: E402

from battle_royale.infrastructure.config.yaml_loader import load_config  # noqa: E402
from battle_royale.infrastructure.physics.mujoco_env import MuJoCoEnvironment  # noqa: E402
from battle_royale.interfaces.pettingzoo.env import BattleRoyaleEnv  # noqa: E402

_Z = np.zeros(17, dtype=np.float32)


def _make_env(config, n, seed):
    config.training.num_agents = n
    menv = MuJoCoEnvironment(config=config)
    menv._rng = np.random.default_rng(seed)  # seed the arena rotation RNG
    return BattleRoyaleEnv(env=menv, config=config)


def _run_episode(env, model, cap, det, dominance, eval_agent):
    obs, _ = env.reset()
    step, done, alive = 0, False, list(env.agents)
    while not done:
        actions = {}
        if dominance:
            for a in env.agents:
                if a == eval_agent:
                    act, _ = model.predict(obs.get(a, _Z), deterministic=det)
                    actions[a] = act
                else:
                    actions[a] = np.random.uniform(-1, 1, 2).astype(np.float32)
        else:
            acting = list(env.agents)
            batch = np.array([obs.get(a, _Z) for a in acting], dtype=np.float32)
            acts, _ = model.predict(batch, deterministic=det)
            actions = {a: acts[i] for i, a in enumerate(acting)}
        obs, _, _, _, _ = env.step(actions)
        step += 1
        alive = list(env.agents)
        done = len(alive) <= 1 or step >= cap
    return alive


def measure(checkpoint, config_path, n, episodes, det, cap):
    model = PPO.load(checkpoint, device="cpu")

    cfg = load_config(config_path)
    env = _make_env(cfg, n, seed=1234)
    np.random.seed(1234)
    resolved, winbyslot = 0, np.zeros(n)
    for _ in range(episodes):
        alive = _run_episode(env, model, cap, det, dominance=False, eval_agent=None)
        if len(alive) == 1:
            resolved += 1
            winbyslot[int(alive[0].split("_")[1])] += 1

    cfg = load_config(config_path)
    env2 = _make_env(cfg, n, seed=4321)
    np.random.seed(4321)
    wins = losses = 0
    for ep in range(episodes):
        ea = f"agent_{ep % n}"
        alive = _run_episode(env2, model, cap, det, dominance=True, eval_agent=ea)
        if len(alive) == 1 and alive[0] == ea:
            wins += 1
        elif ea not in alive:
            losses += 1

    return {
        "n": n,
        "pushing": resolved / episodes,
        "balance": (winbyslot / episodes).tolist(),
        "dom_win": wins / episodes,
        "dom_loss": losses / episodes,
        "dom_edge": (wins - losses) / episodes,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--config", default="config/sumo_light.yaml")
    ap.add_argument("--counts", default="4,6,8")
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument(
        "--det", type=int, default=1, help="1 = deterministic actions (default)"
    )
    ap.add_argument("--cap", type=int, default=600, help="episode step cap")
    args = ap.parse_args()

    print(f"# episodes={args.episodes} deterministic={bool(args.det)} cap={args.cap}")
    for n in (int(x) for x in args.counts.split(",")):
        r = measure(
            args.checkpoint, args.config, n, args.episodes, bool(args.det), args.cap
        )
        bal = " ".join(f"{x:.2f}" for x in r["balance"])
        print(
            f"n={r['n']:>2}  pushing={r['pushing']:.2f}  "
            f"dom_edge={r['dom_edge']:+.2f} (W {r['dom_win']:.2f} / L {r['dom_loss']:.2f})  "
            f"balance=[{bal}]  (ideal {1/n:.3f})"
        )


if __name__ == "__main__":
    main()
