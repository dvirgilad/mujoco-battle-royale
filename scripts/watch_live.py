"""Watch a Battle Royale match live in an interactive MuJoCo window.

Opens a real-time 3-D viewer (drag to orbit, scroll to zoom) and plays episodes
back-to-back until you close the window. Drive the agents with a trained policy
via --checkpoint, or leave it off to watch random agents.

Usage (from repo root, with the poetry env):
    python scripts/watch_live.py --checkpoint runs/selfplay_storm_final/snapshots/snapshot_930000.zip
    python scripts/watch_live.py --config config/combat_smoke.yaml --checkpoint runs/smoke_combat/snapshots/snapshot_300000.zip
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "src"))

import mujoco  # noqa: E402
import mujoco.viewer  # noqa: E402
import numpy as np  # noqa: E402

from battle_royale.domain.entities.arena import Arena  # noqa: E402
from battle_royale.domain.services.observation import ObservationBuilder  # noqa: E402
from battle_royale.infrastructure.config.yaml_loader import load_config  # noqa: E402
from battle_royale.infrastructure.physics.mujoco_env import MuJoCoEnvironment  # noqa: E402


def _policy(checkpoint):
    if not checkpoint:
        return None
    from stable_baselines3 import PPO

    return PPO.load(checkpoint, device="cpu")


def main(config_path, checkpoint, agents_override, realtime):
    config = load_config(config_path)
    n = agents_override or config.training.num_agents
    env = MuJoCoEnvironment(config=config)
    policy = _policy(checkpoint)
    dt = 0.01 / max(realtime, 1e-6)  # sim timestep, scaled by playback speed

    while True:
        agents = env.reset(num_agents=n)
        safe_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_GEOM, "safe_zone")
        ring_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_GEOM, "safe_ring")
        with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
            step = 0
            while viewer.is_running():
                arena = Arena(radius=env.current_radius)
                actions = {}
                for aid, ag in agents.items():
                    if not ag.alive:
                        actions[aid] = np.zeros(2, dtype=np.float32)
                    elif policy is not None:
                        obs = ObservationBuilder.build(ag, list(agents.values()), arena)
                        act, _ = policy.predict(obs, deterministic=True)
                        actions[aid] = act
                    else:
                        actions[aid] = np.random.uniform(-1, 1, 2).astype(np.float32)
                agents, _, _, _ = env.step(actions)
                if safe_id >= 0:
                    env.model.geom_size[safe_id, 0] = env.current_radius
                if ring_id >= 0:
                    env.model.geom_size[ring_id, 0] = env.current_radius * 1.03
                viewer.sync()
                time.sleep(dt)
                step += 1
                if sum(a.alive for a in agents.values()) <= 1 or step >= 1200:
                    time.sleep(1.5)  # pause on the result before the next match
                    break
            if not viewer.is_running():
                return


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Watch a match live")
    p.add_argument("--config", default="config/default.yaml")
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--agents", type=int, default=None, help="override agent count")
    p.add_argument(
        "--realtime",
        type=float,
        default=1.0,
        help="playback speed (1.0 = real time, 0.5 = slow-mo)",
    )
    a = p.parse_args()
    main(a.config, a.checkpoint, a.agents, a.realtime)
