"""Render a Battle Royale episode to a video file.

By default all agents act randomly. Pass --checkpoint to drive every agent with
a trained policy.

Usage:
    python main.py                                   # random agents -> output.mp4
    python main.py --checkpoint runs/latest/snapshots/snapshot_1000000.zip
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import mujoco  # noqa: E402
import numpy as np  # noqa: E402

from battle_royale.domain.entities.arena import Arena  # noqa: E402
from battle_royale.domain.services.observation import ObservationBuilder  # noqa: E402
from battle_royale.infrastructure.config.yaml_loader import load_config  # noqa: E402
from battle_royale.infrastructure.physics.mujoco_env import (  # noqa: E402
    MuJoCoEnvironment,
)
from battle_royale.infrastructure.recording.video_recorder import (  # noqa: E402
    VideoRecorder,
)


def run(
    config_path: str,
    checkpoint: str | None,
    out_path: str,
    max_steps: int,
    fps: int = 24,
) -> None:
    config = load_config(config_path)
    num_agents = config.training.num_agents

    env = MuJoCoEnvironment(config=config)
    agents = env.reset(num_agents=num_agents)

    policy = None
    if checkpoint:
        from stable_baselines3 import PPO

        policy = PPO.load(checkpoint, device="cpu")

    # Geoms whose radius we shrink each frame to visualise the closing storm.
    safe_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_GEOM, "safe_zone")
    ring_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_GEOM, "safe_ring")

    # Free tracking camera: a 3/4 view whose distance follows the storm, so the
    # action keeps filling the frame as the arena shrinks (instead of vanishing
    # to a dot in the centre).
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = [0.0, 0.0, 0.2]
    cam.azimuth = 90.0
    cam.elevation = -28.0

    recorder = VideoRecorder(output_path=out_path, fps=fps)

    with mujoco.Renderer(env.model, height=720, width=1280) as renderer:
        for _ in range(max_steps):
            # Observe the current (possibly shrinking) boundary, as in training.
            arena = Arena(radius=env.current_radius)
            actions = {}
            for aid, agent in agents.items():
                if not agent.alive:
                    actions[aid] = np.zeros(2, dtype=np.float32)
                elif policy is not None:
                    obs = ObservationBuilder.build(agent, list(agents.values()), arena)
                    act, _ = policy.predict(obs, deterministic=True)
                    actions[aid] = act
                else:
                    actions[aid] = np.random.uniform(-1, 1, 2).astype(np.float32)

            agents, _, _, _ = env.step(actions)

            # Shrink the visible stage disk to match the storm boundary.
            if safe_id >= 0:
                env.model.geom_size[safe_id, 0] = env.current_radius
            if ring_id >= 0:
                env.model.geom_size[ring_id, 0] = env.current_radius * 1.03

            # Zoom the camera in as the storm closes (with a floor so it never
            # clips through the agents).
            cam.distance = max(1.6, 3.2 * env.current_radius)
            renderer.update_scene(env.data, camera=cam)
            recorder.add_frame(renderer.render())

            if sum(a.alive for a in agents.values()) <= 1:
                break

    recorder.save()
    print(f"Saved {recorder.frame_count} frames to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Render a Battle Royale episode")
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--out", default="output.mp4")
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument(
        "--fps", type=int, default=24, help="playback fps (lower = slower / more slow-mo)"
    )
    args = parser.parse_args()
    run(args.config, args.checkpoint, args.out, args.max_steps, args.fps)
