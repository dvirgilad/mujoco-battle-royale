"""Render a demo video for a run's latest snapshot -> <run_dir>/demo.mp4.

Usage (from repo root, with the poetry env and ffmpeg on PATH):
    python scripts/render_run.py runs/selfplay_storm_final
    python scripts/render_run.py runs/smoke_combat --config config/combat_smoke.yaml
"""

import argparse
import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from main import run  # noqa: E402


def latest_snapshot(run_dir: str) -> str | None:
    snaps = glob.glob(os.path.join(run_dir, "snapshots", "snapshot_*.zip"))
    if not snaps:
        return None
    return max(snaps, key=lambda p: int(p.split("_")[-1].split(".")[0]))


def main(run_dir: str, config_path: str, max_steps: int) -> None:
    ckpt = latest_snapshot(run_dir)
    if ckpt is None:
        print(f"No snapshots found in {run_dir}/snapshots")
        return
    out = os.path.join(run_dir, "demo.mp4")
    print(f"Rendering {os.path.basename(ckpt)} -> {out}")
    run(config_path, ckpt, out, max_steps)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Render a run's latest snapshot to video")
    p.add_argument("run_dir")
    p.add_argument("--config", default="config/default.yaml")
    p.add_argument("--max-steps", type=int, default=500)
    a = p.parse_args()
    main(a.run_dir, a.config, a.max_steps)
