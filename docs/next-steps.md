# Handoff / Next Steps

_Last updated: 2026-08-01 (end of session)._

## Where we are

The project **runs end-to-end** and is close to submission-ready. What was done this session:

- **Local environment**: Python 3.12 + Poetry + all deps installed; `ffmpeg` installed (for video). `poetry run pytest` → **118 tests pass**, Ruff clean, coverage gate holds.
- **Real self-play implemented** (was missing before): new `SelfPlayEnv` (single-agent Gym view) trains one learner via SB3 PPO against **frozen opponents sampled from the snapshot pool**. The `MetricsTracker` (win-rate + learner-vs-pool Elo) and logger — previously dead code — are now wired into the training loop.
- **Bug fixes**: `SnapshotPool` no longer writes an empty file that broke `PPO.load`; `main.py` renderer's inverted glfw logic fixed (now records a real episode to video); env no longer relies on SuperSuit (training uses `DummyVecEnv`).
- **New capabilities**: offline loggers (`--logger stdout|null`, no WandB needed); `evaluate --sweep` produces the train=4 → eval 4/6/8 generalization table drawing opponents from the real pool.
- **Full 1M-step run completed** → `runs/selfplay_4v4/` (`snapshots/`, `trajectory.csv`, `sweep.log`, `demo.mp4`).

## The one thing to fix tomorrow: reward makes the policy "turtle"

The trained policy learned to **survive passively rather than win**, because the reward is imbalanced:

| Behavior | Reward |
|---|---|
| Turtle: survive 1000 steps | `+0.01 × 1000 = +10` |
| Win: eliminate all 3 opponents | `+3` (then die/survive) |

So PPO correctly optimizes for turtling. Episodes stalemate to the step cap (mean length ~823–990), and the post-training generalization sweep gives:

| num_agents | win_rate | mean_len |
|---|---|---|
| 4 | 0.11 | 823 |
| 6 | 0.05 | 909 |
| 8 | 0.02 | 990 |

— well below the **>60%** target. (Training-time rolling win-rate oscillated 0.2–0.74 with peak Elo ~1646, the expected Red-Queen dynamic vs a self-strengthening pool.)

This is a **design finding, not a plumbing bug** — worth writing up in the report either way.

## Proposed fix (rebalance reward so winning dominates)

Edit `src/battle_royale/domain/services/reward.py` and `tests/unit/domain/test_reward.py`:

```
elimination:  +1.0   (unchanged)
death:        -1.0   (unchanged)
survival:     +0.01  ->  +0.001
time penalty:  none  ->  -0.01 / step   (new)
```

With this, turtling to the cap nets a large negative and eliminating opponents becomes the only way to score. Then:

1. Retrain (~25 min):
   `python -m battle_royale.interfaces.cli.train --config config/default.yaml --run-dir runs/selfplay_4v4_v2 --logger stdout`
2. Re-sweep:
   `python -m battle_royale.interfaces.cli.evaluate --checkpoint runs/selfplay_4v4_v2/snapshots/snapshot_1000000.zip --run-dir runs/selfplay_4v4_v2 --sweep`

Consider also shortening `SelfPlayEnv(max_steps=...)` (default 1000) to ~300–400 to force faster resolution. **>60% is not guaranteed** — may need 1–2 tuning iterations (alternatives: explicit win bonus, higher elimination reward).

## Still to finalize after the fix

- Write up `RESULTS.md` with final trajectory + generalization table.
- Update the README "Success Targets" row with actual numbers.
- (Optional) commit the work — nothing has been committed yet this session.

## Running locally (reminder)

Default `python` is 3.8; use the 3.12 Poetry venv. In PowerShell from repo root:

```powershell
$env:PATH = "C:\Users\oshri\AppData\Local\Programs\Python\Python312\Scripts;$env:PATH"
$env:PYTHONPATH = "C:\Users\oshri\OneDrive\Documents\mujoco-battle-royale\src"  # needed for `python -m ...`
poetry run pytest tests/
```
