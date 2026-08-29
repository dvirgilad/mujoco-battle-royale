# Runbook — install, train, evaluate, render

Everything you need to reproduce the project from a clean checkout: set up the
environment, train the headline policy, measure the three result behaviours, and
render videos. Commands are shown for a POSIX shell; on Windows PowerShell they
are identical except for line-continuation.

> **TL;DR — reproduce the headline result**
> ```bash
> poetry install
> python -m battle_royale.interfaces.cli.train  --config config/sumo_light.yaml --run-dir runs/sumo_light --logger stdout
> python -m battle_royale.interfaces.cli.evaluate --checkpoint runs/sumo_light/snapshots/snapshot_001500000.zip --run-dir runs/sumo_light --sweep
> python main.py --config config/sumo_light.yaml --checkpoint runs/sumo_light/snapshots/snapshot_001500000.zip --out media/demo_n4.mp4 --num-agents 4
> ```

---

## 1. Requirements

| | |
|---|---|
| Python | 3.12 |
| Package manager | [Poetry](https://python-poetry.org/) 2.0+ |
| Physics | MuJoCo 3.6+ (installed automatically by `poetry install`) |
| Video | `ffmpeg` on `PATH` — only needed for `main.py` renders |
| Compute | CPU is enough — ~1.5M steps ≈ 30–40 min on a modern laptop CPU |

No GPU and no external pretrained models are required; every policy is trained
from scratch.

## 2. Install

```bash
git clone https://github.com/dvirgilad/mujoco-battle-royale.git
cd mujoco-battle-royale
poetry install                 # runtime deps
poetry install --with dev      # + linting/testing (optional)
```

All commands below assume the Poetry environment. Either prefix with
`poetry run`, or activate the shell once (`poetry env activate` / `poetry shell`).

> **PYTHONPATH note.** The package lives under `src/`. The CLIs and `main.py`
> add `src/` to the path themselves, so `python -m battle_royale.…` and
> `python main.py` work from the repo root without setting `PYTHONPATH`.

## 3. Train

Self-play trains **one** learner (`agent_0`); the other N−1 agents are driven by
frozen policies sampled from a rolling snapshot pool. Each episode is one
`learner-vs-field` match, which is what makes win-rate and Elo measurable.

```bash
# headline policy (light agents + draw=loss + curriculum, no storm)
python -m battle_royale.interfaces.cli.train \
    --config config/sumo_light.yaml --run-dir runs/sumo_light --logger stdout
```

- `--logger` = `stdout` (offline, prints metrics — no account needed), `wandb`, or `null`.
- Snapshots are written to `runs/<name>/snapshots/snapshot_<step>.zip` every
  `training.snapshot_interval` steps; the final checkpoint is
  `snapshot_001500000.zip` for `sumo_light`.
- Training is resumable-by-pool: `SnapshotPool.discover` picks up existing
  snapshots in the run dir.

**Config knobs that matter** (all in `config/sumo_light.yaml`, see
[`METHODOLOGY.md`](METHODOLOGY.md) for the maths behind each):

| Knob | Value | Effect |
|---|---|---|
| `arena.agent_density` | `150` | light agents → braking distance < arena (the physics fix) |
| `arena.min_radius_frac` | `1.0` | no storm — pushing is the *only* way to win |
| `training.draw_penalty` | `1.0` | a timed-out draw is scored like a loss |
| `training.curriculum_start_agents` | `2` | learn the shove 1v1, carry it into the melee |
| `ppo.gamma` | `0.997` | terminal draw penalty reaches back into the mid-game |

## 4. Evaluate — the three result behaviours

```bash
CKPT=runs/sumo_light/snapshots/snapshot_001500000.zip

# generalisation sweep: prints a table for n = 4 / 6 / 8
python -m battle_royale.interfaces.cli.evaluate --checkpoint $CKPT --run-dir runs/sumo_light --sweep

# single agent-count, opponents from the run's snapshot pool
python -m battle_royale.interfaces.cli.evaluate --checkpoint $CKPT --run-dir runs/sumo_light --num-agents 6

# the three headline KPIs (active pushing / balance / dominance) at n = 4/6/8
python scripts/eval_kpis.py --checkpoint $CKPT --config config/sumo_light.yaml --counts 4,6,8 --episodes 100
```

What each headline number means:

- **Active pushing** — self-play *resolution rate* with no storm. Every resolution
  is a genuine push-out, so a high rate = agents actively eject each other.
- **Dominance** — 1 trained vs N−1 random, reported as `edge = WIN − LOSS`.
- **Balance** — N identical copies play; the per-slot win share should be ≈ 1/N.

## 5. Render videos

`main.py` drives an episode and writes an MP4 (needs `ffmpeg`). The collision
cylinder is invisible; a visual-only humanoid (zero mass, no collision) is drawn
over it, so rendering never changes the physics. Each agent has a fixed colour
(`agent_0` = red).

```bash
CKPT=runs/sumo_light/snapshots/snapshot_001500000.zip

# self-play push-out demo at n=4 (all agents run the trained policy)
python main.py --config config/sumo_light.yaml --checkpoint $CKPT --out media/demo_n4.mp4 --num-agents 4

# zero-shot generalisation at n=6
python main.py --config config/sumo_light.yaml --checkpoint $CKPT --out media/demo_n6.mp4 --num-agents 6

# dominance demo: only red runs the policy, the other 3 act randomly
python main.py --config config/sumo_light.yaml --checkpoint $CKPT --out media/demo_dom.mp4 \
    --num-agents 4 --dominance --seed 2
```

Flags: `--num-agents` overrides the count, `--dominance` runs only `agent_0` on
the policy (rest random), `--seed` fixes the episode, `--fps` sets playback speed,
`--max-steps` caps length.

### Rebuild the highlight reel

The captioned results reel `media/battle_royale_results.mp4` is assembled from the
three clips above plus title/results cards. The generator scripts live in the
session scratchpad; to rebuild, render the three clips and stitch with `ffmpeg`
(title cards → `concat` demuxer). See `docs/METHODOLOGY.md` for the exact result
numbers shown on the cards.

## 6. Tests & linting

```bash
poetry run pytest tests/ -v                                   # full suite
poetry run pytest tests/ --cov=battle_royale --cov-report=term-missing   # coverage (≥80% gate)
poetry run ruff check src/ tests/ && poetry run ruff format --check src/ tests/
```

## 7. Troubleshooting

| Symptom | Fix |
|---|---|
| `ModuleNotFoundError: battle_royale` | run from the repo root, or `poetry run`; the CLIs add `src/` to the path |
| `ffmpeg not found` when rendering | install ffmpeg and put it on `PATH` (only `main.py` needs it) |
| Video is empty / 0 frames | the episode ended immediately — check the checkpoint path is a real `.zip` |
| Eval numbers look random | make sure `--run-dir` points at the run whose snapshot pool you want as opponents |
| Want the *original* heavy-agent physics | set `arena.agent_density: 1000` (the default) — reproduces the pre-fix behaviour |
