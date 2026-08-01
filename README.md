# MuJoCo Battle Royale

A competitive multi-agent simulation where 4–8 cylindrical robot agents compete in a circular sumo-style arena. Agents are eliminated when pushed out of bounds. A single shared PPO policy is trained via self-play against a snapshot pool of past policies.

Built as a workshop project exploring competitive MARL, self-play, and generalization to unseen agent counts.

---

## What It Does

- **Physics**: MuJoCo simulates 2D dynamics (slide joints, applied forces) in a circular arena
- **Observation**: Each agent receives a 17-dimensional vector — own position/velocity, distance to boundary, and relative state of its 3 nearest live neighbors
- **Reward**: `+1.0` per elimination, `-1.0` on death, `+0.01` survival bonus per step
- **Training**: Shared PPO policy (all agents use the same weights) trained via self-play; opponents are sampled from a rolling snapshot pool
- **Generalization test**: Policy trained on 4 agents evaluated zero-shot on 6 and 8 agents

### Success Targets

| Criterion | Target |
|-----------|--------|
| Win rate vs. opponent pool | >60% |
| Elo trajectory | Monotonically increasing |
| Cross-count generalization (train=4, eval=6) | >40% win rate |

---

## Architecture

Clean Architecture with inward-only dependencies:

```
┌──────────────────────────────────────────────┐
│  interfaces/    CLI, PettingZoo wrapper       │
├──────────────────────────────────────────────┤
│  application/   Trainer, Evaluator, Metrics   │
├──────────────────────────────────────────────┤
│  infrastructure/  MuJoCo, WandB, Config       │
├──────────────────────────────────────────────┤
│  domain/        Entities, Protocols, Services │
└──────────────────────────────────────────────┘
          dependencies only point inward
```

- **Domain**: pure Python — immutable `Agent`/`Arena` dataclasses, `IBattleRoyaleEnv`/`IPolicy`/`ILogger` protocols, stateless services (`EliminationService`, `ObservationBuilder`, `RewardCalculator`)
- **Infrastructure**: concrete implementations — `MuJoCoEnvironment`, `WandBLogger`, `YamlLoader`
- **Application**: training loop, evaluation, Elo/metrics — depends only on domain protocols, never on infrastructure directly
- **Interfaces**: wires everything together. `SelfPlayEnv(gym.Env)` presents the arena to SB3 as a single-agent self-play problem — one learner vs. frozen opponents sampled from the snapshot pool. `BattleRoyaleEnv(ParallelEnv)` is the multi-agent PettingZoo view used by the evaluator.

---

## Requirements

- Python 3.12
- [Poetry](https://python-poetry.org/) 2.0+
- MuJoCo 3.6+ (installed automatically via pip)
- `ffmpeg` on `PATH` — only needed to render episode videos (`main.py`)

---

## Installation

```bash
git clone https://github.com/dvirgilad/mujoco-battle-royale.git
cd mujoco-battle-royale
poetry install
```

For development tools (linting, testing):

```bash
poetry install --with dev
poetry run pre-commit install
```

---

## Usage

### Run the physics environment

```python
from battle_royale.infrastructure.physics.mujoco_env import MuJoCoEnvironment
from battle_royale.infrastructure.config.yaml_loader import load_config
import numpy as np

config = load_config("config/default.yaml")
env = MuJoCoEnvironment(config=config)

agents = env.reset(num_agents=4)

actions = {agent_id: np.random.uniform(-1, 1, size=2) for agent_id in agents}
obs, rewards, terminations, truncations = env.step(actions)
```

### Build observations

```python
from battle_royale.domain.services.observation import ObservationBuilder
from battle_royale.domain.entities.arena import Arena

arena = Arena(radius=config.arena.radius)

for agent in agents.values():
    obs = ObservationBuilder.build(agent, list(agents.values()), arena)
    # obs.shape == (17,), dtype float32
```

### Train

Self-play: SB3 trains a single **learner** (`agent_0`); the other agents are
driven by frozen policies sampled from the snapshot pool. Win-rate and Elo are
recorded per match.

```bash
# offline (prints metrics to stdout, no WandB account needed)
python -m battle_royale.interfaces.cli.train --config config/default.yaml --run-dir runs/selfplay_4v4 --logger stdout

# with Weights & Biases
python -m battle_royale.interfaces.cli.train --config config/default.yaml --run-dir runs/selfplay_4v4 --logger wandb
```

`--logger` accepts `stdout` (offline), `wandb`, or `null`.

### Evaluate

```bash
# single agent-count, opponents drawn from the run's snapshot pool
python -m battle_royale.interfaces.cli.evaluate \
    --checkpoint runs/selfplay_4v4/snapshots/snapshot_1000000.zip \
    --run-dir runs/selfplay_4v4 --num-agents 6

# generalization sweep: train=4 -> eval on 4/6/8, prints a table
python -m battle_royale.interfaces.cli.evaluate \
    --checkpoint runs/selfplay_4v4/snapshots/snapshot_1000000.zip \
    --run-dir runs/selfplay_4v4 --sweep
```

### Render an episode

Records an episode to a video file (requires `ffmpeg` on PATH).

```bash
python main.py                                             # random agents -> output.mp4
python main.py --checkpoint runs/selfplay_4v4/snapshots/snapshot_1000000.zip
```

---

## Configuration

All parameters live in `config/default.yaml`:

```yaml
arena:
  radius: 3.0
  wall_height: 0.5

training:
  num_agents: 4
  total_steps: 1_000_000
  snapshot_interval: 10_000
  max_force: 10.0           # N, upper bound on applied force
  snapshot_pool_size: 20

ppo:
  lr: 0.0003
  n_steps: 2048
  batch_size: 64
  clip_range: 0.2
  n_epochs: 10

evaluation:
  eval_freq: 10_000
  num_episodes: 100
  generalization_agent_counts: [4, 6, 8]
```

Experiment-specific overrides go in `config/experiments/`. Example: `config/experiments/4v4_baseline.yaml`.

---

## Development

### Run tests

```bash
poetry run pytest tests/ -v
```

### Run with coverage

```bash
poetry run pytest tests/ --cov=battle_royale --cov-report=term-missing
```

### Lint

```bash
poetry run ruff check src/ tests/
poetry run ruff format src/ tests/
```

Pre-commit hooks run Ruff automatically on each commit.

---

## Project Status

| Layer | Component | Status |
|-------|-----------|--------|
| Domain | `Agent`, `Arena` entities | Done |
| Domain | `IBattleRoyaleEnv`, `IPolicy`, `ILogger` protocols | Done |
| Domain | `EliminationService`, `ObservationBuilder`, `RewardCalculator` | Done |
| Infrastructure | `MuJoCoEnvironment` | Done |
| Infrastructure | `XMLBuilder` (MJCF for N agents) | Done |
| Infrastructure | `YamlLoader` + `Config` dataclasses | Done |
| Infrastructure | `WandBLogger` | Done |
| Infrastructure | `VideoRecorder` | Done |
| Interfaces | `BattleRoyaleEnv` (PettingZoo wrapper, used by evaluator) | Done |
| Interfaces | `SelfPlayEnv` (single-agent self-play Gym env) | Done |
| Infrastructure | `NullLogger` / `StdoutLogger` (offline logging) | Done |
| Application | `EloRatingSystem` (K=32) | Done |
| Application | `MetricsTracker` (win-rate + learner-vs-pool Elo) | Done |
| Application | `SnapshotPool` (save + `discover` existing runs) | Done |
| Application | `Trainer` (SB3 PPO self-play vs snapshot pool) | Done |
| Application | `Evaluator` | Done |
| Interfaces | CLI `train` (`--logger`) + `evaluate` (`--sweep`) | Done |
| Interfaces | `main.py` episode renderer → video | Done |
| Testing | Integration test: full env loop | Done |
| Testing | Coverage gate ≥80% (118 tests) | Done |

---

## Self-Play Training Loop

```
CLI (train.py)
  → load Config
  → MuJoCoEnvironment → SelfPlayEnv (single-agent Gym view)
  → Trainer(env, logger, snapshot_pool, tracker) → SB3 PPO
      each episode (one learner-vs-pool match):
          reset() samples a frozen opponent from SnapshotPool
          per step: learner action from PPO; opponents from the frozen policy
                     ObservationBuilder → EliminationService → RewardCalculator
          on episode end → MetricsTracker.record_match()
                     → EloRatingSystem.update() (learner vs pool) → ILogger.log()
      SelfPlayCallback: SnapshotPool.save() every N steps (grows the opponent pool)
```

---

## Contributing

1. Fork the repo and create a branch from `dev`
2. Install dev dependencies: `poetry install --with dev && poetry run pre-commit install`
3. Follow TDD: write tests first, then implement
4. Open a PR against `dev` — CI runs Ruff + pytest on every PR

---

## Authors

- [@dvirgilad](https://github.com/dvirgilad)
- [@oshribelay](https://github.com/oshribelay)
