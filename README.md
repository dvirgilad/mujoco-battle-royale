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
- **Interfaces**: wires everything together; `BattleRoyaleEnv(ParallelEnv)` exposes the env to SB3 via PettingZoo

---

## Requirements

- Python 3.12
- [Poetry](https://python-poetry.org/) 2.0+
- MuJoCo 3.6+ (installed automatically via pip)

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

> **Note:** The CLI (`train` / `evaluate` commands) is not yet implemented. Current usage is via the Python API directly.

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

### Legacy renderer

```bash
python main.py  # renders a 4-agent episode using the existing MuJoCo viewer
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
| Infrastructure | `VideoRecorder` | In progress |
| Interfaces | `BattleRoyaleEnv` (PettingZoo wrapper) | In progress |
| Application | `EloRatingSystem` (K=32) | Planned |
| Application | `MetricsTracker` | Planned |
| Application | `SnapshotPool` | Planned |
| Application | `Trainer` (SB3 PPO self-play) | Planned |
| Application | `Evaluator` | Planned |
| Interfaces | CLI (`train`, `evaluate`) | Planned |
| Testing | Integration test: full env loop | Planned |
| Testing | Coverage gate ≥80% | Planned |

---

## Planned: Self-Play Training Loop

When complete, training will follow this flow:

```
CLI (train.py)
  → load Config
  → MuJoCoEnvironment → BattleRoyaleEnv (PettingZoo)
  → Trainer(env, logger, snapshot_pool, tracker)
      → episode loop:
          env.reset()
          per step: ObservationBuilder → policy.predict() → env.step()
          EliminationService / RewardCalculator applied each step
          MetricsTracker.update() → EloRatingSystem.update() → ILogger.log()
          SnapshotPool.save() every N steps
```

Training command (once implemented):
```bash
python -m battle_royale.train --config config/experiments/4v4_baseline.yaml
```

Evaluation command:
```bash
python -m battle_royale.evaluate --checkpoint runs/checkpoint_100k --num-agents 6
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
