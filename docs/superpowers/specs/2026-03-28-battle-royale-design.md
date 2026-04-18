# Battle Royale Multi-Agent Simulation — Design Spec

**Date:** 2026-03-28
**Status:** Approved

---

## Context

This project implements a competitive multi-agent simulation where 4–8 cylindrical robot agents compete in a circular sumo-style arena using MuJoCo physics. Agents are eliminated when pushed out of the arena boundary. A shared PPO policy is trained via self-play against a snapshot pool of past policies. The system must follow Clean Architecture and SOLID principles and meet the evaluation criteria from the workshop design document (win rate >60% vs. opponent pool, Elo convergence, cross-agent-count generalization).

---

## Architecture: Clean Architecture with Inward-Only Dependencies

```
┌────────────────────────────────────────────┐
│  interfaces/  (CLI, PettingZoo wrapper)     │
├────────────────────────────────────────────┤
│  application/ (training, eval, metrics)     │
├────────────────────────────────────────────┤
│  infrastructure/ (MuJoCo, WandB, video)     │
├────────────────────────────────────────────┤
│  domain/  (entities, services, protocols)   │
└────────────────────────────────────────────┘
         dependencies only point inward
```

- **Domain** has zero external imports — pure Python dataclasses, protocols, stateless services.
- **Infrastructure** implements domain protocols using MuJoCo, WandB, etc.
- **Application** depends only on domain protocols, never on concrete infrastructure.
- **Interfaces** wires everything together via dependency injection at startup.

---

## File Structure

```
mujoco-battle-royale/
├── config/
│   ├── default.yaml                  # All tuneable params (arena, agents, PPO, self-play)
│   └── experiments/
│       └── 4v4_baseline.yaml         # Experiment-specific overrides
├── src/
│   └── battle_royale/
│       ├── domain/
│       │   ├── entities/
│       │   │   ├── agent.py          # Agent dataclass (id, pos, vel, alive)
│       │   │   └── arena.py          # Arena dataclass (radius)
│       │   ├── interfaces/
│       │   │   ├── environment.py    # IBattleRoyaleEnv protocol
│       │   │   ├── policy.py         # IPolicy protocol
│       │   │   └── logger.py         # ILogger protocol
│       │   └── services/
│       │       ├── elimination.py    # EliminationService
│       │       ├── observation.py    # ObservationBuilder
│       │       └── reward.py         # RewardCalculator
│       ├── application/
│       │   ├── training/
│       │   │   ├── trainer.py        # Self-play training loop
│       │   │   └── snapshot_pool.py  # Policy snapshot manager
│       │   ├── evaluation/
│       │   │   └── evaluator.py      # Win rate + generalization evaluation
│       │   └── metrics/
│       │       ├── elo.py            # EloRatingSystem (K=32)
│       │       └── tracker.py        # MetricsTracker
│       ├── infrastructure/
│       │   ├── physics/
│       │   │   ├── mujoco_env.py     # Implements IBattleRoyaleEnv via MuJoCo
│       │   │   └── xml_builder.py    # Generates MJCF XML for N cylinder agents
│       │   ├── logging/
│       │   │   └── wandb_logger.py   # Implements ILogger via WandB
│       │   ├── recording/
│       │   │   └── video_recorder.py # Renders episodes to MP4
│       │   └── config/
│       │       └── yaml_loader.py    # Loads YAML into typed Config dataclass
│       └── interfaces/
│           ├── pettingzoo/
│           │   └── env.py            # PettingZoo ParallelEnv wrapper
│           └── cli/
│               ├── train.py          # python -m battle_royale.train
│               └── evaluate.py       # python -m battle_royale.evaluate
├── tests/
│   ├── unit/
│   │   ├── domain/                   # Pure Python tests, no MuJoCo required
│   │   └── application/
│   └── integration/
│       └── test_environment.py       # Full env step tests with MuJoCo
├── docs/
│   └── superpowers/specs/
├── main.py                           # Existing render script (retained)
└── pyproject.toml
```

---

## Domain Layer

### Entities

```python
# domain/entities/agent.py
@dataclass(frozen=True)
class Agent:
    id: str
    position: np.ndarray   # (x, y) in arena coordinates
    velocity: np.ndarray   # (vx, vy)
    alive: bool

# domain/entities/arena.py
@dataclass(frozen=True)
class Arena:
    radius: float
```

Entities are immutable dataclasses with no methods and no external dependencies.

### Protocols

```python
# domain/interfaces/environment.py
class IBattleRoyaleEnv(Protocol):
    def reset(self, num_agents: int) -> dict[str, Agent]: ...
    def step(self, actions: dict[str, np.ndarray]) -> tuple[dict, dict, dict, dict]: ...
    def get_agents(self) -> list[Agent]: ...

# domain/interfaces/logger.py
class ILogger(Protocol):
    def log(self, metrics: dict[str, float], step: int) -> None: ...
    def save_artifact(self, path: str, name: str) -> None: ...

# domain/interfaces/policy.py
class IPolicy(Protocol):
    def predict(self, obs: np.ndarray) -> np.ndarray: ...
    def save(self, path: str) -> None: ...
    @classmethod
    def load(cls, path: str) -> "IPolicy": ...
```

### Domain Services

All services are stateless — inputs in, outputs out, no side effects.

**`EliminationService`**
- `is_eliminated(agent: Agent, arena: Arena) -> bool`
- Logic: `np.linalg.norm(agent.position) > arena.radius`

**`ObservationBuilder`**
- `build(agent: Agent, all_agents: list[Agent], arena: Arena) -> np.ndarray`
- Returns a fixed-size **17-dimensional** vector regardless of total agent count:
  - Own position (2), own velocity (2), distance to boundary (1)
  - 3 nearest live neighbors: relative position + velocity each (4 × 3 = 12)
  - Missing neighbors padded with zeros

**`RewardCalculator`**
- `compute(prev_agents: list[Agent], curr_agents: list[Agent], agent_id: str) -> float`
- `+1.0` for each opponent eliminated this step
- `-1.0` if this agent was eliminated this step
- `+0.01` survival bonus per step (encourages staying alive, not just avoiding engagement)

---

## Infrastructure Layer

### `xml_builder.py`

Generates MJCF XML programmatically for N agents:
- **Arena**: flat floor plane + thin cylindrical wall at `arena.radius`
- **Agents**: cylinder bodies (`radius=0.15, height=0.1`), each with 2 slide joints (x, y) and 1 actuator per axis
- **Initial placement**: agents spawned at evenly-spaced angles around a circle at `0.6 * arena.radius`

### `mujoco_env.py`

Implements `IBattleRoyaleEnv`:
1. Calls `XMLBuilder.build(num_agents, arena)` to generate model string
2. Loads via `mj.MjModel.from_xml_string()`
3. `step()`: applies actions as forces (scaled from `[-1,1]` to `[-max_force, max_force]` N, where `max_force` is set in YAML) → `mj_step()` → extracts positions/velocities → runs `EliminationService` → for eliminated agents: zero their velocity and lock position via `mjData.qvel` and `mjData.qacc` zeroing each step
4. Uses headless EGL rendering (same pattern as existing `main.py`)

### `wandb_logger.py`

Implements `ILogger`. Routes `log()` to `wandb.log()` and `save_artifact()` to WandB artifacts for checkpoints and videos.

### `yaml_loader.py`

Loads YAML into a typed `Config` dataclass tree:

```python
@dataclass
class Config:
    arena: ArenaConfig        # radius, wall_height
    training: TrainingConfig  # num_agents, total_steps, snapshot_interval
    ppo: PPOConfig            # lr, n_steps, batch_size, clip_range, ...
    evaluation: EvalConfig    # eval_freq, num_episodes, generalization_agent_counts
```

---

## Application Layer

### `SnapshotPool`

- Stores past SB3 policy checkpoints on disk under `runs/snapshots/`
- `save(policy, step)` — serializes current policy
- `sample(n) -> list[IPolicy]` — randomly returns n policies from pool
- Used to provide opponents: 80% sampled from pool, 20% latest policy

### `Trainer`

Owns the self-play training loop:
1. Constructs `BattleRoyaleEnv` (PettingZoo) and wraps with SB3's `VecEnv` adapter
2. Initializes shared PPO policy (one network, all agents share weights)
3. Every `snapshot_interval` steps: calls `SnapshotPool.save()`, `MetricsTracker.log()`
4. Opponent policies are swapped each episode via snapshot sampling

### `Evaluator`

- Runs `num_episodes` evaluation episodes
- Reports: win rate vs. opponent pool, mean survival time
- Cross-generalization test: loads a policy trained on N agents, evaluates on M agents (e.g., train=4, test=6)

### `MetricsTracker`

- Aggregates win rate and episode outcomes
- Updates `EloRatingSystem` after each episode (K=32 standard update)
- Calls `ILogger.log()` with all metrics — no direct WandB dependency

---

## Interfaces Layer

### `pettingzoo/env.py` — `BattleRoyaleEnv(ParallelEnv)`

- Wraps injected `IBattleRoyaleEnv`
- `observation_space`: `Box(shape=(17,), dtype=float32)` per agent
- `action_space`: `Box(low=-1, high=1, shape=(2,), dtype=float32)` per agent (normalized x/y force, scaled to `[-max_force, max_force]` N inside `mujoco_env.py`)
- Manages PettingZoo agent lifecycle: `.agents`, `.terminations`, `.truncations`

### CLI

```bash
# Training
python -m battle_royale.train --config config/experiments/4v4_baseline.yaml

# Evaluation
python -m battle_royale.evaluate --checkpoint runs/checkpoint_100k --num-agents 6
```

Both CLIs: load YAML config → instantiate infrastructure → inject into application layer → run.

---

## Data Flow

```
CLI (train.py)
  → loads Config (YAMLConfigLoader)
  → constructs MuJoCoEnvironment(config.arena)
  → constructs BattleRoyaleEnv(mujoco_env) [PettingZoo]
  → constructs WandBLogger, SnapshotPool, MetricsTracker
  → constructs Trainer(env, logger, snapshot_pool, tracker, config.training)
  → Trainer.run()
       → episode loop:
           env.reset()
           step loop:
               ObservationBuilder.build() per agent
               policy.predict(obs) per agent
               env.step(actions)
               EliminationService.is_eliminated() per agent
               RewardCalculator.compute() per agent
           MetricsTracker.update(episode_result)
           EloRatingSystem.update()
           ILogger.log(metrics)
           SnapshotPool.save() [every N steps]
```

---

## Testing Strategy

- **Unit tests** (`tests/unit/domain/`): test `EliminationService`, `ObservationBuilder`, `RewardCalculator`, `EloRatingSystem` with plain Python — no MuJoCo needed
- **Unit tests** (`tests/unit/application/`): test `SnapshotPool`, `MetricsTracker` with mock `IPolicy` and `ILogger`
- **Integration tests** (`tests/integration/`): spin up `MuJoCoEnvironment` with 4 agents, run 10 steps, assert shapes and no crashes

---

## Key Dependencies to Add

```toml
pettingzoo = ">=1.24"
stable-baselines3 = ">=2.0"
wandb = ">=0.17"
gymnasium = ">=0.29"
pyyaml = ">=6.0"
```

---

## Success Criteria (from design doc)

| Criterion | Target |
|---|---|
| Win rate vs. opponent pool | >60% |
| Elo trajectory | Monotonically increasing |
| Cross-agent-count generalization | >40% win rate (train=4, test=6) |
| Reproducibility | Fixed seed, <15 min setup |
| Code quality | SOLID-compliant, Clean Architecture |
