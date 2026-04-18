# MuJoCo Battle Royale — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a multi-agent sumo battle royale simulation with MuJoCo physics, PettingZoo + SB3 PPO training, self-play snapshot pool, WandB metrics, and Clean Architecture.

**Architecture:** Clean Architecture with inward-only dependencies: Domain (pure Python entities/protocols/services) → Infrastructure (MuJoCo, WandB, YAML) → Application (training, evaluation, metrics) → Interfaces (PettingZoo ParallelEnv, CLI). All layers communicate via domain protocols injected at startup.

**Tech Stack:** Python 3.12, MuJoCo 3.6+, PettingZoo, Stable-Baselines3 PPO, SuperSuit, WandB, PyYAML, pytest

---


## Task 1 — Project Setup: Add Missing Dependencies and Pytest Config

### Steps

- [ ] **1.1** Add missing runtime and dev dependencies:

```bash
poetry add pettingzoo stable-baselines3 wandb gymnasium pyyaml supersuit
poetry add --group dev pytest pytest-cov
```

Expected output: Poetry resolves and installs all packages. The virtualenv gains `pettingzoo`, `stable_baselines3`, `wandb`, `gymnasium`, `supersuit`, `pytest`.

- [ ] **1.2** Add pytest configuration to `pyproject.toml` by appending:

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["src"]
```

- [ ] **1.3** Verify the test suite can be discovered with no tests yet:

```bash
poetry run pytest --collect-only
```

Expected output:
```
collected 0 items
```
No errors.

- [ ] **1.4** Verify `main.py` still works (smoke test, no assertion):

```bash
poetry run python3 -c "from src.battle_royale.infrastructure.physics import load_model, load_data; load_model(); print('ok')"
```

Expected output: `ok`

- [ ] **1.5** Commit:

```bash
git add pyproject.toml poetry.lock
git commit -m "chore: add pettingzoo, sb3, wandb, gymnasium, supersuit, pytest dependencies"
```

---

## Task 2 — Domain Entities: `Agent` and `Arena`

### Steps

- [ ] **2.1** Write the failing test at `tests/unit/domain/test_entities.py`:

```python
import numpy as np
import pytest
from battle_royale.domain.entities.agent import Agent
from battle_royale.domain.entities.arena import Arena


def test_agent_is_frozen():
    agent = Agent(id="a0", position=np.array([1.0, 0.0]), velocity=np.array([0.0, 0.5]), alive=True)
    with pytest.raises((AttributeError, TypeError)):
        agent.alive = False  # type: ignore


def test_agent_fields():
    pos = np.array([1.5, -2.0])
    vel = np.array([0.1, 0.2])
    agent = Agent(id="a1", position=pos, velocity=vel, alive=True)
    assert agent.id == "a1"
    np.testing.assert_array_equal(agent.position, pos)
    np.testing.assert_array_equal(agent.velocity, vel)
    assert agent.alive is True


def test_arena_is_frozen():
    arena = Arena(radius=3.0)
    with pytest.raises((AttributeError, TypeError)):
        arena.radius = 5.0  # type: ignore


def test_arena_fields():
    arena = Arena(radius=3.0)
    assert arena.radius == 3.0
```

- [ ] **2.2** Run, expect failure:

```bash
poetry run pytest tests/unit/domain/test_entities.py -v
```

Expected failure:
```
ModuleNotFoundError: No module named 'battle_royale.domain.entities.agent'
```

- [ ] **2.3** Implement `src/battle_royale/domain/entities/agent.py`:

```python
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class Agent:
    id: str
    position: np.ndarray
    velocity: np.ndarray
    alive: bool
```

- [ ] **2.4** Implement `src/battle_royale/domain/entities/arena.py`:

```python
from dataclasses import dataclass


@dataclass(frozen=True)
class Arena:
    radius: float
```

- [ ] **2.5** Update `src/battle_royale/domain/entities/__init__.py`:

```python
from .agent import Agent
from .arena import Arena
```

- [ ] **2.6** Run, expect pass:

```bash
poetry run pytest tests/unit/domain/test_entities.py -v
```

Expected:
```
4 passed in 0.xxs
```

Note: frozen dataclasses with mutable numpy arrays will pass the attribute-assignment test because `frozen=True` blocks `__setattr__`. The arrays themselves are mutable internally but the attribute cannot be rebound.

- [ ] **2.7** Commit:

```bash
git add src/battle_royale/domain/entities/agent.py src/battle_royale/domain/entities/arena.py src/battle_royale/domain/entities/__init__.py tests/unit/domain/test_entities.py
git commit -m "feat(domain): add frozen Agent and Arena dataclass entities"
```

---

## Task 3 — Domain Interfaces (Protocols)

### Steps

- [ ] **3.1** Write the failing test at `tests/unit/domain/test_interfaces.py`:

```python
import numpy as np
from typing import runtime_checkable
from battle_royale.domain.interfaces.environment import IBattleRoyaleEnv
from battle_royale.domain.interfaces.logger import ILogger
from battle_royale.domain.interfaces.policy import IPolicy


def test_environment_protocol_is_protocol():
    # Protocols should be importable and have the required method names
    assert hasattr(IBattleRoyaleEnv, "reset")
    assert hasattr(IBattleRoyaleEnv, "step")
    assert hasattr(IBattleRoyaleEnv, "get_agents")


def test_logger_protocol_has_required_methods():
    assert hasattr(ILogger, "log")
    assert hasattr(ILogger, "save_artifact")


def test_policy_protocol_has_required_methods():
    assert hasattr(IPolicy, "predict")
    assert hasattr(IPolicy, "save")
```

- [ ] **3.2** Run, expect failure:

```bash
poetry run pytest tests/unit/domain/test_interfaces.py -v
```

Expected failure:
```
ModuleNotFoundError: No module named 'battle_royale.domain.interfaces.environment'
```

- [ ] **3.3** Implement `src/battle_royale/domain/interfaces/environment.py`:

```python
from typing import Protocol
import numpy as np
from battle_royale.domain.entities.agent import Agent


class IBattleRoyaleEnv(Protocol):
    def reset(self, num_agents: int) -> dict[str, Agent]: ...
    def step(self, actions: dict[str, np.ndarray]) -> tuple[dict, dict, dict, dict]: ...
    def get_agents(self) -> list[Agent]: ...
```

- [ ] **3.4** Implement `src/battle_royale/domain/interfaces/logger.py`:

```python
from typing import Protocol


class ILogger(Protocol):
    def log(self, metrics: dict[str, float], step: int) -> None: ...
    def save_artifact(self, path: str, name: str) -> None: ...
```

- [ ] **3.5** Implement `src/battle_royale/domain/interfaces/policy.py`:

```python
from typing import Protocol
import numpy as np


class IPolicy(Protocol):
    def predict(self, obs: np.ndarray) -> np.ndarray: ...
    def save(self, path: str) -> None: ...
```

- [ ] **3.6** Update `src/battle_royale/domain/interfaces/__init__.py`:

```python
from .environment import IBattleRoyaleEnv
from .logger import ILogger
from .policy import IPolicy
```

- [ ] **3.7** Run, expect pass:

```bash
poetry run pytest tests/unit/domain/test_interfaces.py -v
```

Expected:
```
3 passed in 0.xxs
```

- [ ] **3.8** Commit:

```bash
git add src/battle_royale/domain/interfaces/ tests/unit/domain/test_interfaces.py
git commit -m "feat(domain): add IBattleRoyaleEnv, ILogger, IPolicy protocols"
```

---

## Task 4 — Domain Service: `EliminationService`

### Steps

- [ ] **4.1** Write the failing test at `tests/unit/domain/test_elimination.py`:

```python
import numpy as np
import pytest
from battle_royale.domain.entities.agent import Agent
from battle_royale.domain.entities.arena import Arena
from battle_royale.domain.services.elimination import EliminationService


@pytest.fixture
def arena():
    return Arena(radius=3.0)


def make_agent(x: float, y: float, alive: bool = True) -> Agent:
    return Agent(id="a0", position=np.array([x, y]), velocity=np.zeros(2), alive=alive)


def test_agent_inside_arena_not_eliminated(arena):
    agent = make_agent(1.0, 0.0)
    assert EliminationService.is_eliminated(agent, arena) is False


def test_agent_exactly_on_boundary_not_eliminated(arena):
    # norm == radius → not eliminated (strictly greater than)
    agent = make_agent(3.0, 0.0)
    assert EliminationService.is_eliminated(agent, arena) is False


def test_agent_outside_arena_eliminated(arena):
    agent = make_agent(3.1, 0.0)
    assert EliminationService.is_eliminated(agent, arena) is True


def test_dead_agent_considered_eliminated_regardless(arena):
    agent = Agent(id="a0", position=np.array([0.0, 0.0]), velocity=np.zeros(2), alive=False)
    assert EliminationService.is_eliminated(agent, arena) is True


def test_diagonal_position_eliminated(arena):
    # sqrt(2^2 + 2^2) = 2.828 < 3.0 → not eliminated
    agent = make_agent(2.0, 2.0)
    assert EliminationService.is_eliminated(agent, arena) is False


def test_diagonal_position_outside(arena):
    # sqrt(2.2^2 + 2.2^2) = 3.111 > 3.0 → eliminated
    agent = make_agent(2.2, 2.2)
    assert EliminationService.is_eliminated(agent, arena) is True
```

- [ ] **4.2** Run, expect failure:

```bash
poetry run pytest tests/unit/domain/test_elimination.py -v
```

Expected failure:
```
ModuleNotFoundError: No module named 'battle_royale.domain.services.elimination'
```

- [ ] **4.3** Implement `src/battle_royale/domain/services/elimination.py`:

```python
import numpy as np
from battle_royale.domain.entities.agent import Agent
from battle_royale.domain.entities.arena import Arena


class EliminationService:
    @staticmethod
    def is_eliminated(agent: Agent, arena: Arena) -> bool:
        if not agent.alive:
            return True
        return float(np.linalg.norm(agent.position)) > arena.radius
```

- [ ] **4.4** Run, expect pass:

```bash
poetry run pytest tests/unit/domain/test_elimination.py -v
```

Expected:
```
6 passed in 0.xxs
```

- [ ] **4.5** Commit:

```bash
git add src/battle_royale/domain/services/elimination.py tests/unit/domain/test_elimination.py
git commit -m "feat(domain): add EliminationService with boundary detection"
```

---

## Task 5 — Domain Service: `ObservationBuilder`

### Steps

- [ ] **5.1** Write the failing test at `tests/unit/domain/test_observation.py`:

```python
import numpy as np
import pytest
from battle_royale.domain.entities.agent import Agent
from battle_royale.domain.entities.arena import Arena
from battle_royale.domain.services.observation import ObservationBuilder


def make_agent(agent_id: str, x: float, y: float, vx: float = 0.0, vy: float = 0.0, alive: bool = True) -> Agent:
    return Agent(id=agent_id, position=np.array([x, y]), velocity=np.array([vx, vy]), alive=alive)


@pytest.fixture
def arena():
    return Arena(radius=3.0)


def test_observation_shape_four_agents(arena):
    agents = [make_agent(f"a{i}", float(i) * 0.5, 0.0) for i in range(4)]
    obs = ObservationBuilder.build(agents[0], agents, arena)
    assert obs.shape == (17,)
    assert obs.dtype == np.float32


def test_observation_own_position_first_two_elements(arena):
    agent = make_agent("a0", 1.5, -0.5)
    agents = [agent, make_agent("a1", 0.0, 0.0), make_agent("a2", 1.0, 1.0), make_agent("a3", -1.0, 0.5)]
    obs = ObservationBuilder.build(agent, agents, arena)
    np.testing.assert_allclose(obs[0], 1.5, atol=1e-5)
    np.testing.assert_allclose(obs[1], -0.5, atol=1e-5)


def test_observation_own_velocity_elements_2_3(arena):
    agent = make_agent("a0", 0.0, 0.0, vx=2.0, vy=-1.0)
    agents = [agent, make_agent("a1", 0.5, 0.0), make_agent("a2", -0.5, 0.0), make_agent("a3", 0.0, 0.5)]
    obs = ObservationBuilder.build(agent, agents, arena)
    np.testing.assert_allclose(obs[2], 2.0, atol=1e-5)
    np.testing.assert_allclose(obs[3], -1.0, atol=1e-5)


def test_observation_dist_to_boundary_element_4(arena):
    # agent at (1, 0): dist_to_boundary = 3.0 - 1.0 = 2.0
    agent = make_agent("a0", 1.0, 0.0)
    agents = [agent, make_agent("a1", 0.5, 0.0), make_agent("a2", -0.5, 0.0), make_agent("a3", 0.0, 0.5)]
    obs = ObservationBuilder.build(agent, agents, arena)
    np.testing.assert_allclose(obs[4], 2.0, atol=1e-5)


def test_observation_three_nearest_neighbors_by_distance(arena):
    # a0 at origin; a1 closest, a2 middle, a3 furthest, a4 even further
    agent = make_agent("a0", 0.0, 0.0)
    a1 = make_agent("a1", 0.1, 0.0)   # nearest
    a2 = make_agent("a2", 0.5, 0.0)   # 2nd
    a3 = make_agent("a3", 1.0, 0.0)   # 3rd
    a4 = make_agent("a4", 2.0, 0.0)   # 4th — should be excluded
    all_agents = [agent, a1, a2, a3, a4]
    obs = ObservationBuilder.build(agent, all_agents, arena)
    # First neighbor rel_pos should be (0.1, 0.0)
    np.testing.assert_allclose(obs[5], 0.1, atol=1e-5)
    np.testing.assert_allclose(obs[6], 0.0, atol=1e-5)
    # Third neighbor rel_pos x should be 1.0
    np.testing.assert_allclose(obs[9], 1.0, atol=1e-5)


def test_observation_padded_with_zeros_when_fewer_than_three_neighbors(arena):
    agent = make_agent("a0", 0.0, 0.0)
    other = make_agent("a1", 1.0, 0.0)
    obs = ObservationBuilder.build(agent, [agent, other], arena)
    assert obs.shape == (17,)
    # Slots for neighbors 2 and 3 (indices 9..16) should all be zero
    np.testing.assert_array_equal(obs[9:], np.zeros(8, dtype=np.float32))
```

- [ ] **5.2** Run, expect failure:

```bash
poetry run pytest tests/unit/domain/test_observation.py -v
```

Expected failure:
```
ModuleNotFoundError: No module named 'battle_royale.domain.services.observation'
```

- [ ] **5.3** Implement `src/battle_royale/domain/services/observation.py`:

```python
import numpy as np
from battle_royale.domain.entities.agent import Agent
from battle_royale.domain.entities.arena import Arena

_NUM_NEIGHBORS = 3
_OBS_DIM = 17  # 2 (pos) + 2 (vel) + 1 (dist_boundary) + 3 * 4 (neighbor rel_pos + rel_vel)


class ObservationBuilder:
    @staticmethod
    def build(agent: Agent, all_agents: list[Agent], arena: Arena) -> np.ndarray:
        obs = np.zeros(_OBS_DIM, dtype=np.float32)
        obs[0:2] = agent.position
        obs[2:4] = agent.velocity
        obs[4] = arena.radius - float(np.linalg.norm(agent.position))

        others = [a for a in all_agents if a.id != agent.id]
        others_sorted = sorted(
            others,
            key=lambda a: float(np.linalg.norm(a.position - agent.position)),
        )
        neighbors = others_sorted[:_NUM_NEIGHBORS]

        for i, neighbor in enumerate(neighbors):
            base = 5 + i * 4
            obs[base:base + 2] = neighbor.position - agent.position
            obs[base + 2:base + 4] = neighbor.velocity - agent.velocity

        return obs
```

- [ ] **5.4** Run, expect pass:

```bash
poetry run pytest tests/unit/domain/test_observation.py -v
```

Expected:
```
7 passed in 0.xxs
```

- [ ] **5.5** Commit:

```bash
git add src/battle_royale/domain/services/observation.py tests/unit/domain/test_observation.py
git commit -m "feat(domain): add ObservationBuilder producing 17-dim agent observations"
```

---

## Task 6 — Domain Service: `RewardCalculator`

### Steps

- [ ] **6.1** Write the failing test at `tests/unit/domain/test_reward.py`:

```python
import numpy as np
import pytest
from battle_royale.domain.entities.agent import Agent
from battle_royale.domain.services.reward import RewardCalculator


def make_agent(agent_id: str, alive: bool) -> Agent:
    return Agent(id=agent_id, position=np.zeros(2), velocity=np.zeros(2), alive=alive)


def test_survival_reward_when_nothing_changes():
    prev = {"a0": make_agent("a0", True), "a1": make_agent("a1", True)}
    curr = {"a0": make_agent("a0", True), "a1": make_agent("a1", True)}
    reward = RewardCalculator.compute(prev, curr, "a0")
    assert pytest.approx(reward, abs=1e-6) == 0.01


def test_reward_for_eliminating_one_opponent():
    prev = {"a0": make_agent("a0", True), "a1": make_agent("a1", True), "a2": make_agent("a2", True)}
    curr = {"a0": make_agent("a0", True), "a1": make_agent("a1", False), "a2": make_agent("a2", True)}
    reward = RewardCalculator.compute(prev, curr, "a0")
    # +1 for elimination + 0.01 survival
    assert pytest.approx(reward, abs=1e-6) == 1.01


def test_reward_for_eliminating_two_opponents():
    prev = {f"a{i}": make_agent(f"a{i}", True) for i in range(4)}
    curr = {f"a{i}": make_agent(f"a{i}", i not in (1, 2)) for i in range(4)}
    reward = RewardCalculator.compute(prev, curr, "a0")
    assert pytest.approx(reward, abs=1e-6) == 2.01


def test_penalty_when_self_eliminated():
    prev = {"a0": make_agent("a0", True), "a1": make_agent("a1", True)}
    curr = {"a0": make_agent("a0", False), "a1": make_agent("a1", True)}
    reward = RewardCalculator.compute(prev, curr, "a0")
    assert pytest.approx(reward, abs=1e-6) == -1.0


def test_no_survival_bonus_when_already_dead_before():
    # If agent was already dead in prev, it was dead last step — no survival
    prev = {"a0": make_agent("a0", False), "a1": make_agent("a1", True)}
    curr = {"a0": make_agent("a0", False), "a1": make_agent("a1", True)}
    reward = RewardCalculator.compute(prev, curr, "a0")
    assert pytest.approx(reward, abs=1e-6) == 0.0


def test_already_dead_opponent_elimination_not_double_counted():
    # a1 was already dead in prev — should not count as a new elimination
    prev = {"a0": make_agent("a0", True), "a1": make_agent("a1", False)}
    curr = {"a0": make_agent("a0", True), "a1": make_agent("a1", False)}
    reward = RewardCalculator.compute(prev, curr, "a0")
    assert pytest.approx(reward, abs=1e-6) == 0.01
```

- [ ] **6.2** Run, expect failure:

```bash
poetry run pytest tests/unit/domain/test_reward.py -v
```

Expected failure:
```
ModuleNotFoundError: No module named 'battle_royale.domain.services.reward'
```

- [ ] **6.3** Implement `src/battle_royale/domain/services/reward.py`:

```python
from battle_royale.domain.entities.agent import Agent

_ELIMINATION_REWARD = 1.0
_DEATH_PENALTY = -1.0
_SURVIVAL_REWARD = 0.01


class RewardCalculator:
    @staticmethod
    def compute(
        prev_agents: dict[str, Agent],
        curr_agents: dict[str, Agent],
        agent_id: str,
    ) -> float:
        prev_self = prev_agents[agent_id]
        curr_self = curr_agents[agent_id]

        # Agent was already dead — no rewards or penalties
        if not prev_self.alive:
            return 0.0

        reward = 0.0

        # Count new eliminations of opponents this step
        for aid, prev_agent in prev_agents.items():
            if aid == agent_id:
                continue
            if prev_agent.alive and not curr_agents[aid].alive:
                reward += _ELIMINATION_REWARD

        if not curr_self.alive:
            # Self eliminated this step
            reward += _DEATH_PENALTY
        else:
            reward += _SURVIVAL_REWARD

        return reward
```

- [ ] **6.4** Run, expect pass:

```bash
poetry run pytest tests/unit/domain/test_reward.py -v
```

Expected:
```
6 passed in 0.xxs
```

- [ ] **6.5** Commit:

```bash
git add src/battle_royale/domain/services/reward.py tests/unit/domain/test_reward.py
git commit -m "feat(domain): add RewardCalculator with elimination, death, and survival signals"
```

---

## Task 7 — Infrastructure Config: `Config` Dataclasses and `YamlLoader`

### Steps

- [ ] **7.1** Write the failing test at `tests/unit/infrastructure/test_yaml_loader.py` (create `tests/unit/infrastructure/__init__.py` too):

```python
import pytest
import tempfile
import os
from battle_royale.infrastructure.config.yaml_loader import (
    Config,
    ArenaConfig,
    TrainingConfig,
    PPOConfig,
    EvaluationConfig,
    load_config,
)


def test_default_config_values():
    config = Config()
    assert config.arena.radius == 3.0
    assert config.arena.wall_height == 0.5
    assert config.training.num_agents == 4
    assert config.training.total_steps == 1_000_000
    assert config.training.snapshot_interval == 10_000
    assert config.training.max_force == 10.0
    assert config.training.snapshot_pool_size == 20
    assert config.ppo.lr == pytest.approx(3e-4)
    assert config.ppo.n_steps == 2048
    assert config.ppo.batch_size == 64
    assert config.ppo.clip_range == pytest.approx(0.2)
    assert config.ppo.n_epochs == 10
    assert config.evaluation.eval_freq == 10_000
    assert config.evaluation.num_episodes == 100
    assert config.evaluation.generalization_agent_counts == [4, 6, 8]


def test_load_config_from_yaml(tmp_path):
    yaml_content = """
arena:
  radius: 5.0
  wall_height: 1.0
training:
  num_agents: 6
  total_steps: 500000
  snapshot_interval: 5000
  max_force: 15.0
  snapshot_pool_size: 10
ppo:
  lr: 0.001
  n_steps: 1024
  batch_size: 32
  clip_range: 0.3
  n_epochs: 5
evaluation:
  eval_freq: 5000
  num_episodes: 50
  generalization_agent_counts: [4, 8]
"""
    yaml_file = tmp_path / "test_config.yaml"
    yaml_file.write_text(yaml_content)

    config = load_config(str(yaml_file))
    assert config.arena.radius == 5.0
    assert config.training.num_agents == 6
    assert config.ppo.lr == pytest.approx(0.001)
    assert config.evaluation.generalization_agent_counts == [4, 8]


def test_load_config_partial_override(tmp_path):
    yaml_content = """
arena:
  radius: 4.0
"""
    yaml_file = tmp_path / "partial.yaml"
    yaml_file.write_text(yaml_content)
    config = load_config(str(yaml_file))
    # Overridden
    assert config.arena.radius == 4.0
    # Defaults preserved
    assert config.training.num_agents == 4
    assert config.ppo.n_steps == 2048


def test_load_config_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        load_config("/nonexistent/path/config.yaml")
```

- [ ] **7.2** Run, expect failure:

```bash
poetry run pytest tests/unit/infrastructure/test_yaml_loader.py -v
```

Expected failure:
```
ModuleNotFoundError: No module named 'battle_royale.infrastructure.config.yaml_loader'
```

- [ ] **7.3** Implement `src/battle_royale/infrastructure/config/yaml_loader.py`:

```python
from dataclasses import dataclass, field
import yaml


@dataclass
class ArenaConfig:
    radius: float = 3.0
    wall_height: float = 0.5


@dataclass
class TrainingConfig:
    num_agents: int = 4
    total_steps: int = 1_000_000
    snapshot_interval: int = 10_000
    max_force: float = 10.0
    snapshot_pool_size: int = 20


@dataclass
class PPOConfig:
    lr: float = 3e-4
    n_steps: int = 2048
    batch_size: int = 64
    clip_range: float = 0.2
    n_epochs: int = 10


@dataclass
class EvaluationConfig:
    eval_freq: int = 10_000
    num_episodes: int = 100
    generalization_agent_counts: list[int] = field(default_factory=lambda: [4, 6, 8])


@dataclass
class Config:
    arena: ArenaConfig = field(default_factory=ArenaConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)


def _apply_dict(dataclass_instance, data: dict):
    for key, value in data.items():
        if hasattr(dataclass_instance, key):
            setattr(dataclass_instance, key, value)


def load_config(path: str) -> Config:
    import os
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r") as f:
        data = yaml.safe_load(f) or {}
    config = Config()
    if "arena" in data:
        _apply_dict(config.arena, data["arena"])
    if "training" in data:
        _apply_dict(config.training, data["training"])
    if "ppo" in data:
        _apply_dict(config.ppo, data["ppo"])
    if "evaluation" in data:
        _apply_dict(config.evaluation, data["evaluation"])
    return config
```

- [ ] **7.4** Create `tests/unit/infrastructure/__init__.py` (empty).

- [ ] **7.5** Run, expect pass:

```bash
poetry run pytest tests/unit/infrastructure/test_yaml_loader.py -v
```

Expected:
```
4 passed in 0.xxs
```

- [ ] **7.6** Write content for `config/default.yaml`:

```yaml
arena:
  radius: 3.0
  wall_height: 0.5
training:
  num_agents: 4
  total_steps: 1000000
  snapshot_interval: 10000
  max_force: 10.0
  snapshot_pool_size: 20
ppo:
  lr: 0.0003
  n_steps: 2048
  batch_size: 64
  clip_range: 0.2
  n_epochs: 10
evaluation:
  eval_freq: 10000
  num_episodes: 100
  generalization_agent_counts: [4, 6, 8]
```

- [ ] **7.7** Write content for `config/experiments/4v4_baseline.yaml`:

```yaml
arena:
  radius: 3.0
training:
  num_agents: 4
  total_steps: 1000000
```

- [ ] **7.8** Commit:

```bash
git add src/battle_royale/infrastructure/config/yaml_loader.py tests/unit/infrastructure/ config/default.yaml config/experiments/4v4_baseline.yaml
git commit -m "feat(infra): add Config dataclasses and YAML loader with partial override support"
```

---

## Task 8 — Infrastructure: `XMLBuilder` for N Cylinder Agents

### Steps

- [ ] **8.1** Write the failing test at `tests/unit/infrastructure/test_xml_builder.py`:

```python
import math
import xml.etree.ElementTree as ET
import pytest
from battle_royale.infrastructure.physics.xml_builder import XMLBuilder


def parse(xml_str: str) -> ET.Element:
    return ET.fromstring(xml_str.strip())


def test_xml_builder_produces_valid_xml():
    xml_str = XMLBuilder.build(num_agents=2, arena_radius=3.0, max_force=10.0)
    root = parse(xml_str)
    assert root.tag == "mujoco"


def test_xml_builder_has_floor_plane():
    xml_str = XMLBuilder.build(num_agents=2, arena_radius=3.0, max_force=10.0)
    root = parse(xml_str)
    worldbody = root.find("worldbody")
    assert worldbody is not None
    plane = worldbody.find(".//geom[@type='plane']")
    assert plane is not None


def test_xml_builder_creates_correct_number_of_agents():
    for n in [2, 4, 6, 8]:
        xml_str = XMLBuilder.build(num_agents=n, arena_radius=3.0, max_force=10.0)
        root = parse(xml_str)
        bodies = root.findall(".//body[@name]")
        agent_bodies = [b for b in bodies if b.get("name", "").startswith("agent_")]
        assert len(agent_bodies) == n, f"Expected {n} agents, got {len(agent_bodies)}"


def test_xml_builder_agent_cylinders():
    xml_str = XMLBuilder.build(num_agents=4, arena_radius=3.0, max_force=10.0)
    root = parse(xml_str)
    geoms = root.findall(".//geom[@type='cylinder']")
    assert len(geoms) == 4


def test_xml_builder_agents_have_slide_joints():
    xml_str = XMLBuilder.build(num_agents=3, arena_radius=3.0, max_force=10.0)
    root = parse(xml_str)
    # Each agent body should have 2 slide joints (x and y)
    for i in range(3):
        body = root.find(f".//body[@name='agent_{i}']")
        assert body is not None
        joints = body.findall("joint")
        assert len(joints) == 2
        axes = {j.get("axis") for j in joints}
        assert "1 0 0" in axes
        assert "0 1 0" in axes


def test_xml_builder_motors_per_agent():
    xml_str = XMLBuilder.build(num_agents=4, arena_radius=3.0, max_force=10.0)
    root = parse(xml_str)
    actuators = root.find("actuator")
    assert actuators is not None
    motors = actuators.findall("motor")
    # 2 motors per agent
    assert len(motors) == 8


def test_xml_builder_spawn_positions_on_ring():
    n = 4
    arena_radius = 3.0
    xml_str = XMLBuilder.build(num_agents=n, arena_radius=arena_radius, max_force=10.0)
    root = parse(xml_str)
    spawn_r = 0.6 * arena_radius  # 1.8
    for i in range(n):
        body = root.find(f".//body[@name='agent_{i}']")
        assert body is not None
        pos_str = body.get("pos")
        assert pos_str is not None
        parts = pos_str.split()
        x, y = float(parts[0]), float(parts[1])
        actual_r = math.sqrt(x**2 + y**2)
        assert abs(actual_r - spawn_r) < 1e-4, f"Agent {i} spawn radius {actual_r} != {spawn_r}"


def test_xml_builder_agent_colors_cycle():
    xml_str = XMLBuilder.build(num_agents=4, arena_radius=3.0, max_force=10.0)
    root = parse(xml_str)
    expected_colors = [
        (1, 0, 0), (0, 0, 1), (0, 1, 0), (1, 1, 0),
    ]
    for i, (r, g, b) in enumerate(expected_colors):
        body = root.find(f".//body[@name='agent_{i}']")
        geom = body.find("geom")
        rgba = geom.get("rgba")
        parts = rgba.split()
        assert abs(float(parts[0]) - r) < 1e-3
        assert abs(float(parts[1]) - g) < 1e-3
        assert abs(float(parts[2]) - b) < 1e-3


def test_xml_builder_qpos_layout_2i_convention():
    # Verifies joint naming convention so qpos[2*i]=x, qpos[2*i+1]=y
    xml_str = XMLBuilder.build(num_agents=3, arena_radius=3.0, max_force=10.0)
    root = parse(xml_str)
    for i in range(3):
        body = root.find(f".//body[@name='agent_{i}']")
        joints = body.findall("joint")
        names = {j.get("name") for j in joints}
        assert f"agent_{i}_x" in names
        assert f"agent_{i}_y" in names
```

- [ ] **8.2** Run, expect failure:

```bash
poetry run pytest tests/unit/infrastructure/test_xml_builder.py -v
```

Expected failure: tests fail because `build()` is a module-level function taking no arguments, and agents use spheres, not cylinders.

- [ ] **8.3** Rewrite `src/battle_royale/infrastructure/physics/xml_builder.py`:

```python
import math

_AGENT_COLORS = [
    (1, 0, 0),
    (0, 0, 1),
    (0, 1, 0),
    (1, 1, 0),
    (1, 0, 1),
    (0, 1, 1),
    (0.5, 0.5, 0),
    (0, 0.5, 0.5),
]

_CYLINDER_RADIUS = 0.15
_CYLINDER_HALF_HEIGHT = 0.05
_SPAWN_RADIUS_FRACTION = 0.6


class XMLBuilder:
    @staticmethod
    def build(num_agents: int, arena_radius: float, max_force: float) -> str:
        spawn_r = _SPAWN_RADIUS_FRACTION * arena_radius
        bodies_xml = ""
        for i in range(num_agents):
            angle = (i * 2 * math.pi) / num_agents
            x = spawn_r * math.cos(angle)
            y = spawn_r * math.sin(angle)
            r, g, b = _AGENT_COLORS[i % len(_AGENT_COLORS)]
            bodies_xml += f"""
        <body name="agent_{i}" pos="{x:.6f} {y:.6f} {_CYLINDER_HALF_HEIGHT}">
          <joint name="agent_{i}_x" type="slide" axis="1 0 0" limited="false"/>
          <joint name="agent_{i}_y" type="slide" axis="0 1 0" limited="false"/>
          <geom type="cylinder" size="{_CYLINDER_RADIUS} {_CYLINDER_HALF_HEIGHT}" rgba="{r} {g} {b} 1"/>
        </body>"""

        motors_xml = ""
        for i in range(num_agents):
            motors_xml += f"""
        <motor name="agent_{i}_motor_x" joint="agent_{i}_x" gear="{max_force}" ctrlrange="-1 1"/>
        <motor name="agent_{i}_motor_y" joint="agent_{i}_y" gear="{max_force}" ctrlrange="-1 1"/>"""

        return f"""<mujoco>
  <option timestep="0.01"/>
  <worldbody>
    <geom type="plane" size="10 10 0.1" rgba="0.8 0.8 0.8 1"/>
{bodies_xml}
  </worldbody>
  <actuator>
{motors_xml}
  </actuator>
</mujoco>"""


# Backward-compatible module-level function for main.py
def build() -> str:
    return XMLBuilder.build(num_agents=1, arena_radius=3.0, max_force=10.0)
```

- [ ] **8.4** Run, expect pass:

```bash
poetry run pytest tests/unit/infrastructure/test_xml_builder.py -v
```

Expected:
```
9 passed in 0.xxs
```

- [ ] **8.5** Verify `main.py` still imports cleanly:

```bash
poetry run python3 -c "from src.battle_royale.infrastructure.physics import load_model; load_model(); print('ok')"
```

Expected: `ok`

- [ ] **8.6** Commit:

```bash
git add src/battle_royale/infrastructure/physics/xml_builder.py tests/unit/infrastructure/test_xml_builder.py
git commit -m "feat(infra): rewrite XMLBuilder for N cylinder agents with slide joints and motors"
```

---

## Task 9 — Infrastructure: `MuJoCoEnvironment` (IBattleRoyaleEnv Implementation)

### Steps

- [ ] **9.1** Write the failing test at `tests/unit/infrastructure/test_mujoco_env.py`:

```python
import numpy as np
import pytest
from battle_royale.infrastructure.physics.mujoco_env import MuJoCoEnvironment
from battle_royale.domain.entities.agent import Agent
from battle_royale.domain.entities.arena import Arena
from battle_royale.infrastructure.config.yaml_loader import Config


@pytest.fixture
def config():
    c = Config()
    c.training.num_agents = 4
    c.training.max_force = 10.0
    c.arena.radius = 3.0
    return c


@pytest.fixture
def env(config):
    return MuJoCoEnvironment(config=config)


def test_reset_returns_correct_number_of_agents(env):
    agents = env.reset(num_agents=4)
    assert len(agents) == 4


def test_reset_agent_ids(env):
    agents = env.reset(num_agents=4)
    assert set(agents.keys()) == {"agent_0", "agent_1", "agent_2", "agent_3"}


def test_reset_all_agents_alive(env):
    agents = env.reset(num_agents=4)
    for agent in agents.values():
        assert agent.alive is True


def test_reset_agent_positions_have_shape_2(env):
    agents = env.reset(num_agents=4)
    for agent in agents.values():
        assert agent.position.shape == (2,)
        assert agent.velocity.shape == (2,)


def test_reset_agents_on_spawn_ring(env):
    agents = env.reset(num_agents=4)
    spawn_r = 0.6 * 3.0
    for agent in agents.values():
        r = np.linalg.norm(agent.position)
        assert abs(r - spawn_r) < 0.01


def test_step_returns_correct_keys(env):
    env.reset(num_agents=4)
    actions = {f"agent_{i}": np.zeros(2) for i in range(4)}
    obs, rewards, terminations, truncations = env.step(actions)
    assert set(obs.keys()) == set(rewards.keys()) == set(terminations.keys()) == set(truncations.keys())


def test_step_zero_actions_keep_agents_alive(env):
    env.reset(num_agents=4)
    actions = {f"agent_{i}": np.zeros(2) for i in range(4)}
    _, _, terminations, _ = env.step(actions)
    assert all(not v for v in terminations.values())


def test_get_agents_returns_agent_list(env):
    env.reset(num_agents=4)
    agents = env.get_agents()
    assert len(agents) == 4
    assert all(isinstance(a, Agent) for a in agents)


def test_step_with_force_moves_agents(env):
    env.reset(num_agents=4)
    actions = {"agent_0": np.array([1.0, 0.0])}
    for i in range(1, 4):
        actions[f"agent_{i}"] = np.zeros(2)
    _, _, _, _ = env.step(actions)
    agents_after = {a.id: a for a in env.get_agents()}
    # agent_0 should have moved right somewhat
    assert agents_after["agent_0"].velocity[0] > 0 or agents_after["agent_0"].position[0] > 0.6 * 3.0 * np.cos(0) - 0.001


def test_reset_with_different_num_agents(env):
    agents_4 = env.reset(num_agents=4)
    assert len(agents_4) == 4
    agents_6 = env.reset(num_agents=6)
    assert len(agents_6) == 6


def test_eliminated_agent_stays_dead(env):
    env.reset(num_agents=2)
    # Manually move agent_0 far outside the arena
    import mujoco
    env._data.qpos[0] = 100.0
    env._data.qpos[1] = 0.0
    mujoco.mj_forward(env._model, env._data)
    actions = {"agent_0": np.zeros(2), "agent_1": np.zeros(2)}
    _, _, terminations, _ = env.step(actions)
    assert terminations["agent_0"] is True
    # Step again — still dead
    _, _, terminations2, _ = env.step(actions)
    assert terminations2["agent_0"] is True
```

- [ ] **9.2** Run, expect failure:

```bash
poetry run pytest tests/unit/infrastructure/test_mujoco_env.py -v
```

Expected failure:
```
ImportError: cannot import name 'MuJoCoEnvironment' from 'battle_royale.infrastructure.physics.mujoco_env'
```

- [ ] **9.3** Rewrite `src/battle_royale/infrastructure/physics/mujoco_env.py`:

```python
import math
import numpy as np
import mujoco
from battle_royale.domain.entities.agent import Agent
from battle_royale.domain.entities.arena import Arena
from battle_royale.domain.services.elimination import EliminationService
from battle_royale.infrastructure.physics.xml_builder import XMLBuilder
from battle_royale.infrastructure.config.yaml_loader import Config


class MuJoCoEnvironment:
    def __init__(self, config: Config) -> None:
        self._config = config
        self._model: mujoco.MjModel | None = None
        self._data: mujoco.MjData | None = None
        self._num_agents: int = 0
        self._alive: dict[str, bool] = {}
        self._arena: Arena = Arena(radius=config.arena.radius)

    def reset(self, num_agents: int) -> dict[str, Agent]:
        self._num_agents = num_agents
        xml = XMLBuilder.build(
            num_agents=num_agents,
            arena_radius=self._config.arena.radius,
            max_force=self._config.training.max_force,
        )
        self._model = mujoco.MjModel.from_xml_string(xml)
        self._data = mujoco.MjData(self._model)
        mujoco.mj_forward(self._model, self._data)
        self._alive = {f"agent_{i}": True for i in range(num_agents)}
        return self._extract_agents()

    def step(self, actions: dict[str, np.ndarray]) -> tuple[dict, dict, dict, dict]:
        assert self._model is not None and self._data is not None

        prev_agents = self._extract_agents()

        # Apply actions scaled by motor gear (already baked into MJCF ctrlrange * gear)
        for i in range(self._num_agents):
            agent_id = f"agent_{i}"
            if not self._alive.get(agent_id, False):
                # Freeze eliminated agents
                self._data.ctrl[2 * i] = 0.0
                self._data.ctrl[2 * i + 1] = 0.0
                self._data.qvel[2 * i] = 0.0
                self._data.qvel[2 * i + 1] = 0.0
            else:
                action = actions.get(agent_id, np.zeros(2))
                self._data.ctrl[2 * i] = float(np.clip(action[0], -1.0, 1.0))
                self._data.ctrl[2 * i + 1] = float(np.clip(action[1], -1.0, 1.0))

        mujoco.mj_step(self._model, self._data)

        curr_agents_raw = self._extract_agents_raw()

        # Run elimination
        for i in range(self._num_agents):
            agent_id = f"agent_{i}"
            if not self._alive[agent_id]:
                continue
            raw_agent = curr_agents_raw[agent_id]
            temp_agent = Agent(
                id=agent_id,
                position=raw_agent["position"],
                velocity=raw_agent["velocity"],
                alive=True,
            )
            if EliminationService.is_eliminated(temp_agent, self._arena):
                self._alive[agent_id] = False
                # Freeze in place
                self._data.qvel[2 * i] = 0.0
                self._data.qvel[2 * i + 1] = 0.0
                self._data.ctrl[2 * i] = 0.0
                self._data.ctrl[2 * i + 1] = 0.0

        curr_agents = self._extract_agents()

        from battle_royale.domain.services.reward import RewardCalculator
        rewards = {
            aid: RewardCalculator.compute(prev_agents, curr_agents, aid)
            for aid in curr_agents
        }
        terminations = {aid: not agent.alive for aid, agent in curr_agents.items()}
        truncations = {aid: False for aid in curr_agents}

        return curr_agents, rewards, terminations, truncations

    def get_agents(self) -> list[Agent]:
        return list(self._extract_agents().values())

    def _extract_agents_raw(self) -> dict[str, dict]:
        result = {}
        for i in range(self._num_agents):
            agent_id = f"agent_{i}"
            result[agent_id] = {
                "position": np.array(self._data.qpos[2 * i: 2 * i + 2], dtype=np.float64),
                "velocity": np.array(self._data.qvel[2 * i: 2 * i + 2], dtype=np.float64),
            }
        return result

    def _extract_agents(self) -> dict[str, Agent]:
        agents = {}
        for i in range(self._num_agents):
            agent_id = f"agent_{i}"
            agents[agent_id] = Agent(
                id=agent_id,
                position=np.array(self._data.qpos[2 * i: 2 * i + 2], dtype=np.float64),
                velocity=np.array(self._data.qvel[2 * i: 2 * i + 2], dtype=np.float64),
                alive=self._alive.get(agent_id, True),
            )
        return agents


# Backward-compatible module-level functions for main.py
def load_model() -> mujoco.MjModel:
    from battle_royale.infrastructure.physics.xml_builder import build
    return mujoco.MjModel.from_xml_string(build())


def load_data(model: mujoco.MjModel) -> mujoco.MjData:
    return mujoco.MjData(model)


def render(model: mujoco.MjModel, data: mujoco.MjData, output_path: str) -> None:
    import mediapy
    with mujoco.Renderer(model, height=480, width=640) as renderer:
        renderer.update_scene(data)
        frame = renderer.render()
    mediapy.write_image(output_path, frame)
```

- [ ] **9.4** Update `src/battle_royale/infrastructure/physics/__init__.py`:

```python
from .xml_builder import build, XMLBuilder
from .mujoco_env import MuJoCoEnvironment, load_model, load_data, render
```

- [ ] **9.5** Run, expect pass:

```bash
poetry run pytest tests/unit/infrastructure/test_mujoco_env.py -v
```

Expected:
```
11 passed in 0.xxs
```

- [ ] **9.6** Commit:

```bash
git add src/battle_royale/infrastructure/physics/mujoco_env.py src/battle_royale/infrastructure/physics/__init__.py tests/unit/infrastructure/test_mujoco_env.py
git commit -m "feat(infra): implement MuJoCoEnvironment as IBattleRoyaleEnv with N-agent support"
```

---

## Task 10 — Application Metrics: `EloRatingSystem`

### Steps

- [ ] **10.1** Write the failing test at `tests/unit/application/test_elo.py`:

```python
import pytest
from battle_royale.application.metrics.elo import EloRatingSystem


@pytest.fixture
def elo():
    return EloRatingSystem(initial_rating=1500.0)


def test_initial_rating(elo):
    assert elo.initial_rating == 1500.0


def test_expected_score_equal_ratings(elo):
    score = elo.expected_score(1500.0, 1500.0)
    assert pytest.approx(score, abs=1e-6) == 0.5


def test_expected_score_higher_vs_lower(elo):
    score = elo.expected_score(1700.0, 1500.0)
    assert score > 0.5


def test_expected_score_lower_vs_higher(elo):
    score = elo.expected_score(1500.0, 1700.0)
    assert score < 0.5


def test_expected_scores_sum_to_one(elo):
    a = elo.expected_score(1600.0, 1400.0)
    b = elo.expected_score(1400.0, 1600.0)
    assert pytest.approx(a + b, abs=1e-6) == 1.0


def test_update_winner_gains_rating(elo):
    new_winner, new_loser = elo.update(1500.0, 1500.0)
    assert new_winner > 1500.0


def test_update_loser_loses_rating(elo):
    new_winner, new_loser = elo.update(1500.0, 1500.0)
    assert new_loser < 1500.0


def test_update_preserves_total_rating(elo):
    w, l = elo.update(1600.0, 1400.0)
    assert pytest.approx(w + l, abs=1e-4) == 3000.0


def test_update_k32_equal_ratings(elo):
    # Equal ratings: winner gains K * (1 - 0.5) = 16
    w, l = elo.update(1500.0, 1500.0)
    assert pytest.approx(w, abs=1e-3) == 1516.0
    assert pytest.approx(l, abs=1e-3) == 1484.0


def test_update_strong_favorite_small_gain(elo):
    # Strong favorite wins — small gain
    w, l = elo.update(1800.0, 1200.0)
    gain = w - 1800.0
    assert gain < 5.0  # Expected score was high, so small gain
```

- [ ] **10.2** Run, expect failure:

```bash
poetry run pytest tests/unit/application/test_elo.py -v
```

Expected failure:
```
ModuleNotFoundError: No module named 'battle_royale.application.metrics.elo'
```

- [ ] **10.3** Implement `src/battle_royale/application/metrics/elo.py`:

```python
_K = 32.0


class EloRatingSystem:
    def __init__(self, initial_rating: float = 1500.0) -> None:
        self.initial_rating = initial_rating

    def expected_score(self, rating_a: float, rating_b: float) -> float:
        return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))

    def update(self, winner_rating: float, loser_rating: float) -> tuple[float, float]:
        exp_w = self.expected_score(winner_rating, loser_rating)
        exp_l = self.expected_score(loser_rating, winner_rating)
        new_winner = winner_rating + _K * (1.0 - exp_w)
        new_loser = loser_rating + _K * (0.0 - exp_l)
        return new_winner, new_loser
```

- [ ] **10.4** Run, expect pass:

```bash
poetry run pytest tests/unit/application/test_elo.py -v
```

Expected:
```
10 passed in 0.xxs
```

- [ ] **10.5** Commit:

```bash
git add src/battle_royale/application/metrics/elo.py tests/unit/application/test_elo.py
git commit -m "feat(app): add EloRatingSystem with K=32 and expected score calculation"
```

---

## Task 11 — Application Metrics: `MetricsTracker`

### Steps

- [ ] **11.1** Write the failing test at `tests/unit/application/test_tracker.py`:

```python
from collections import deque
from unittest.mock import MagicMock
import pytest
from battle_royale.application.metrics.tracker import MetricsTracker


@pytest.fixture
def logger():
    return MagicMock()


@pytest.fixture
def tracker(logger):
    return MetricsTracker(logger=logger)


def test_initial_state(tracker):
    assert tracker.episode_count == 0


def test_record_episode_increments_count(tracker):
    tracker.record_episode(winner_id="agent_0", episode_length=100, eliminations=3, step=0)
    assert tracker.episode_count == 1


def test_win_rate_rolling_100_episodes(tracker):
    for i in range(50):
        tracker.record_episode(winner_id="agent_0", episode_length=50, eliminations=1, step=i)
    for i in range(50):
        tracker.record_episode(winner_id="agent_1", episode_length=50, eliminations=1, step=50 + i)
    # 50% win rate for agent_0 over last 100
    win_rates = tracker.get_win_rates()
    assert pytest.approx(win_rates.get("agent_0", 0.0), abs=0.01) == 0.5


def test_logger_called_on_record(tracker, logger):
    tracker.record_episode(winner_id="agent_0", episode_length=100, eliminations=2, step=42)
    logger.log.assert_called_once()
    call_args = logger.log.call_args
    metrics = call_args[0][0]
    assert "episode_length" in metrics
    assert "eliminations" in metrics


def test_mean_episode_length_tracked(tracker):
    tracker.record_episode(winner_id="agent_0", episode_length=100, eliminations=1, step=0)
    tracker.record_episode(winner_id="agent_1", episode_length=200, eliminations=1, step=1)
    assert tracker.mean_episode_length() == pytest.approx(150.0, abs=0.1)
```

- [ ] **11.2** Run, expect failure:

```bash
poetry run pytest tests/unit/application/test_tracker.py -v
```

Expected failure:
```
ModuleNotFoundError: No module named 'battle_royale.application.metrics.tracker'
```

- [ ] **11.3** Implement `src/battle_royale/application/metrics/tracker.py`:

```python
from collections import deque
from battle_royale.domain.interfaces.logger import ILogger
from battle_royale.application.metrics.elo import EloRatingSystem

_ROLLING_WINDOW = 100


class MetricsTracker:
    def __init__(self, logger: ILogger) -> None:
        self._logger = logger
        self._elo = EloRatingSystem()
        self._ratings: dict[str, float] = {}
        self._win_history: deque[str] = deque(maxlen=_ROLLING_WINDOW)
        self._episode_lengths: list[int] = []
        self._eliminations: list[int] = []
        self.episode_count: int = 0

    def record_episode(
        self,
        winner_id: str,
        episode_length: int,
        eliminations: int,
        step: int,
    ) -> None:
        self._win_history.append(winner_id)
        self._episode_lengths.append(episode_length)
        self._eliminations.append(eliminations)
        self.episode_count += 1

        # Update Elo for winner vs all others seen
        if winner_id not in self._ratings:
            self._ratings[winner_id] = self._elo.initial_rating

        metrics = {
            "episode_length": float(episode_length),
            "eliminations": float(eliminations),
            "episode_count": float(self.episode_count),
        }
        for agent_id, rate in self.get_win_rates().items():
            metrics[f"win_rate/{agent_id}"] = rate

        self._logger.log(metrics, step)

    def get_win_rates(self) -> dict[str, float]:
        if not self._win_history:
            return {}
        counts: dict[str, int] = {}
        for winner in self._win_history:
            counts[winner] = counts.get(winner, 0) + 1
        total = len(self._win_history)
        return {aid: count / total for aid, count in counts.items()}

    def mean_episode_length(self) -> float:
        if not self._episode_lengths:
            return 0.0
        return sum(self._episode_lengths) / len(self._episode_lengths)
```

- [ ] **11.4** Run, expect pass:

```bash
poetry run pytest tests/unit/application/test_tracker.py -v
```

Expected:
```
5 passed in 0.xxs
```

- [ ] **11.5** Commit:

```bash
git add src/battle_royale/application/metrics/tracker.py tests/unit/application/test_tracker.py
git commit -m "feat(app): add MetricsTracker with rolling win rate, Elo integration, and ILogger calls"
```

---

## Task 12 — Application Training: `SnapshotPool`

### Steps

- [ ] **12.1** Write the failing test at `tests/unit/application/test_snapshot_pool.py`:

```python
import os
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest
from battle_royale.application.training.snapshot_pool import SnapshotPool


@pytest.fixture
def pool(tmp_path):
    return SnapshotPool(save_dir=str(tmp_path), max_size=3)


def test_pool_starts_empty(pool):
    assert pool.is_empty() is True


def test_sample_path_returns_none_when_empty(pool):
    assert pool.sample_path() is None


def test_save_creates_file(pool, tmp_path):
    mock_model = MagicMock()
    pool.save(mock_model, step=1000)
    files = list(tmp_path.iterdir())
    assert len(files) == 1


def test_save_calls_model_save(pool, tmp_path):
    mock_model = MagicMock()
    pool.save(mock_model, step=1000)
    mock_model.save.assert_called_once()


def test_pool_not_empty_after_save(pool):
    mock_model = MagicMock()
    pool.save(mock_model, step=1000)
    assert pool.is_empty() is False


def test_sample_path_returns_path_after_save(pool):
    mock_model = MagicMock()
    pool.save(mock_model, step=1000)
    path = pool.sample_path()
    assert path is not None
    assert isinstance(path, Path)


def test_pool_evicts_oldest_when_full(pool, tmp_path):
    mock_model = MagicMock()
    pool.save(mock_model, step=1000)
    pool.save(mock_model, step=2000)
    pool.save(mock_model, step=3000)
    pool.save(mock_model, step=4000)  # should evict step=1000
    files = list(tmp_path.iterdir())
    assert len(files) == 3
    names = {f.stem for f in files}
    assert "snapshot_001000" not in names


def test_sample_path_is_random(pool):
    mock_model = MagicMock()
    for step in [1000, 2000, 3000]:
        pool.save(mock_model, step=step)
    # Sample multiple times — should occasionally differ (probabilistic)
    paths = {pool.sample_path() for _ in range(50)}
    assert len(paths) > 1  # With 3 paths and 50 samples, very unlikely to get same every time
```

- [ ] **12.2** Run, expect failure:

```bash
poetry run pytest tests/unit/application/test_snapshot_pool.py -v
```

Expected failure:
```
ModuleNotFoundError: No module named 'battle_royale.application.training.snapshot_pool'
```

- [ ] **12.3** Implement `src/battle_royale/application/training/snapshot_pool.py`:

```python
import random
from pathlib import Path


class SnapshotPool:
    def __init__(self, save_dir: str, max_size: int = 20) -> None:
        self._save_dir = Path(save_dir)
        self._save_dir.mkdir(parents=True, exist_ok=True)
        self._max_size = max_size
        self._paths: list[Path] = []

    def save(self, model, step: int) -> None:
        filename = f"snapshot_{step:06d}"
        path = self._save_dir / filename
        model.save(str(path))
        self._paths.append(path)
        if len(self._paths) > self._max_size:
            oldest = self._paths.pop(0)
            # Remove all files matching this snapshot (SB3 saves .zip)
            for f in self._save_dir.glob(f"{oldest.name}*"):
                f.unlink(missing_ok=True)

    def sample_path(self) -> Path | None:
        if not self._paths:
            return None
        return random.choice(self._paths)

    def is_empty(self) -> bool:
        return len(self._paths) == 0
```

- [ ] **12.4** Run, expect pass:

```bash
poetry run pytest tests/unit/application/test_snapshot_pool.py -v
```

Expected:
```
8 passed in 0.xxs
```

- [ ] **12.5** Commit:

```bash
git add src/battle_royale/application/training/snapshot_pool.py tests/unit/application/test_snapshot_pool.py
git commit -m "feat(app): add SnapshotPool with FIFO eviction and random sampling"
```

---

## Task 13 — Infrastructure Logging: `WandBLogger`

### Steps

- [ ] **13.1** Write the failing test at `tests/unit/infrastructure/test_wandb_logger.py`:

```python
from unittest.mock import MagicMock, patch
import pytest
from battle_royale.infrastructure.logging.wandb_logger import WandBLogger


@patch("battle_royale.infrastructure.logging.wandb_logger.wandb")
def test_log_calls_wandb_log(mock_wandb):
    logger = WandBLogger(project="test", run_name="run1", config={})
    logger.log({"loss": 0.5, "reward": 1.2}, step=100)
    mock_wandb.log.assert_called_once_with({"loss": 0.5, "reward": 1.2}, step=100)


@patch("battle_royale.infrastructure.logging.wandb_logger.wandb")
def test_save_artifact_calls_wandb(mock_wandb):
    mock_artifact = MagicMock()
    mock_wandb.Artifact.return_value = mock_artifact
    logger = WandBLogger(project="test", run_name="run1", config={})
    logger.save_artifact("/some/path/model.zip", "model_checkpoint")
    mock_wandb.Artifact.assert_called_once()
    mock_artifact.add_file.assert_called_once_with("/some/path/model.zip")
    mock_wandb.log_artifact.assert_called_once_with(mock_artifact)


@patch("battle_royale.infrastructure.logging.wandb_logger.wandb")
def test_init_calls_wandb_init(mock_wandb):
    logger = WandBLogger(project="myproject", run_name="exp1", config={"lr": 0.001})
    mock_wandb.init.assert_called_once_with(
        project="myproject",
        name="exp1",
        config={"lr": 0.001},
    )
```

- [ ] **13.2** Run, expect failure:

```bash
poetry run pytest tests/unit/infrastructure/test_wandb_logger.py -v
```

Expected failure:
```
ModuleNotFoundError: No module named 'battle_royale.infrastructure.logging.wandb_logger'
```

- [ ] **13.3** Implement `src/battle_royale/infrastructure/logging/wandb_logger.py`:

```python
import wandb


class WandBLogger:
    def __init__(self, project: str, run_name: str, config: dict) -> None:
        wandb.init(project=project, name=run_name, config=config)

    def log(self, metrics: dict[str, float], step: int) -> None:
        wandb.log(metrics, step=step)

    def save_artifact(self, path: str, name: str) -> None:
        artifact = wandb.Artifact(name=name, type="model")
        artifact.add_file(path)
        wandb.log_artifact(artifact)
```

- [ ] **13.4** Run, expect pass:

```bash
poetry run pytest tests/unit/infrastructure/test_wandb_logger.py -v
```

Expected:
```
3 passed in 0.xxs
```

- [ ] **13.5** Commit:

```bash
git add src/battle_royale/infrastructure/logging/wandb_logger.py tests/unit/infrastructure/test_wandb_logger.py
git commit -m "feat(infra): add WandBLogger implementing ILogger protocol"
```

---

## Task 14 — Integration Test: Full Environment Loop

### Steps

- [ ] **14.1** Write the failing integration test at `tests/integration/test_environment.py`:

```python
import numpy as np
import pytest
from battle_royale.infrastructure.physics.mujoco_env import MuJoCoEnvironment
from battle_royale.infrastructure.config.yaml_loader import Config
from battle_royale.domain.services.observation import ObservationBuilder
from battle_royale.domain.entities.arena import Arena


@pytest.fixture
def config():
    c = Config()
    c.training.num_agents = 4
    c.training.max_force = 10.0
    c.arena.radius = 3.0
    return c


@pytest.fixture
def env(config):
    return MuJoCoEnvironment(config=config)


def test_full_episode_runs_without_error(env):
    agents = env.reset(num_agents=4)
    assert len(agents) == 4
    for step in range(10):
        actions = {aid: np.random.uniform(-1, 1, 2).astype(np.float32) for aid in agents}
        agents, rewards, terminations, truncations = env.step(actions)
    assert all(isinstance(r, float) for r in rewards.values())


def test_observation_builder_integrates_with_env(env):
    arena = Arena(radius=3.0)
    agents_dict = env.reset(num_agents=4)
    agents_list = list(agents_dict.values())
    for agent in agents_list:
        obs = ObservationBuilder.build(agent, agents_list, arena)
        assert obs.shape == (17,)
        assert obs.dtype == np.float32


def test_reward_sums_are_finite(env):
    env.reset(num_agents=4)
    for _ in range(5):
        actions = {f"agent_{i}": np.zeros(2) for i in range(4)}
        _, rewards, _, _ = env.step(actions)
        for r in rewards.values():
            assert np.isfinite(r)


def test_terminations_are_bool(env):
    env.reset(num_agents=4)
    actions = {f"agent_{i}": np.zeros(2) for i in range(4)}
    _, _, terminations, _ = env.step(actions)
    for v in terminations.values():
        assert isinstance(v, bool)


def test_multi_step_episode_with_random_actions(env):
    agents_dict = env.reset(num_agents=4)
    alive_agents = set(agents_dict.keys())
    for step in range(50):
        actions = {aid: np.random.uniform(-1, 1, 2) for aid in alive_agents}
        agents_dict, rewards, terminations, truncations = env.step(actions)
        # Remove newly eliminated agents for next step
        alive_agents = {aid for aid, term in terminations.items() if not term}
        if not alive_agents:
            break
    # Test passed if no exception was raised


def test_reset_clears_previous_state(env):
    env.reset(num_agents=4)
    actions = {f"agent_{i}": np.ones(2) for i in range(4)}
    for _ in range(20):
        env.step(actions)
    # Reset and verify fresh state
    agents = env.reset(num_agents=4)
    for agent in agents.values():
        assert agent.alive is True
```

- [ ] **14.2** Run, expect failure:

```bash
poetry run pytest tests/integration/test_environment.py -v
```

At this point all implementation exists; the test should pass immediately. If any test fails, it reveals an integration bug to fix before committing.

- [ ] **14.3** Run, expect pass:

```bash
poetry run pytest tests/integration/test_environment.py -v
```

Expected:
```
6 passed in 0.xxs
```

- [ ] **14.4** Commit:

```bash
git add tests/integration/test_environment.py
git commit -m "test(integration): add full environment loop integration tests"
```

---

## Task 15 — Interfaces: PettingZoo `BattleRoyaleEnv`

### Steps

- [ ] **15.1** Write the failing test at `tests/unit/interfaces/test_pettingzoo_env.py` (create `tests/unit/interfaces/__init__.py` too):

```python
import numpy as np
import pytest
from unittest.mock import MagicMock
from pettingzoo.test import parallel_api_test
from battle_royale.interfaces.pettingzoo.env import BattleRoyaleEnv
from battle_royale.infrastructure.physics.mujoco_env import MuJoCoEnvironment
from battle_royale.infrastructure.config.yaml_loader import Config


@pytest.fixture
def config():
    c = Config()
    c.training.num_agents = 4
    c.arena.radius = 3.0
    c.training.max_force = 10.0
    return c


@pytest.fixture
def pz_env(config):
    mujoco_env = MuJoCoEnvironment(config=config)
    return BattleRoyaleEnv(env=mujoco_env, config=config)


def test_pettingzoo_env_reset(pz_env):
    observations, infos = pz_env.reset()
    assert len(observations) == 4
    for obs in observations.values():
        assert obs.shape == (17,)
        assert obs.dtype == np.float32


def test_pettingzoo_env_agents_list(pz_env):
    pz_env.reset()
    assert len(pz_env.agents) == 4
    assert "agent_0" in pz_env.agents


def test_pettingzoo_env_observation_space(pz_env):
    pz_env.reset()
    for agent in pz_env.agents:
        space = pz_env.observation_space(agent)
        assert space.shape == (17,)


def test_pettingzoo_env_action_space(pz_env):
    pz_env.reset()
    for agent in pz_env.agents:
        space = pz_env.action_space(agent)
        assert space.shape == (2,)
        np.testing.assert_array_equal(space.low, np.full(2, -1.0))
        np.testing.assert_array_equal(space.high, np.full(2, 1.0))


def test_pettingzoo_env_step(pz_env):
    pz_env.reset()
    actions = {agent: pz_env.action_space(agent).sample() for agent in pz_env.agents}
    observations, rewards, terminations, truncations, infos = pz_env.step(actions)
    assert set(observations.keys()) == set(rewards.keys())
    assert set(terminations.keys()) == set(truncations.keys())


def test_pettingzoo_api_compliance(config):
    """Runs the official PettingZoo parallel API test."""
    mujoco_env = MuJoCoEnvironment(config=config)
    env = BattleRoyaleEnv(env=mujoco_env, config=config)
    parallel_api_test(env, num_cycles=10)
```

- [ ] **15.2** Run, expect failure:

```bash
poetry run pytest tests/unit/interfaces/test_pettingzoo_env.py -v
```

Expected failure:
```
ModuleNotFoundError: No module named 'battle_royale.interfaces.pettingzoo.env'
```

- [ ] **15.3** Implement `src/battle_royale/interfaces/pettingzoo/env.py`:

```python
from __future__ import annotations

import numpy as np
from gymnasium import spaces
from pettingzoo import ParallelEnv

from battle_royale.domain.interfaces.environment import IBattleRoyaleEnv
from battle_royale.domain.services.observation import ObservationBuilder
from battle_royale.domain.entities.arena import Arena
from battle_royale.infrastructure.config.yaml_loader import Config

_OBS_DIM = 17


class BattleRoyaleEnv(ParallelEnv):
    metadata = {"render_modes": [], "name": "battle_royale_v0"}

    def __init__(self, env: IBattleRoyaleEnv, config: Config) -> None:
        super().__init__()
        self._env = env
        self._config = config
        self._arena = Arena(radius=config.arena.radius)
        self._num_agents = config.training.num_agents
        self.possible_agents = [f"agent_{i}" for i in range(self._num_agents)]
        self.agents: list[str] = []
        self._observations: dict[str, np.ndarray] = {}
        self._agent_objects: list = []

    def reset(self, seed=None, options=None):
        self._num_agents = self._config.training.num_agents
        self.possible_agents = [f"agent_{i}" for i in range(self._num_agents)]
        agents_dict = self._env.reset(num_agents=self._num_agents)
        self.agents = list(agents_dict.keys())
        self._agent_objects = list(agents_dict.values())
        observations = self._build_observations()
        self.terminations = {aid: False for aid in self.agents}
        self.truncations = {aid: False for aid in self.agents}
        return observations, {aid: {} for aid in self.agents}

    def step(self, actions: dict[str, np.ndarray]):
        # Only pass actions for alive agents
        filtered = {aid: actions[aid] for aid in self.agents if not self.terminations.get(aid, False)}
        # Pad with zeros for terminated agents
        for aid in self.possible_agents:
            if aid not in filtered:
                filtered[aid] = np.zeros(2, dtype=np.float32)

        agents_dict, rewards, terminations, truncations = self._env.step(filtered)
        self._agent_objects = list(agents_dict.values())

        observations = self._build_observations()
        self.terminations = terminations
        self.truncations = truncations

        # Remove fully done agents from self.agents
        self.agents = [aid for aid in self.agents if not (terminations.get(aid, False) and truncations.get(aid, False))]

        return observations, rewards, terminations, truncations, {aid: {} for aid in observations}

    def observation_space(self, agent: str) -> spaces.Box:
        return spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(_OBS_DIM,),
            dtype=np.float32,
        )

    def action_space(self, agent: str) -> spaces.Box:
        return spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(2,),
            dtype=np.float32,
        )

    def _build_observations(self) -> dict[str, np.ndarray]:
        obs = {}
        for agent_obj in self._agent_objects:
            obs[agent_obj.id] = ObservationBuilder.build(
                agent_obj, self._agent_objects, self._arena
            )
        return obs
```

- [ ] **15.4** Create `tests/unit/interfaces/__init__.py` (empty).

- [ ] **15.5** Run, expect pass:

```bash
poetry run pytest tests/unit/interfaces/test_pettingzoo_env.py -v
```

Expected:
```
6 passed in 0.xxs
```

- [ ] **15.6** Commit:

```bash
git add src/battle_royale/interfaces/pettingzoo/env.py tests/unit/interfaces/ 
git commit -m "feat(interfaces): add PettingZoo ParallelEnv wrapper with observation/action spaces"
```

---

## Task 16 — Application: `Trainer` with SB3 PPO

### Steps

- [ ] **16.1** Write the failing test at `tests/unit/application/test_trainer.py`:

```python
from unittest.mock import MagicMock, patch, call
import pytest
from battle_royale.application.training.trainer import Trainer
from battle_royale.infrastructure.config.yaml_loader import Config


@pytest.fixture
def config():
    c = Config()
    c.training.num_agents = 4
    c.training.total_steps = 100
    c.training.snapshot_interval = 50
    c.ppo.n_steps = 8
    c.ppo.batch_size = 4
    return c


@pytest.fixture
def mock_env():
    env = MagicMock()
    env.agents = ["agent_0", "agent_1", "agent_2", "agent_3"]
    return env


@pytest.fixture
def mock_logger():
    return MagicMock()


@pytest.fixture
def mock_snapshot_pool():
    pool = MagicMock()
    pool.is_empty.return_value = True
    pool.sample_path.return_value = None
    return pool


@pytest.fixture
def mock_tracker():
    return MagicMock()


def test_trainer_can_be_constructed(mock_env, mock_logger, mock_snapshot_pool, mock_tracker, config):
    trainer = Trainer(
        env=mock_env,
        logger=mock_logger,
        snapshot_pool=mock_snapshot_pool,
        tracker=mock_tracker,
        config=config,
    )
    assert trainer is not None


@patch("battle_royale.application.training.trainer.PPO")
@patch("battle_royale.application.training.trainer.pettingzoo_env_to_vec_env_v1")
def test_trainer_run_creates_ppo_model(mock_supersuit, mock_ppo, mock_env, mock_logger, mock_snapshot_pool, mock_tracker, config):
    mock_vec_env = MagicMock()
    mock_supersuit.return_value = mock_vec_env
    mock_model_instance = MagicMock()
    mock_ppo.return_value = mock_model_instance

    trainer = Trainer(
        env=mock_env,
        logger=mock_logger,
        snapshot_pool=mock_snapshot_pool,
        tracker=mock_tracker,
        config=config,
    )
    trainer.run()

    mock_ppo.assert_called_once()
    mock_model_instance.learn.assert_called_once()


@patch("battle_royale.application.training.trainer.PPO")
@patch("battle_royale.application.training.trainer.pettingzoo_env_to_vec_env_v1")
def test_trainer_run_calls_supersuit_wrapper(mock_supersuit, mock_ppo, mock_env, mock_logger, mock_snapshot_pool, mock_tracker, config):
    mock_vec_env = MagicMock()
    mock_supersuit.return_value = mock_vec_env
    mock_model_instance = MagicMock()
    mock_ppo.return_value = mock_model_instance

    trainer = Trainer(
        env=mock_env,
        logger=mock_logger,
        snapshot_pool=mock_snapshot_pool,
        tracker=mock_tracker,
        config=config,
    )
    trainer.run()
    mock_supersuit.assert_called_once_with(mock_env)
```

- [ ] **16.2** Run, expect failure:

```bash
poetry run pytest tests/unit/application/test_trainer.py -v
```

Expected failure:
```
ModuleNotFoundError: No module named 'battle_royale.application.training.trainer'
```

- [ ] **16.3** Implement `src/battle_royale/application/training/trainer.py`:

```python
from __future__ import annotations

from stable_baselines3 import PPO
from supersuit import pettingzoo_env_to_vec_env_v1

from battle_royale.domain.interfaces.logger import ILogger
from battle_royale.application.training.snapshot_pool import SnapshotPool
from battle_royale.application.metrics.tracker import MetricsTracker
from battle_royale.infrastructure.config.yaml_loader import Config


class Trainer:
    def __init__(
        self,
        env,
        logger: ILogger,
        snapshot_pool: SnapshotPool,
        tracker: MetricsTracker,
        config: Config,
    ) -> None:
        self._env = env
        self._logger = logger
        self._snapshot_pool = snapshot_pool
        self._tracker = tracker
        self._config = config

    def run(self) -> None:
        vec_env = pettingzoo_env_to_vec_env_v1(self._env)

        model = PPO(
            policy="MlpPolicy",
            env=vec_env,
            learning_rate=self._config.ppo.lr,
            n_steps=self._config.ppo.n_steps,
            batch_size=self._config.ppo.batch_size,
            clip_range=self._config.ppo.clip_range,
            n_epochs=self._config.ppo.n_epochs,
            verbose=1,
        )

        model.learn(
            total_timesteps=self._config.training.total_steps,
            callback=self._make_callback(model),
        )

    def _make_callback(self, model):
        from stable_baselines3.common.callbacks import BaseCallback
        config = self._config
        pool = self._snapshot_pool

        class SnapshotCallback(BaseCallback):
            def __init__(self):
                super().__init__()

            def _on_step(self) -> bool:
                if self.n_calls % config.training.snapshot_interval == 0:
                    pool.save(self.model, step=self.num_timesteps)
                return True

        return SnapshotCallback()
```

- [ ] **16.4** Run, expect pass:

```bash
poetry run pytest tests/unit/application/test_trainer.py -v
```

Expected:
```
3 passed in 0.xxs
```

- [ ] **16.5** Commit:

```bash
git add src/battle_royale/application/training/trainer.py tests/unit/application/test_trainer.py
git commit -m "feat(app): add Trainer wrapping SB3 PPO with SuperSuit and snapshot callback"
```

---

## Task 17 — Application: `Evaluator`

### Steps

- [ ] **17.1** Write the failing test at `tests/unit/application/test_evaluator.py`:

```python
import numpy as np
import pytest
from unittest.mock import MagicMock, patch
from battle_royale.application.evaluation.evaluator import Evaluator


@pytest.fixture
def mock_env():
    env = MagicMock()
    env.agents = ["agent_0", "agent_1", "agent_2", "agent_3"]
    env.reset.return_value = (
        {"agent_0": np.zeros(17), "agent_1": np.zeros(17), "agent_2": np.zeros(17), "agent_3": np.zeros(17)},
        {},
    )
    # step returns obs, rewards, terminations (all True after first step), truncations, infos
    env.step.return_value = (
        {"agent_0": np.zeros(17)},
        {"agent_0": 1.0, "agent_1": 0.0, "agent_2": 0.0, "agent_3": 0.0},
        {"agent_0": False, "agent_1": True, "agent_2": True, "agent_3": True},
        {"agent_0": False, "agent_1": False, "agent_2": False, "agent_3": False},
        {},
    )
    env.terminations = {"agent_0": False, "agent_1": True, "agent_2": True, "agent_3": True}
    env.action_space = MagicMock(return_value=MagicMock(sample=lambda: np.zeros(2)))
    return env


@pytest.fixture
def mock_model():
    model = MagicMock()
    model.predict.return_value = (np.zeros(2), None)
    return model


@pytest.fixture
def mock_logger():
    return MagicMock()


@pytest.fixture
def mock_pool():
    pool = MagicMock()
    pool.is_empty.return_value = True
    return pool


def test_evaluator_can_be_constructed(mock_env, mock_pool, mock_logger):
    env_factory = lambda n: mock_env
    evaluator = Evaluator(env_factory=env_factory, snapshot_pool=mock_pool, logger=mock_logger)
    assert evaluator is not None


def test_evaluate_returns_metrics_dict(mock_env, mock_model, mock_pool, mock_logger):
    env_factory = lambda n: mock_env
    evaluator = Evaluator(env_factory=env_factory, snapshot_pool=mock_pool, logger=mock_logger)
    metrics = evaluator.evaluate(model=mock_model, num_agents=4, num_episodes=3)
    assert isinstance(metrics, dict)
    assert "win_rate" in metrics or "mean_episode_length" in metrics


def test_evaluate_calls_logger(mock_env, mock_model, mock_pool, mock_logger):
    env_factory = lambda n: mock_env
    evaluator = Evaluator(env_factory=env_factory, snapshot_pool=mock_pool, logger=mock_logger)
    evaluator.evaluate(model=mock_model, num_agents=4, num_episodes=2)
    mock_logger.log.assert_called()
```

- [ ] **17.2** Run, expect failure:

```bash
poetry run pytest tests/unit/application/test_evaluator.py -v
```

Expected failure:
```
ModuleNotFoundError: No module named 'battle_royale.application.evaluation.evaluator'
```

- [ ] **17.3** Implement `src/battle_royale/application/evaluation/evaluator.py`:

```python
from __future__ import annotations

from typing import Callable
import numpy as np

from battle_royale.application.training.snapshot_pool import SnapshotPool
from battle_royale.domain.interfaces.logger import ILogger


class Evaluator:
    def __init__(
        self,
        env_factory: Callable[[int], object],
        snapshot_pool: SnapshotPool,
        logger: ILogger,
    ) -> None:
        self._env_factory = env_factory
        self._snapshot_pool = snapshot_pool
        self._logger = logger

    def evaluate(self, model, num_agents: int, num_episodes: int) -> dict[str, float]:
        env = self._env_factory(num_agents)
        win_counts: dict[str, int] = {}
        episode_lengths: list[int] = []

        for _ in range(num_episodes):
            observations, _ = env.reset()
            done = False
            step_count = 0
            while not done:
                actions = {}
                for agent in env.agents:
                    obs = observations.get(agent, np.zeros(17))
                    action, _ = model.predict(obs)
                    actions[agent] = action
                observations, rewards, terminations, truncations, _ = env.step(actions)
                step_count += 1
                alive = [a for a in env.agents if not terminations.get(a, True)]
                done = len(alive) <= 1 or step_count >= 1000
                if len(alive) == 1:
                    winner = alive[0]
                    win_counts[winner] = win_counts.get(winner, 0) + 1
            episode_lengths.append(step_count)

        total = sum(win_counts.values()) or 1
        win_rates = {aid: count / total for aid, count in win_counts.items()}
        mean_len = float(np.mean(episode_lengths)) if episode_lengths else 0.0

        metrics = {"mean_episode_length": mean_len, **{f"win_rate/{k}": v for k, v in win_rates.items()}}
        if win_rates:
            metrics["win_rate"] = max(win_rates.values())
        self._logger.log(metrics, step=0)
        return metrics
```

- [ ] **17.4** Run, expect pass:

```bash
poetry run pytest tests/unit/application/test_evaluator.py -v
```

Expected:
```
3 passed in 0.xxs
```

- [ ] **17.5** Commit:

```bash
git add src/battle_royale/application/evaluation/evaluator.py tests/unit/application/test_evaluator.py
git commit -m "feat(app): add Evaluator running multi-episode rollouts with win rate tracking"
```

---

## Task 18 — Interfaces: CLI `train` and `evaluate`

### Steps

- [ ] **18.1** Write the failing test at `tests/unit/interfaces/test_cli.py`:

```python
import sys
from unittest.mock import patch, MagicMock
import pytest


def test_train_cli_imports():
    """Verify the train CLI module is importable."""
    import importlib
    spec = importlib.util.find_spec("battle_royale.interfaces.cli.train")
    assert spec is not None


def test_evaluate_cli_imports():
    """Verify the evaluate CLI module is importable."""
    import importlib
    spec = importlib.util.find_spec("battle_royale.interfaces.cli.evaluate")
    assert spec is not None


@patch("battle_royale.interfaces.cli.train.Trainer")
@patch("battle_royale.interfaces.cli.train.load_config")
@patch("battle_royale.interfaces.cli.train.MuJoCoEnvironment")
@patch("battle_royale.interfaces.cli.train.BattleRoyaleEnv")
@patch("battle_royale.interfaces.cli.train.WandBLogger")
@patch("battle_royale.interfaces.cli.train.SnapshotPool")
@patch("battle_royale.interfaces.cli.train.MetricsTracker")
def test_train_main_wires_dependencies(
    mock_tracker_cls, mock_pool_cls, mock_logger_cls,
    mock_env_cls, mock_mujoco_cls, mock_load_config, mock_trainer_cls
):
    mock_config = MagicMock()
    mock_config.training.num_agents = 4
    mock_config.training.snapshot_pool_size = 20
    mock_load_config.return_value = mock_config

    from battle_royale.interfaces.cli.train import main
    main(config_path="config/default.yaml", run_dir="/tmp/test_run")

    mock_load_config.assert_called_once_with("config/default.yaml")
    mock_mujoco_cls.assert_called_once()
    mock_env_cls.assert_called_once()
    mock_trainer_cls.assert_called_once()
    mock_trainer_cls.return_value.run.assert_called_once()


@patch("battle_royale.interfaces.cli.evaluate.Evaluator")
@patch("battle_royale.interfaces.cli.evaluate.load_config")
@patch("battle_royale.interfaces.cli.evaluate.MuJoCoEnvironment")
@patch("battle_royale.interfaces.cli.evaluate.BattleRoyaleEnv")
@patch("battle_royale.interfaces.cli.evaluate.SnapshotPool")
@patch("battle_royale.interfaces.cli.evaluate.PPO")
def test_evaluate_main_loads_checkpoint(
    mock_ppo_cls, mock_pool_cls, mock_env_cls,
    mock_mujoco_cls, mock_load_config, mock_evaluator_cls
):
    mock_config = MagicMock()
    mock_config.training.num_agents = 4
    mock_config.arena.radius = 3.0
    mock_config.training.max_force = 10.0
    mock_load_config.return_value = mock_config
    mock_evaluator_cls.return_value.evaluate.return_value = {"win_rate": 0.5}

    from battle_royale.interfaces.cli.evaluate import main
    main(checkpoint_path="/tmp/checkpoint", num_agents=6, config_path="config/default.yaml")

    mock_ppo_cls.load.assert_called_once_with("/tmp/checkpoint")
    mock_evaluator_cls.return_value.evaluate.assert_called_once()
```

- [ ] **18.2** Run, expect failure:

```bash
poetry run pytest tests/unit/interfaces/test_cli.py -v
```

Expected failure:
```
ModuleNotFoundError: No module named 'battle_royale.interfaces.cli.train'
```

- [ ] **18.3** Implement `src/battle_royale/interfaces/cli/train.py`:

```python
"""
Usage:
    python -m battle_royale.interfaces.cli.train --config config/experiments/4v4_baseline.yaml
"""
from __future__ import annotations

import argparse
import os

from battle_royale.infrastructure.config.yaml_loader import load_config, Config
from battle_royale.infrastructure.physics.mujoco_env import MuJoCoEnvironment
from battle_royale.infrastructure.logging.wandb_logger import WandBLogger
from battle_royale.interfaces.pettingzoo.env import BattleRoyaleEnv
from battle_royale.application.training.snapshot_pool import SnapshotPool
from battle_royale.application.metrics.tracker import MetricsTracker
from battle_royale.application.training.trainer import Trainer


def main(config_path: str, run_dir: str) -> None:
    config = load_config(config_path)
    os.makedirs(run_dir, exist_ok=True)

    mujoco_env = MuJoCoEnvironment(config=config)
    pz_env = BattleRoyaleEnv(env=mujoco_env, config=config)

    logger = WandBLogger(
        project="battle-royale",
        run_name=os.path.basename(run_dir),
        config={"config_path": config_path},
    )
    snapshot_pool = SnapshotPool(
        save_dir=os.path.join(run_dir, "snapshots"),
        max_size=config.training.snapshot_pool_size,
    )
    tracker = MetricsTracker(logger=logger)

    trainer = Trainer(
        env=pz_env,
        logger=logger,
        snapshot_pool=snapshot_pool,
        tracker=tracker,
        config=config,
    )
    trainer.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Battle Royale agents")
    parser.add_argument("--config", default="config/default.yaml", help="Path to YAML config")
    parser.add_argument("--run-dir", default="runs/latest", help="Directory to save outputs")
    args = parser.parse_args()
    main(config_path=args.config, run_dir=args.run_dir)
```

- [ ] **18.4** Implement `src/battle_royale/interfaces/cli/evaluate.py`:

```python
"""
Usage:
    python -m battle_royale.interfaces.cli.evaluate --checkpoint runs/checkpoint_100k --num-agents 6
"""
from __future__ import annotations

import argparse

from stable_baselines3 import PPO

from battle_royale.infrastructure.config.yaml_loader import load_config
from battle_royale.infrastructure.physics.mujoco_env import MuJoCoEnvironment
from battle_royale.interfaces.pettingzoo.env import BattleRoyaleEnv
from battle_royale.application.training.snapshot_pool import SnapshotPool
from battle_royale.application.evaluation.evaluator import Evaluator


def main(checkpoint_path: str, num_agents: int, config_path: str = "config/default.yaml") -> dict:
    config = load_config(config_path)
    config.training.num_agents = num_agents

    def env_factory(n: int):
        mujoco_env = MuJoCoEnvironment(config=config)
        return BattleRoyaleEnv(env=mujoco_env, config=config)

    model = PPO.load(checkpoint_path)
    pool = SnapshotPool(save_dir="/tmp/eval_snapshots")

    # Null logger for evaluation
    class _PrintLogger:
        def log(self, metrics, step):
            print(f"[eval step={step}] {metrics}")
        def save_artifact(self, path, name):
            pass

    evaluator = Evaluator(env_factory=env_factory, snapshot_pool=pool, logger=_PrintLogger())
    metrics = evaluator.evaluate(model=model, num_agents=num_agents, num_episodes=10)
    print(metrics)
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Battle Royale agents")
    parser.add_argument("--checkpoint", required=True, help="Path to SB3 model checkpoint")
    parser.add_argument("--num-agents", type=int, default=4, help="Number of agents to evaluate with")
    parser.add_argument("--config", default="config/default.yaml", help="Path to YAML config")
    args = parser.parse_args()
    main(checkpoint_path=args.checkpoint, num_agents=args.num_agents, config_path=args.config)
```

- [ ] **18.5** Run, expect pass:

```bash
poetry run pytest tests/unit/interfaces/test_cli.py -v
```

Expected:
```
5 passed in 0.xxs
```

- [ ] **18.6** Commit:

```bash
git add src/battle_royale/interfaces/cli/train.py src/battle_royale/interfaces/cli/evaluate.py tests/unit/interfaces/test_cli.py
git commit -m "feat(interfaces): add train and evaluate CLI entry points with DI wiring"
```

---

## Task 19 — Infrastructure: `VideoRecorder`

### Steps

- [ ] **19.1** Write the failing test at `tests/unit/infrastructure/test_video_recorder.py`:

```python
import numpy as np
import pytest
from unittest.mock import patch, MagicMock
from battle_royale.infrastructure.recording.video_recorder import VideoRecorder


def test_video_recorder_can_be_constructed(tmp_path):
    recorder = VideoRecorder(output_path=str(tmp_path / "video.mp4"), fps=30)
    assert recorder is not None


@patch("battle_royale.infrastructure.recording.video_recorder.mediapy")
def test_add_frame_stores_frames(mock_mediapy, tmp_path):
    recorder = VideoRecorder(output_path=str(tmp_path / "video.mp4"), fps=30)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    recorder.add_frame(frame)
    assert recorder.frame_count == 1


@patch("battle_royale.infrastructure.recording.video_recorder.mediapy")
def test_save_calls_mediapy_write_video(mock_mediapy, tmp_path):
    recorder = VideoRecorder(output_path=str(tmp_path / "video.mp4"), fps=30)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    recorder.add_frame(frame)
    recorder.save()
    mock_mediapy.write_video.assert_called_once()


@patch("battle_royale.infrastructure.recording.video_recorder.mediapy")
def test_save_passes_correct_fps(mock_mediapy, tmp_path):
    recorder = VideoRecorder(output_path=str(tmp_path / "video.mp4"), fps=60)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    recorder.add_frame(frame)
    recorder.save()
    call_kwargs = mock_mediapy.write_video.call_args
    assert call_kwargs[1].get("fps") == 60 or call_kwargs[0][2] == 60
```

- [ ] **19.2** Run, expect failure:

```bash
poetry run pytest tests/unit/infrastructure/test_video_recorder.py -v
```

Expected failure:
```
ModuleNotFoundError: No module named 'battle_royale.infrastructure.recording.video_recorder'
```

- [ ] **19.3** Implement `src/battle_royale/infrastructure/recording/video_recorder.py`:

```python
from __future__ import annotations

import numpy as np
import mediapy


class VideoRecorder:
    def __init__(self, output_path: str, fps: int = 30) -> None:
        self._output_path = output_path
        self._fps = fps
        self._frames: list[np.ndarray] = []

    @property
    def frame_count(self) -> int:
        return len(self._frames)

    def add_frame(self, frame: np.ndarray) -> None:
        self._frames.append(frame)

    def save(self) -> None:
        if not self._frames:
            return
        mediapy.write_video(self._output_path, self._frames, fps=self._fps)
```

- [ ] **19.4** Run, expect pass:

```bash
poetry run pytest tests/unit/infrastructure/test_video_recorder.py -v
```

Expected:
```
4 passed in 0.xxs
```

- [ ] **19.5** Commit:

```bash
git add src/battle_royale/infrastructure/recording/video_recorder.py tests/unit/infrastructure/test_video_recorder.py
git commit -m "feat(infra): add VideoRecorder wrapping mediapy for episode recording"
```

---

## Task 20 — Full Test Suite and Coverage Gate

### Steps

- [ ] **20.1** Run the entire test suite:

```bash
poetry run pytest tests/ -v --tb=short
```

Expected: All tests pass. No failures.

- [ ] **20.2** Run with coverage:

```bash
poetry run pytest tests/ --cov=battle_royale --cov-report=term-missing --cov-fail-under=80
```

Expected: Coverage at or above 80% for the `battle_royale` package.

- [ ] **20.3** Add coverage config to `pyproject.toml`:

```toml
[tool.coverage.run]
source = ["src/battle_royale"]
omit = ["*/interfaces/cli/*"]

[tool.coverage.report]
fail_under = 80
```

- [ ] **20.4** Verify `main.py` backward compatibility one final time:

```bash
poetry run python3 -c "
from src.battle_royale.infrastructure.physics import load_model, load_data, render
model = load_model()
data = load_data(model)
print('main.py still works: ok')
"
```

Expected: `main.py still works: ok`

- [ ] **20.5** Final commit:

```bash
git add pyproject.toml
git commit -m "chore: add coverage config with 80% gate; all tests passing"
```

---

## Task Execution Order Summary

| Task | What Gets Implemented | Depends On |
|------|----------------------|------------|
| 1 | Dependencies + pytest | Nothing |
| 2 | Agent, Arena entities | Task 1 |
| 3 | IBattleRoyaleEnv, ILogger, IPolicy protocols | Task 2 |
| 4 | EliminationService | Task 2 |
| 5 | ObservationBuilder | Task 2 |
| 6 | RewardCalculator | Task 2 |
| 7 | Config dataclasses + YamlLoader | Task 1 |
| 8 | XMLBuilder (N agents) | Task 7 |
| 9 | MuJoCoEnvironment | Tasks 2, 4, 6, 7, 8 |
| 10 | EloRatingSystem | Task 1 |
| 11 | MetricsTracker | Tasks 3, 10 |
| 12 | SnapshotPool | Task 1 |
| 13 | WandBLogger | Task 3 |
| 14 | Integration tests | Tasks 5, 9 |
| 15 | BattleRoyaleEnv (PettingZoo) | Tasks 3, 5, 7, 9 |
| 16 | Trainer (SB3) | Tasks 11, 12, 15 |
| 17 | Evaluator | Tasks 12, 13, 15 |
| 18 | CLI train + evaluate | Tasks 7, 9, 11, 12, 13, 15, 16, 17 |
| 19 | VideoRecorder | Task 1 |
| 20 | Full suite + coverage gate | All tasks |

---

### Critical Files for Implementation

- `/home/dvir/projects/mujoco-battle-royale/src/battle_royale/infrastructure/physics/xml_builder.py`
- `/home/dvir/projects/mujoco-battle-royale/src/battle_royale/infrastructure/physics/mujoco_env.py`
- `/home/dvir/projects/mujoco-battle-royale/src/battle_royale/interfaces/pettingzoo/env.py`
- `/home/dvir/projects/mujoco-battle-royale/src/battle_royale/infrastructure/config/yaml_loader.py`
- `/home/dvir/projects/mujoco-battle-royale/pyproject.toml`