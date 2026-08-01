import dataclasses
import os
from dataclasses import dataclass, field

import yaml


@dataclass
class ArenaConfig:
    radius: float = 3.0
    wall_height: float = 0.5
    # Battle-royale "storm": the arena boundary holds full size for
    # ``shrink_delay`` steps (a grace / fighting phase), then shrinks from
    # ``radius`` to ``radius * min_radius_frac`` linearly over ``shrink_steps``
    # env steps. The grace period gives a stable arena in which pushing -- not
    # the closing boundary -- is the primary way to eliminate opponents; the
    # later shrink then breaks any stalemate. Defaults (frac 1.0, steps 0) mean
    # no shrink.
    min_radius_frac: float = 1.0
    shrink_steps: int = 0
    shrink_delay: int = 0
    # Joint velocity damping. High (~8) makes agents "sticky" -- they stop
    # instantly, so a shove barely moves the opponent. Lower (~3-4) lets a shove
    # send the opponent sliding toward the edge (visible sumo pushing), at the
    # cost of trickier self-control.
    damping: float = 8.0


@dataclass
class TrainingConfig:
    num_agents: int = 4
    total_steps: int = 1_000_000
    snapshot_interval: int = 10_000
    max_force: float = 10.0
    snapshot_pool_size: int = 20
    episode_max_steps: int = 400


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
    valid_keys = {f.name for f in dataclasses.fields(dataclass_instance)}
    for key, value in data.items():
        if key not in valid_keys:
            raise ValueError(
                f"Unknown config key '{key}' in {type(dataclass_instance).__name__}"
            )
        setattr(dataclass_instance, key, value)


def load_config(path: str) -> Config:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise TypeError(
            f"Config file must contain a YAML mapping, got {type(data).__name__}"
        )
    config = Config()
    for section, instance in (
        ("arena", config.arena),
        ("training", config.training),
        ("ppo", config.ppo),
        ("evaluation", config.evaluation),
    ):
        if section in data:
            if not isinstance(data[section], dict):
                raise TypeError(
                    f"'{section}' section must be a mapping, got {type(data[section]).__name__}"
                )
            _apply_dict(instance, data[section])
    return config
