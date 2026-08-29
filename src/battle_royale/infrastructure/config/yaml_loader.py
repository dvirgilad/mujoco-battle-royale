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
    # Collision-cylinder density (kg/m^3). MuJoCo's default 1000 gives a ~7 kg
    # agent whose passive braking distance from terminal velocity (v * m / damping)
    # is ~6 m -- far larger than the ~1-2 m arena, so an agent at speed physically
    # CANNOT stop inside the ring. That single fact drives both self-ejection
    # (overshooting the edge) and caution (moving slowly is the only way not to
    # fly out). Lowering density shrinks the braking distance (linear in mass) AND
    # makes shoves displace opponents more -- helping self-control and pushing at
    # once. Default 1000 preserves the original physics.
    agent_density: float = 1000.0
    # Per-episode spawn perturbation (radians of angular jitter; the same
    # fraction is reused for radial jitter). Agents otherwise spawn on a perfect
    # regular n-gon, where the two adjacent neighbours are EXACTLY equidistant --
    # so the distance-sorted observation breaks that tie by agent index, giving
    # each slot a fixed clockwise/counter-clockwise "first neighbour" and hence a
    # per-slot win bias. Jittering the spawns removes the ties (the sort becomes
    # canonical: nearest first) and stops any slot mapping to a fixed geometric
    # role, evening out the win distribution. 0 (default) = exact n-gon.
    spawn_jitter: float = 0.0


@dataclass
class TrainingConfig:
    num_agents: int = 4
    total_steps: int = 1_000_000
    snapshot_interval: int = 10_000
    max_force: float = 10.0
    snapshot_pool_size: int = 20
    episode_max_steps: int = 400
    # Agent-count curriculum. When > 0, training starts with this many agents
    # and ramps up to ``num_agents`` (see Trainer). Pushing is learnable at n=2
    # (no melee risk) but collapses to a cautious no-op if training starts
    # directly at n=4; ramping lets the policy carry the shove behaviour into the
    # melee instead of re-discovering caution from scratch. 0 disables the ramp
    # (train at ``num_agents`` throughout) so default/storm runs are unaffected.
    curriculum_start_agents: int = 0
    # Terminal penalty applied to a survivor when the episode TIMES OUT with more
    # than one agent still alive (a draw). OpenAI's sumo makes a draw as bad as a
    # loss (both -1000), which destroys the cautious "don't commit, run out the
    # clock" equilibrium -- if not-winning costs the same as losing, the only good
    # move is to attack. 0 (default) = draws are free (old behaviour).
    draw_penalty: float = 0.0
    # When True, the training observation shuffles the neighbour list before the
    # nearest-first sort, so genuinely-tied neighbours (e.g. the two adjacent
    # agents on a symmetric spawn) are ordered randomly rather than by agent
    # index. This makes the policy order-invariant and removes the per-slot win
    # bias WITHOUT perturbing the physics (unlike spawn jitter, which collapsed
    # dominance). Default False so existing runs/evals are unchanged.
    shuffle_neighbors: bool = False


@dataclass
class PPOConfig:
    lr: float = 3e-4
    n_steps: int = 2048
    batch_size: int = 64
    clip_range: float = 0.2
    n_epochs: int = 10
    # Discount. A terminal draw/win/lose signal at step T is worth ~gamma**T from
    # the episode start, so with long episodes a low gamma makes terminal rewards
    # (like the draw penalty) invisible until the very end. A higher gamma
    # lengthens the effective horizon so the terminal incentive reaches back into
    # the mid-game where the caution actually happens.
    gamma: float = 0.99


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
