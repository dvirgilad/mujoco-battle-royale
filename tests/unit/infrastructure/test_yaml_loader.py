import pytest
from battle_royale.infrastructure.config.yaml_loader import (
    Config,
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


def test_load_config_unknown_key_raises(tmp_path):
    yaml_file = tmp_path / "bad.yaml"
    yaml_file.write_text("arena:\n  typo_key: 99\n")
    with pytest.raises(ValueError, match="Unknown config key 'typo_key'"):
        load_config(str(yaml_file))


def test_load_config_section_not_mapping_raises(tmp_path):
    yaml_file = tmp_path / "bad.yaml"
    yaml_file.write_text("arena: 3.0\n")
    with pytest.raises(TypeError, match="'arena' section must be a mapping"):
        load_config(str(yaml_file))
