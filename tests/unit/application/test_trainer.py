from unittest.mock import MagicMock, patch

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


def test_trainer_can_be_constructed(
    mock_env, mock_logger, mock_snapshot_pool, mock_tracker, config
):
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
def test_trainer_run_creates_ppo_model(
    mock_supersuit,
    mock_ppo,
    mock_env,
    mock_logger,
    mock_snapshot_pool,
    mock_tracker,
    config,
):
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
def test_trainer_run_calls_supersuit_wrapper(
    mock_supersuit,
    mock_ppo,
    mock_env,
    mock_logger,
    mock_snapshot_pool,
    mock_tracker,
    config,
):
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
