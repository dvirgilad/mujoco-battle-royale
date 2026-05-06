import numpy as np
import pytest
from unittest.mock import MagicMock, patch
from battle_royale.application.evaluation.evaluator import Evaluator


@pytest.fixture
def mock_env():
    env = MagicMock()
    env.agents = ["agent_0", "agent_1", "agent_2", "agent_3"]
    env.reset.return_value = (
        {
            "agent_0": np.zeros(17),
            "agent_1": np.zeros(17),
            "agent_2": np.zeros(17),
            "agent_3": np.zeros(17),
        },
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
    env.terminations = {
        "agent_0": False,
        "agent_1": True,
        "agent_2": True,
        "agent_3": True,
    }
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
    def env_factory(n):
        return mock_env

    evaluator = Evaluator(
        env_factory=env_factory, snapshot_pool=mock_pool, logger=mock_logger
    )
    assert evaluator is not None


def test_evaluate_returns_metrics_dict(mock_env, mock_model, mock_pool, mock_logger):
    def env_factory(n):
        return mock_env

    evaluator = Evaluator(
        env_factory=env_factory, snapshot_pool=mock_pool, logger=mock_logger
    )
    metrics = evaluator.evaluate(model=mock_model, num_agents=4, num_episodes=3)
    assert isinstance(metrics, dict)
    assert "win_rate" in metrics or "mean_episode_length" in metrics


def test_evaluate_calls_logger(mock_env, mock_model, mock_pool, mock_logger):
    def env_factory(n):
        return mock_env

    evaluator = Evaluator(
        env_factory=env_factory, snapshot_pool=mock_pool, logger=mock_logger
    )
    evaluator.evaluate(model=mock_model, num_agents=4, num_episodes=2)
    mock_logger.log.assert_called()


@patch("stable_baselines3.PPO")
def test_evaluate_loads_opponent_from_pool(
    mock_ppo, mock_env, mock_model, mock_logger, tmp_path
):
    opponent = MagicMock()
    opponent.predict.return_value = (np.zeros(2), None)
    mock_ppo.load.return_value = opponent

    pool = MagicMock()
    pool.is_empty.return_value = False
    pool.sample_path.return_value = tmp_path / "snapshot_000001"

    evaluator = Evaluator(
        env_factory=lambda _: mock_env, snapshot_pool=pool, logger=mock_logger
    )
    evaluator.evaluate(model=mock_model, num_agents=4, num_episodes=1)

    mock_ppo.load.assert_called_once()
    opponent.predict.assert_called()
