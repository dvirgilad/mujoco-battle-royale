import numpy as np
import pytest
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
