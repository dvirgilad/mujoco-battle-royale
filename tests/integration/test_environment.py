import numpy as np
import pytest

from battle_royale.domain.entities.arena import Arena
from battle_royale.domain.services.observation import ObservationBuilder
from battle_royale.infrastructure.config.yaml_loader import Config
from battle_royale.infrastructure.physics.mujoco_env import MuJoCoEnvironment


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
        actions = {
            aid: np.random.uniform(-1, 1, 2).astype(np.float32) for aid in agents
        }
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
