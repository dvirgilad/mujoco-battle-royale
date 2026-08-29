import numpy as np
import pytest
from battle_royale.infrastructure.physics.mujoco_env import MuJoCoEnvironment
from battle_royale.domain.entities.agent import Agent
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
    assert (
        set(obs.keys())
        == set(rewards.keys())
        == set(terminations.keys())
        == set(truncations.keys())
    )


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
    env.step(actions)
    agents_after = {a.id: a for a in env.get_agents()}
    assert agents_after["agent_0"].velocity[0] > 0


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
