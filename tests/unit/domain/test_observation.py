import numpy as np
import pytest
from battle_royale.domain.entities.agent import Agent
from battle_royale.domain.entities.arena import Arena
from battle_royale.domain.services.observation import ObservationBuilder


def make_agent(
    agent_id: str,
    x: float,
    y: float,
    vx: float = 0.0,
    vy: float = 0.0,
    alive: bool = True,
) -> Agent:
    return Agent(
        id=agent_id, position=np.array([x, y]), velocity=np.array([vx, vy]), alive=alive
    )


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
    agents = [
        agent,
        make_agent("a1", 0.0, 0.0),
        make_agent("a2", 1.0, 1.0),
        make_agent("a3", -1.0, 0.5),
    ]
    obs = ObservationBuilder.build(agent, agents, arena)
    np.testing.assert_allclose(obs[0], 1.5, atol=1e-5)
    np.testing.assert_allclose(obs[1], -0.5, atol=1e-5)


def test_observation_own_velocity_elements_2_3(arena):
    agent = make_agent("a0", 0.0, 0.0, vx=2.0, vy=-1.0)
    agents = [
        agent,
        make_agent("a1", 0.5, 0.0),
        make_agent("a2", -0.5, 0.0),
        make_agent("a3", 0.0, 0.5),
    ]
    obs = ObservationBuilder.build(agent, agents, arena)
    np.testing.assert_allclose(obs[2], 2.0, atol=1e-5)
    np.testing.assert_allclose(obs[3], -1.0, atol=1e-5)


def test_observation_dist_to_boundary_element_4(arena):
    # agent at (1, 0): dist_to_boundary = 3.0 - 1.0 = 2.0
    agent = make_agent("a0", 1.0, 0.0)
    agents = [
        agent,
        make_agent("a1", 0.5, 0.0),
        make_agent("a2", -0.5, 0.0),
        make_agent("a3", 0.0, 0.5),
    ]
    obs = ObservationBuilder.build(agent, agents, arena)
    np.testing.assert_allclose(obs[4], 2.0, atol=1e-5)


def test_observation_three_nearest_neighbors_by_distance(arena):
    # a0 at origin; a1 closest, a2 middle, a3 furthest, a4 even further
    agent = make_agent("a0", 0.0, 0.0)
    a1 = make_agent("a1", 0.1, 0.0)  # nearest
    a2 = make_agent("a2", 0.5, 0.0)  # 2nd
    a3 = make_agent("a3", 1.0, 0.0)  # 3rd
    a4 = make_agent("a4", 2.0, 0.0)  # 4th — should be excluded
    all_agents = [agent, a1, a2, a3, a4]
    obs = ObservationBuilder.build(agent, all_agents, arena)
    # First neighbor rel_pos should be (0.1, 0.0)
    np.testing.assert_allclose(obs[5], 0.1, atol=1e-5)
    np.testing.assert_allclose(obs[6], 0.0, atol=1e-5)
    # Third neighbor rel_pos x should be 1.0
    np.testing.assert_allclose(obs[9], 1.0, atol=1e-5)


def test_observation_padded_with_zeros_when_fewer_than_three_neighbors(arena):
    agent = make_agent("a0", 0.0, 0.0)
    other = make_agent("a1", 1.0, 0.0, vx=2.0, vy=3.0)
    obs = ObservationBuilder.build(agent, [agent, other], arena)
    assert obs.shape == (17,)
    # First neighbor populated
    np.testing.assert_allclose(obs[5:7], [1.0, 0.0], atol=1e-5)
    np.testing.assert_allclose(obs[11:13], [2.0, 3.0], atol=1e-5)
    # Neighbor 2 and 3 slots are zero (padded)
    np.testing.assert_array_equal(obs[7:11], np.zeros(4, dtype=np.float32))
    np.testing.assert_array_equal(obs[13:17], np.zeros(4, dtype=np.float32))


def test_observation_dead_agents_excluded_from_neighbors(arena):
    agent = make_agent("a0", 0.0, 0.0)
    dead = make_agent("a1", 0.1, 0.0, alive=False)  # nearest but dead
    alive_far = make_agent("a2", 1.0, 0.0)
    obs = ObservationBuilder.build(agent, [agent, dead, alive_far], arena)
    # dead agent must not appear; first neighbor slot should be alive_far
    np.testing.assert_allclose(obs[5], 1.0, atol=1e-5)
    np.testing.assert_allclose(obs[6], 0.0, atol=1e-5)
    # second neighbor slot padded
    np.testing.assert_array_equal(obs[7:9], np.zeros(2, dtype=np.float32))
