import numpy as np
import pytest

from battle_royale.domain.entities.agent import Agent
from battle_royale.domain.entities.arena import Arena


def test_agent_is_frozen():
    agent = Agent(
        id="a0",
        position=np.array([1.0, 0.0]),
        velocity=np.array([0.0, 0.5]),
        alive=True,
    )
    with pytest.raises((AttributeError, TypeError)):
        agent.alive = False  # type: ignore[misc]


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
        arena.radius = 5.0  # type: ignore[misc]


def test_arena_fields():
    arena = Arena(radius=3.0)
    assert arena.radius == 3.0
