import numpy as np
import pytest

from battle_royale.domain.entities.agent import Agent
from battle_royale.domain.entities.arena import Arena
from battle_royale.domain.services.elimination import EliminationService


@pytest.fixture
def arena():
    return Arena(radius=3.0)


def make_agent(x: float, y: float, alive: bool = True) -> Agent:
    return Agent(
        id="a0", position=np.array([x, y]), velocity=np.zeros(2), alive=alive
    )


def test_agent_inside_arena_not_eliminated(arena):
    agent = make_agent(1.0, 0.0)
    assert EliminationService.is_eliminated(agent, arena) is False


def test_agent_exactly_on_boundary_not_eliminated(arena):
    # norm == radius → not eliminated (strictly greater than)
    agent = make_agent(3.0, 0.0)
    assert EliminationService.is_eliminated(agent, arena) is False


def test_agent_outside_arena_eliminated(arena):
    agent = make_agent(3.1, 0.0)
    assert EliminationService.is_eliminated(agent, arena) is True


def test_dead_agent_considered_eliminated_regardless(arena):
    agent = Agent(
        id="a0", position=np.array([0.0, 0.0]), velocity=np.zeros(2), alive=False
    )
    assert EliminationService.is_eliminated(agent, arena) is True


def test_diagonal_position_eliminated(arena):
    # sqrt(2^2 + 2^2) = 2.828 < 3.0 → not eliminated
    agent = make_agent(2.0, 2.0)
    assert EliminationService.is_eliminated(agent, arena) is False


def test_diagonal_position_outside(arena):
    # sqrt(2.2^2 + 2.2^2) = 3.111 > 3.0 → eliminated
    agent = make_agent(2.2, 2.2)
    assert EliminationService.is_eliminated(agent, arena) is True
