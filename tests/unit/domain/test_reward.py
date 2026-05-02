import numpy as np
import pytest

from battle_royale.domain.entities.agent import Agent
from battle_royale.domain.services.reward import RewardCalculator


def make_agent(agent_id: str, alive: bool) -> Agent:
    return Agent(id=agent_id, position=np.zeros(2), velocity=np.zeros(2), alive=alive)


def test_survival_reward_when_nothing_changes():
    prev = {"a0": make_agent("a0", True), "a1": make_agent("a1", True)}
    curr = {"a0": make_agent("a0", True), "a1": make_agent("a1", True)}
    reward = RewardCalculator.compute(prev, curr, "a0")
    assert pytest.approx(reward, abs=1e-6) == 0.01


def test_reward_for_eliminating_one_opponent():
    prev = {"a0": make_agent("a0", True), "a1": make_agent("a1", True), "a2": make_agent("a2", True)}
    curr = {"a0": make_agent("a0", True), "a1": make_agent("a1", False), "a2": make_agent("a2", True)}
    reward = RewardCalculator.compute(prev, curr, "a0")
    assert pytest.approx(reward, abs=1e-6) == 1.01


def test_reward_for_eliminating_two_opponents():
    prev = {f"a{i}": make_agent(f"a{i}", True) for i in range(4)}
    curr = {f"a{i}": make_agent(f"a{i}", i not in (1, 2)) for i in range(4)}
    reward = RewardCalculator.compute(prev, curr, "a0")
    assert pytest.approx(reward, abs=1e-6) == 2.01


def test_penalty_when_self_eliminated():
    prev = {"a0": make_agent("a0", True), "a1": make_agent("a1", True)}
    curr = {"a0": make_agent("a0", False), "a1": make_agent("a1", True)}
    reward = RewardCalculator.compute(prev, curr, "a0")
    assert pytest.approx(reward, abs=1e-6) == -1.0


def test_no_survival_bonus_when_already_dead_before():
    # If agent was already dead in prev, it was dead last step — no survival
    prev = {"a0": make_agent("a0", False), "a1": make_agent("a1", True)}
    curr = {"a0": make_agent("a0", False), "a1": make_agent("a1", True)}
    reward = RewardCalculator.compute(prev, curr, "a0")
    assert pytest.approx(reward, abs=1e-6) == 0.0


def test_already_dead_opponent_elimination_not_double_counted():
    # a1 was already dead in prev — should not count as a new elimination
    prev = {"a0": make_agent("a0", True), "a1": make_agent("a1", False)}
    curr = {"a0": make_agent("a0", True), "a1": make_agent("a1", False)}
    reward = RewardCalculator.compute(prev, curr, "a0")
    assert pytest.approx(reward, abs=1e-6) == 0.01