import numpy as np
import pytest

from battle_royale.domain.entities.agent import Agent
from battle_royale.domain.services.reward import RewardCalculator


def make_agent(agent_id: str, alive: bool) -> Agent:
    return Agent(id=agent_id, position=np.zeros(2), velocity=np.zeros(2), alive=alive)


def make_agent_at(agent_id: str, x: float, alive: bool = True) -> Agent:
    return Agent(
        id=agent_id,
        position=np.array([x, 0.0]),
        velocity=np.zeros(2),
        alive=alive,
    )


def test_idle_alive_step_nets_small_negative():
    # Alive, opponents still alive: survival (+0.001) + time penalty (-0.003),
    # no win bonus. Net slightly negative so idling is mildly discouraged.
    prev = {f"a{i}": make_agent(f"a{i}", True) for i in range(3)}
    curr = {f"a{i}": make_agent(f"a{i}", True) for i in range(3)}
    reward = RewardCalculator.compute(prev, curr, "a0")
    assert pytest.approx(reward, abs=1e-6) == -0.002


def test_reward_for_eliminating_one_opponent():
    prev = {
        "a0": make_agent("a0", True),
        "a1": make_agent("a1", True),
        "a2": make_agent("a2", True),
    }
    curr = {
        "a0": make_agent("a0", True),
        "a1": make_agent("a1", False),
        "a2": make_agent("a2", True),
    }
    # elimination (+1) + net idle (-0.002); a2 still alive so no win bonus.
    reward = RewardCalculator.compute(prev, curr, "a0")
    assert pytest.approx(reward, abs=1e-6) == 0.998


def test_reward_for_eliminating_two_opponents():
    prev = {f"a{i}": make_agent(f"a{i}", True) for i in range(4)}
    curr = {f"a{i}": make_agent(f"a{i}", i not in (1, 2)) for i in range(4)}
    # two eliminations (+2) + net idle (-0.002); a3 still alive so no win bonus.
    reward = RewardCalculator.compute(prev, curr, "a0")
    assert pytest.approx(reward, abs=1e-6) == 1.998


def test_win_bonus_when_last_survivor():
    # a0 eliminates the last remaining opponent and becomes sole survivor.
    prev = {f"a{i}": make_agent(f"a{i}", i in (0, 3)) for i in range(4)}
    curr = {f"a{i}": make_agent(f"a{i}", i == 0) for i in range(4)}
    # elimination of a3 (+1) + net idle (-0.002) + win bonus (+10).
    reward = RewardCalculator.compute(prev, curr, "a0")
    assert pytest.approx(reward, abs=1e-6) == 10.998


def test_push_shaping_rewards_driving_opponent_outward():
    # Opponent moves 1.0 -> 1.5 outward with the learner stationary at centre:
    # push 2.0*0.5=+1.0, approach -0.4*(1.5-1.0)=-0.2 (gap grew), net idle -0.002.
    prev = {"a0": make_agent_at("a0", 0.0), "a1": make_agent_at("a1", 1.0)}
    curr = {"a0": make_agent_at("a0", 0.0), "a1": make_agent_at("a1", 1.5)}
    reward = RewardCalculator.compute(prev, curr, "a0")
    assert pytest.approx(reward, abs=1e-6) == 1.0 - 0.2 - 0.002


def test_aggression_scale_multiplies_push_and_approach_only():
    # Same push+approach setup as above but with aggression_scale=2.0: the push
    # (+1.0) and approach (-0.2) shaping double to +2.0 and -0.4; the non-shaping
    # net idle (-0.002) is unchanged.
    prev = {"a0": make_agent_at("a0", 0.0), "a1": make_agent_at("a1", 1.0)}
    curr = {"a0": make_agent_at("a0", 0.0), "a1": make_agent_at("a1", 1.5)}
    reward = RewardCalculator.compute(prev, curr, "a0", aggression_scale=2.0)
    assert pytest.approx(reward, abs=1e-6) == 2.0 - 0.4 - 0.002


def test_aggression_scale_default_is_one():
    # Default (unset) scale reproduces the baseline push+approach reward exactly.
    prev = {"a0": make_agent_at("a0", 0.0), "a1": make_agent_at("a1", 1.0)}
    curr = {"a0": make_agent_at("a0", 0.0), "a1": make_agent_at("a1", 1.5)}
    assert RewardCalculator.compute(
        prev, curr, "a0"
    ) == RewardCalculator.compute(prev, curr, "a0", aggression_scale=1.0)


def test_push_shaping_penalizes_opponent_moving_inward():
    # Opponent retreats 1.5 -> 1.0: push 2.0*-0.5=-1.0, approach +0.2 (gap
    # shrank), net idle -0.002.
    prev = {"a0": make_agent_at("a0", 0.0), "a1": make_agent_at("a1", 1.5)}
    curr = {"a0": make_agent_at("a0", 0.0), "a1": make_agent_at("a1", 1.0)}
    reward = RewardCalculator.compute(prev, curr, "a0")
    assert pytest.approx(reward, abs=1e-6) == -1.0 + 0.2 - 0.002


def test_approach_shaping_rewards_closing_on_opponent():
    # Learner advances 0.0 -> 0.5 toward a stationary opponent at 2.0: gap
    # shrinks 2.0 -> 1.5 so approach +0.4*0.5=+0.2; opponent's own distance
    # unchanged so push 0; net idle -0.002.
    prev = {"a0": make_agent_at("a0", 0.0), "a1": make_agent_at("a1", 2.0)}
    curr = {"a0": make_agent_at("a0", 0.5), "a1": make_agent_at("a1", 2.0)}
    reward = RewardCalculator.compute(prev, curr, "a0")
    assert pytest.approx(reward, abs=1e-6) == 0.2 - 0.002


def test_edge_penalty_in_outer_ring():
    # Learner sits at |p|=2.9 (outer ring; safe radius = 0.92*3 = 2.76):
    # edge penalty 1.0*(2.76-2.9) = -0.14, plus net idle -0.002. Opponent at
    # centre and unmoved, so push/approach are 0.
    prev = {"a0": make_agent_at("a0", 2.9), "a1": make_agent_at("a1", 0.0)}
    curr = {"a0": make_agent_at("a0", 2.9), "a1": make_agent_at("a1", 0.0)}
    reward = RewardCalculator.compute(prev, curr, "a0")
    assert pytest.approx(reward, abs=1e-6) == -0.14 - 0.002


def test_no_edge_penalty_inside_safe_radius():
    # At |p|=2.0 (inside 2.76) there is no edge penalty; just net idle.
    prev = {"a0": make_agent_at("a0", 2.0), "a1": make_agent_at("a1", 0.0)}
    curr = {"a0": make_agent_at("a0", 2.0), "a1": make_agent_at("a1", 0.0)}
    reward = RewardCalculator.compute(prev, curr, "a0")
    assert pytest.approx(reward, abs=1e-6) == -0.002


def test_penalty_when_self_eliminated():
    prev = {"a0": make_agent("a0", True), "a1": make_agent("a1", True)}
    curr = {"a0": make_agent("a0", False), "a1": make_agent("a1", True)}
    # Death penalty only; no survival or win bonus for a dead agent.
    reward = RewardCalculator.compute(prev, curr, "a0")
    assert pytest.approx(reward, abs=1e-6) == -1.0


def test_no_survival_bonus_when_already_dead_before():
    # If agent was already dead in prev, it was dead last step — no reward.
    prev = {"a0": make_agent("a0", False), "a1": make_agent("a1", True)}
    curr = {"a0": make_agent("a0", False), "a1": make_agent("a1", True)}
    reward = RewardCalculator.compute(prev, curr, "a0")
    assert pytest.approx(reward, abs=1e-6) == 0.0


def test_already_dead_opponent_elimination_not_double_counted():
    # a1 was already dead in prev — should not count as a new elimination.
    # a2 is still alive, so no win bonus either; isolates the double-count check.
    prev = {
        "a0": make_agent("a0", True),
        "a1": make_agent("a1", False),
        "a2": make_agent("a2", True),
    }
    curr = {
        "a0": make_agent("a0", True),
        "a1": make_agent("a1", False),
        "a2": make_agent("a2", True),
    }
    reward = RewardCalculator.compute(prev, curr, "a0")
    assert pytest.approx(reward, abs=1e-6) == -0.002
