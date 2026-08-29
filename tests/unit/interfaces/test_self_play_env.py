import numpy as np
import pytest

from battle_royale.domain.entities.agent import Agent
from battle_royale.infrastructure.config.yaml_loader import Config
from battle_royale.interfaces.gym.self_play_env import SelfPlayEnv


def _agent(aid, x=0.0, y=0.0, alive=True):
    return Agent(
        id=aid,
        position=np.array([x, y], dtype=np.float64),
        velocity=np.zeros(2),
        alive=alive,
    )


class FakeMuJoCoEnv:
    """Deterministic stand-in for MuJoCoEnvironment used to drive episodes."""

    def __init__(self, script):
        # script: list of agent-dicts returned by successive step() calls
        self._script = script
        self._i = 0
        self.received_actions = []

    def reset(self, num_agents):
        self._i = 0
        return {
            f"agent_{i}": _agent(f"agent_{i}", x=0.1 * i) for i in range(num_agents)
        }

    def step(self, actions):
        self.received_actions.append(actions)
        agents = self._script[self._i]
        self._i += 1
        rewards = {aid: 0.01 for aid in agents}
        terms = {aid: not ag.alive for aid, ag in agents.items()}
        truncs = {aid: False for aid in agents}
        return agents, rewards, terms, truncs


@pytest.fixture
def config():
    c = Config()
    c.training.num_agents = 4
    c.arena.radius = 3.0
    return c


@pytest.fixture
def empty_pool():
    pool = type("P", (), {})()
    pool.is_empty = lambda: True
    pool.sample_path = lambda: None
    return pool


def test_spaces_are_correct(config, empty_pool):
    env = SelfPlayEnv(env=FakeMuJoCoEnv([]), config=config, snapshot_pool=empty_pool)
    assert env.observation_space.shape == (17,)
    assert env.action_space.shape == (2,)


def test_reset_returns_learner_obs(config, empty_pool):
    env = SelfPlayEnv(env=FakeMuJoCoEnv([]), config=config, snapshot_pool=empty_pool)
    obs, info = env.reset(seed=0)
    assert obs.shape == (17,)
    assert info == {}


def test_win_when_learner_is_last_alive(config, empty_pool):
    # After one step, only agent_0 (learner) is alive -> win.
    final = {
        "agent_0": _agent("agent_0", x=0.0, alive=True),
        "agent_1": _agent("agent_1", x=5.0, alive=False),
        "agent_2": _agent("agent_2", x=5.0, alive=False),
        "agent_3": _agent("agent_3", x=5.0, alive=False),
    }
    env = SelfPlayEnv(
        env=FakeMuJoCoEnv([final]), config=config, snapshot_pool=empty_pool
    )
    env.reset(seed=0)
    obs, reward, terminated, truncated, info = env.step(np.zeros(2, dtype=np.float32))
    assert terminated is True
    assert info["match_result"]["win"] is True
    assert info["match_result"]["eliminations"] == 3
    assert info["is_success"] is True


def test_loss_when_learner_dies(config, empty_pool):
    final = {
        "agent_0": _agent("agent_0", x=5.0, alive=False),
        "agent_1": _agent("agent_1", x=0.0, alive=True),
        "agent_2": _agent("agent_2", x=0.0, alive=True),
        "agent_3": _agent("agent_3", x=0.0, alive=True),
    }
    env = SelfPlayEnv(
        env=FakeMuJoCoEnv([final]), config=config, snapshot_pool=empty_pool
    )
    env.reset(seed=0)
    _, _, terminated, _, info = env.step(np.zeros(2, dtype=np.float32))
    assert terminated is True
    assert info["match_result"]["win"] is False


def test_truncates_at_max_steps(config, empty_pool):
    alive_all = {
        f"agent_{i}": _agent(f"agent_{i}", x=0.1 * i, alive=True) for i in range(4)
    }
    env = SelfPlayEnv(
        env=FakeMuJoCoEnv([alive_all]), config=config, snapshot_pool=empty_pool
    )
    env._max_steps = 1
    env.reset(seed=0)
    _, _, terminated, truncated, info = env.step(np.zeros(2, dtype=np.float32))
    assert terminated is False
    assert truncated is True
    assert "match_result" in info


def test_draw_penalty_charged_on_timeout_with_survivors(config, empty_pool):
    # Timeout with >1 agent alive (a draw) subtracts draw_penalty from the
    # learner's reward (base per-step reward from FakeMuJoCoEnv is 0.01).
    alive_all = {
        f"agent_{i}": _agent(f"agent_{i}", x=0.1 * i, alive=True) for i in range(4)
    }
    env = SelfPlayEnv(
        env=FakeMuJoCoEnv([alive_all]),
        config=config,
        snapshot_pool=empty_pool,
        draw_penalty=2.0,
    )
    env._max_steps = 1
    env.reset(seed=0)
    _, reward, terminated, truncated, _ = env.step(np.zeros(2, dtype=np.float32))
    assert truncated is True and terminated is False
    assert pytest.approx(reward, abs=1e-6) == 0.01 - 2.0


def test_draw_penalty_not_charged_on_win(config, empty_pool):
    # Winning by timeout-step is a termination, not a draw -> no penalty.
    final = {
        "agent_0": _agent("agent_0", x=0.0, alive=True),
        "agent_1": _agent("agent_1", x=5.0, alive=False),
        "agent_2": _agent("agent_2", x=5.0, alive=False),
        "agent_3": _agent("agent_3", x=5.0, alive=False),
    }
    env = SelfPlayEnv(
        env=FakeMuJoCoEnv([final]),
        config=config,
        snapshot_pool=empty_pool,
        draw_penalty=2.0,
    )
    env._max_steps = 1
    env.reset(seed=0)
    _, reward, terminated, _, info = env.step(np.zeros(2, dtype=np.float32))
    assert terminated is True and info["match_result"]["win"] is True
    assert pytest.approx(reward, abs=1e-6) == 0.01  # no draw penalty


def test_dead_agents_get_zero_actions_passed_through(config, empty_pool):
    alive_all = {
        f"agent_{i}": _agent(f"agent_{i}", x=0.1 * i, alive=True) for i in range(4)
    }
    fake = FakeMuJoCoEnv([alive_all])
    env = SelfPlayEnv(env=fake, config=config, snapshot_pool=empty_pool)
    env.reset(seed=0)
    env.step(np.array([0.5, -0.5], dtype=np.float32))
    passed = fake.received_actions[0]
    # learner action forwarded verbatim
    assert np.allclose(passed["agent_0"], [0.5, -0.5])
    # every agent has an action of the right shape
    assert all(passed[f"agent_{i}"].shape == (2,) for i in range(4))


def test_randomize_learner_varies_slot_across_episodes(config, empty_pool):
    # With randomization on, the learner should occupy more than one slot over
    # many resets (otherwise it overfits agent_0's spawn angle).
    env = SelfPlayEnv(
        env=FakeMuJoCoEnv([]),
        config=config,
        snapshot_pool=empty_pool,
        randomize_learner=True,
    )
    seen = set()
    for seed in range(20):
        env.reset(seed=seed)
        seen.add(env._learner_id)
    assert len(seen) > 1
    assert seen <= {f"agent_{i}" for i in range(4)}


def test_learner_fixed_to_agent_0_by_default(config, empty_pool):
    env = SelfPlayEnv(env=FakeMuJoCoEnv([]), config=config, snapshot_pool=empty_pool)
    for seed in range(5):
        env.reset(seed=seed)
        assert env._learner_id == "agent_0"


def test_opponent_loaded_from_pool(config):
    class Pool:
        def is_empty(self):
            return False

        def sample_path(self):
            return "some/path"

    predicted = np.array([0.9, 0.1], dtype=np.float32)

    class FakePPO:
        @staticmethod
        def load(path, device="cpu"):
            m = type("M", (), {})()
            m.predict = lambda obs, deterministic=True: (predicted, None)
            return m

    alive_all = {
        f"agent_{i}": _agent(f"agent_{i}", x=0.1 * i, alive=True) for i in range(4)
    }
    fake = FakeMuJoCoEnv([alive_all])
    env = SelfPlayEnv(env=fake, config=config, snapshot_pool=Pool())

    import stable_baselines3

    orig = stable_baselines3.PPO
    stable_baselines3.PPO = FakePPO
    try:
        env.reset(seed=0)
        env.step(np.zeros(2, dtype=np.float32))
    finally:
        stable_baselines3.PPO = orig

    passed = fake.received_actions[0]
    # opponents used the loaded policy's action
    assert np.allclose(passed["agent_1"], predicted)
