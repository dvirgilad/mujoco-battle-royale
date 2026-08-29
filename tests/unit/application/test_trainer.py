from unittest.mock import MagicMock, patch

import pytest

from battle_royale.application.training.trainer import Trainer
from battle_royale.infrastructure.config.yaml_loader import Config


@pytest.fixture
def config():
    c = Config()
    c.training.num_agents = 4
    c.training.total_steps = 100
    c.training.snapshot_interval = 50
    c.ppo.n_steps = 8
    c.ppo.batch_size = 4
    return c


@pytest.fixture
def mock_env():
    return MagicMock()


@pytest.fixture
def mock_logger():
    return MagicMock()


@pytest.fixture
def mock_snapshot_pool():
    return MagicMock()


@pytest.fixture
def mock_tracker():
    return MagicMock()


def _make_trainer(env, logger, pool, tracker, config):
    return Trainer(
        env=env,
        logger=logger,
        snapshot_pool=pool,
        tracker=tracker,
        config=config,
    )


def test_trainer_can_be_constructed(
    mock_env, mock_logger, mock_snapshot_pool, mock_tracker, config
):
    trainer = _make_trainer(
        mock_env, mock_logger, mock_snapshot_pool, mock_tracker, config
    )
    assert trainer is not None


@patch("battle_royale.application.training.trainer.PPO")
@patch("battle_royale.application.training.trainer.DummyVecEnv")
def test_trainer_run_creates_and_learns_ppo(
    mock_vec, mock_ppo, mock_env, mock_logger, mock_snapshot_pool, mock_tracker, config
):
    model = MagicMock()
    mock_ppo.return_value = model

    trainer = _make_trainer(
        mock_env, mock_logger, mock_snapshot_pool, mock_tracker, config
    )
    returned = trainer.run()

    mock_ppo.assert_called_once()
    model.learn.assert_called_once()
    # total_timesteps flows from config
    assert (
        model.learn.call_args.kwargs["total_timesteps"] == config.training.total_steps
    )
    assert returned is model


@patch("battle_royale.application.training.trainer.PPO")
@patch("battle_royale.application.training.trainer.DummyVecEnv")
def test_trainer_saves_final_snapshot(
    mock_vec, mock_ppo, mock_env, mock_logger, mock_snapshot_pool, mock_tracker, config
):
    model = MagicMock()
    mock_ppo.return_value = model

    trainer = _make_trainer(
        mock_env, mock_logger, mock_snapshot_pool, mock_tracker, config
    )
    trainer.run()

    mock_snapshot_pool.save.assert_called_with(model, step=config.training.total_steps)


def test_callback_records_match_and_saves_snapshot(
    mock_env, mock_logger, mock_snapshot_pool, mock_tracker, config
):
    trainer = _make_trainer(
        mock_env, mock_logger, mock_snapshot_pool, mock_tracker, config
    )
    callback = trainer._make_callback()
    callback.model = MagicMock()
    callback.num_timesteps = 50
    callback.n_calls = config.training.snapshot_interval  # divisible -> save fires
    callback.locals = {
        "infos": [
            {"match_result": {"win": True, "length": 120, "eliminations": 3}},
            {},  # no result for this env -> ignored
        ]
    }

    assert callback._on_step() is True
    mock_tracker.record_match.assert_called_once_with(
        won=True, episode_length=120, eliminations=3, step=50
    )
    mock_snapshot_pool.save.assert_called_once_with(callback.model, step=50)


def test_random_opponent_prob_decays_warmup_to_residual():
    # 100% random warmup, then a linear decay to the residual end_prob.
    p = Trainer._random_opponent_prob
    assert p(0.0) == pytest.approx(Trainer._CURRICULUM_START_PROB)
    assert p(0.1) == pytest.approx(Trainer._CURRICULUM_START_PROB)  # in warmup
    assert p(1.0) == pytest.approx(Trainer._CURRICULUM_END_PROB)  # fully decayed
    # Monotonic non-increasing after the warmup.
    assert p(0.5) >= p(0.9) >= p(1.0)


def test_aggression_scale_off_without_curriculum():
    # No agent-count curriculum (start_n=0) => always baseline 1.0.
    a = Trainer._aggression_scale
    assert a(0.0, 0, 4) == pytest.approx(1.0)
    assert a(0.5, 0, 4) == pytest.approx(1.0)
    assert a(1.0, 0, 4) == pytest.approx(1.0)


def test_aggression_scale_boosts_through_melee_then_anneals():
    # With the curriculum active: 1.0 before the ramp, rises to the max across
    # the ramp, holds, then anneals back to 1.0 by the end of training.
    a = Trainer._aggression_scale
    assert a(Trainer._AGENT_RAMP_START - 1e-6, 2, 4) == pytest.approx(1.0)  # pre-ramp
    # Fully ramped (>= _AGENT_RAMP_END, still within the hold window) -> max.
    assert a(Trainer._AGGR_BOOST_HOLD_END - 1e-6, 2, 4) == pytest.approx(
        Trainer._AGGR_BOOST_MAX
    )
    # Mid-ramp is strictly between baseline and max.
    mid = a((Trainer._AGENT_RAMP_START + Trainer._AGENT_RAMP_END) / 2, 2, 4)
    assert 1.0 < mid < Trainer._AGGR_BOOST_MAX
    # Annealed back to baseline at the end.
    assert a(1.0, 2, 4) == pytest.approx(1.0)


def test_callback_skips_snapshot_off_interval(
    mock_env, mock_logger, mock_snapshot_pool, mock_tracker, config
):
    trainer = _make_trainer(
        mock_env, mock_logger, mock_snapshot_pool, mock_tracker, config
    )
    callback = trainer._make_callback()
    callback.model = MagicMock()
    callback.num_timesteps = 51
    callback.n_calls = config.training.snapshot_interval + 1  # not divisible
    callback.locals = {"infos": [{}]}

    assert callback._on_step() is True
    mock_snapshot_pool.save.assert_not_called()
    mock_tracker.record_match.assert_not_called()
