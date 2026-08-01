from unittest.mock import MagicMock
import pytest
from battle_royale.application.metrics.tracker import MetricsTracker


@pytest.fixture
def logger():
    return MagicMock()


@pytest.fixture
def tracker(logger):
    return MetricsTracker(logger=logger)


def test_initial_state(tracker):
    assert tracker.episode_count == 0


def test_record_episode_increments_count(tracker):
    tracker.record_episode(
        winner_id="agent_0", episode_length=100, eliminations=3, step=0
    )
    assert tracker.episode_count == 1


def test_win_rate_rolling_100_episodes(tracker):
    for i in range(50):
        tracker.record_episode(
            winner_id="agent_0", episode_length=50, eliminations=1, step=i
        )
    for i in range(50):
        tracker.record_episode(
            winner_id="agent_1", episode_length=50, eliminations=1, step=50 + i
        )
    # 50% win rate for agent_0 over last 100
    win_rates = tracker.get_win_rates()
    assert pytest.approx(win_rates.get("agent_0", 0.0), abs=0.01) == 0.5


def test_logger_called_on_record(tracker, logger):
    tracker.record_episode(
        winner_id="agent_0", episode_length=100, eliminations=2, step=42
    )
    logger.log.assert_called_once()
    call_args = logger.log.call_args
    metrics = call_args[0][0]
    assert "episode_length" in metrics
    assert "eliminations" in metrics


def test_mean_episode_length_tracked(tracker):
    tracker.record_episode(
        winner_id="agent_0", episode_length=100, eliminations=1, step=0
    )
    tracker.record_episode(
        winner_id="agent_1", episode_length=200, eliminations=1, step=1
    )
    assert tracker.mean_episode_length() == pytest.approx(150.0, abs=0.1)


def test_record_match_updates_win_rate(tracker):
    for i in range(4):
        tracker.record_match(won=True, episode_length=50, eliminations=1, step=i)
    tracker.record_match(won=False, episode_length=50, eliminations=0, step=4)
    assert tracker.selfplay_win_rate() == pytest.approx(0.8, abs=0.01)


def test_record_match_winner_gains_elo(tracker):
    start = tracker.learner_rating
    tracker.record_match(won=True, episode_length=50, eliminations=2, step=0)
    assert tracker.learner_rating > start
    assert tracker.pool_rating < start


def test_record_match_loser_loses_elo(tracker):
    start = tracker.learner_rating
    tracker.record_match(won=False, episode_length=50, eliminations=0, step=0)
    assert tracker.learner_rating < start


def test_record_match_logs_selfplay_metrics(tracker, logger):
    tracker.record_match(won=True, episode_length=77, eliminations=2, step=9)
    metrics = logger.log.call_args[0][0]
    assert "selfplay/win_rate" in metrics
    assert "selfplay/elo" in metrics
