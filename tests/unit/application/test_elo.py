import pytest

from battle_royale.application.metrics.elo import EloRatingSystem


@pytest.fixture
def elo():
    return EloRatingSystem(initial_rating=1500.0)


def test_initial_rating(elo):
    assert elo.initial_rating == 1500.0


def test_expected_score_equal_ratings(elo):
    score = elo.expected_score(1500.0, 1500.0)
    assert pytest.approx(score, abs=1e-6) == 0.5


def test_expected_score_higher_vs_lower(elo):
    score = elo.expected_score(1700.0, 1500.0)
    assert score > 0.5


def test_expected_score_lower_vs_higher(elo):
    score = elo.expected_score(1500.0, 1700.0)
    assert score < 0.5


def test_expected_scores_sum_to_one(elo):
    a = elo.expected_score(1600.0, 1400.0)
    b = elo.expected_score(1400.0, 1600.0)
    assert pytest.approx(a + b, abs=1e-6) == 1.0


def test_update_winner_gains_rating(elo):
    new_winner, new_loser = elo.update(1500.0, 1500.0)
    assert new_winner > 1500.0


def test_update_loser_loses_rating(elo):
    new_winner, new_loser = elo.update(1500.0, 1500.0)
    assert new_loser < 1500.0


def test_update_preserves_total_rating(elo):
    w, loser = elo.update(1600.0, 1400.0)
    assert pytest.approx(w + loser, abs=1e-4) == 3000.0


def test_update_k32_equal_ratings(elo):
    # Equal ratings: winner gains K * (1 - 0.5) = 16
    w, loser = elo.update(1500.0, 1500.0)
    assert pytest.approx(w, abs=1e-3) == 1516.0
    assert pytest.approx(loser, abs=1e-3) == 1484.0


def test_update_strong_favorite_small_gain(elo):
    # Strong favorite wins — small gain
    w, loser = elo.update(1800.0, 1200.0)
    gain = w - 1800.0
    assert gain < 5.0  # Expected score was high, so small gain
