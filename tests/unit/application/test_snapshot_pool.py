from pathlib import Path
from unittest.mock import MagicMock

import pytest

from battle_royale.application.training.snapshot_pool import SnapshotPool


@pytest.fixture
def pool(tmp_path):
    return SnapshotPool(save_dir=str(tmp_path), max_size=3)


def test_pool_starts_empty(pool):
    assert pool.is_empty() is True


def test_sample_path_returns_none_when_empty(pool):
    assert pool.sample_path() is None


def test_save_creates_file(pool, tmp_path):
    mock_model = MagicMock()
    pool.save(mock_model, step=1000)
    files = list(tmp_path.iterdir())
    assert len(files) == 1


def test_save_calls_model_save(pool, tmp_path):
    mock_model = MagicMock()
    pool.save(mock_model, step=1000)
    mock_model.save.assert_called_once()


def test_pool_not_empty_after_save(pool):
    mock_model = MagicMock()
    pool.save(mock_model, step=1000)
    assert pool.is_empty() is False


def test_sample_path_returns_path_after_save(pool):
    mock_model = MagicMock()
    pool.save(mock_model, step=1000)
    path = pool.sample_path()
    assert path is not None
    assert isinstance(path, Path)


def test_pool_evicts_oldest_when_full(pool, tmp_path):
    mock_model = MagicMock()
    pool.save(mock_model, step=1000)
    pool.save(mock_model, step=2000)
    pool.save(mock_model, step=3000)
    pool.save(mock_model, step=4000)  # should evict step=1000
    files = list(tmp_path.iterdir())
    assert len(files) == 3
    names = {f.stem for f in files}
    assert "snapshot_000001000" not in names


def test_eviction_does_not_delete_digit_prefix_sibling(tmp_path):
    """Regression: evicting snapshot_100000 must not also delete the still-live
    snapshot_1000000 (a stem-prefix glob used to nuke it, crashing training once
    steps passed 1M)."""
    pool = SnapshotPool(save_dir=str(tmp_path), max_size=2)
    mock_model = MagicMock()
    pool.save(mock_model, step=100_000)
    pool.save(mock_model, step=1_000_000)
    pool.save(mock_model, step=2_000_000)  # evicts 100_000 only
    remaining = {f.name for f in tmp_path.iterdir()}
    assert f"snapshot_{100_000:09d}" not in remaining
    assert f"snapshot_{1_000_000:09d}" in remaining
    assert f"snapshot_{2_000_000:09d}" in remaining


def test_sample_path_is_random(pool):
    mock_model = MagicMock()
    for step in [1000, 2000, 3000]:
        pool.save(mock_model, step=step)
    # Sample multiple times — should occasionally differ (probabilistic)
    paths = {pool.sample_path() for _ in range(50)}
    assert (
        len(paths) > 1
    )  # With 3 paths and 50 samples, very unlikely to get same every time
