import importlib
from unittest.mock import MagicMock, patch


def test_train_cli_imports():
    """Verify the train CLI module is importable."""
    spec = importlib.util.find_spec("battle_royale.interfaces.cli.train")
    assert spec is not None


def test_evaluate_cli_imports():
    """Verify the evaluate CLI module is importable."""
    spec = importlib.util.find_spec("battle_royale.interfaces.cli.evaluate")
    assert spec is not None


@patch("battle_royale.interfaces.cli.train.Trainer")
@patch("battle_royale.interfaces.cli.train.load_config")
@patch("battle_royale.interfaces.cli.train.MuJoCoEnvironment")
@patch("battle_royale.interfaces.cli.train.BattleRoyaleEnv")
@patch("battle_royale.interfaces.cli.train.WandBLogger")
@patch("battle_royale.interfaces.cli.train.SnapshotPool")
@patch("battle_royale.interfaces.cli.train.MetricsTracker")
def test_train_main_wires_dependencies(
    mock_tracker_cls,
    mock_pool_cls,
    mock_logger_cls,
    mock_env_cls,
    mock_mujoco_cls,
    mock_load_config,
    mock_trainer_cls,
):
    mock_config = MagicMock()
    mock_config.training.num_agents = 4
    mock_config.training.snapshot_pool_size = 20
    mock_load_config.return_value = mock_config

    from battle_royale.interfaces.cli.train import main

    main(config_path="config/default.yaml", run_dir="/tmp/test_run")

    mock_load_config.assert_called_once_with("config/default.yaml")
    mock_mujoco_cls.assert_called_once()
    mock_env_cls.assert_called_once()
    mock_trainer_cls.assert_called_once()
    mock_trainer_cls.return_value.run.assert_called_once()


@patch("battle_royale.interfaces.cli.evaluate.Evaluator")
@patch("battle_royale.interfaces.cli.evaluate.load_config")
@patch("battle_royale.interfaces.cli.evaluate.MuJoCoEnvironment")
@patch("battle_royale.interfaces.cli.evaluate.BattleRoyaleEnv")
@patch("battle_royale.interfaces.cli.evaluate.SnapshotPool")
@patch("battle_royale.interfaces.cli.evaluate.PPO")
def test_evaluate_main_loads_checkpoint(
    mock_ppo_cls,
    mock_pool_cls,
    mock_env_cls,
    mock_mujoco_cls,
    mock_load_config,
    mock_evaluator_cls,
):
    mock_config = MagicMock()
    mock_config.training.num_agents = 4
    mock_config.arena.radius = 3.0
    mock_config.training.max_force = 10.0
    mock_load_config.return_value = mock_config
    mock_evaluator_cls.return_value.evaluate.return_value = {"win_rate": 0.5}

    from battle_royale.interfaces.cli.evaluate import main

    main(
        checkpoint_path="/tmp/checkpoint",
        num_agents=6,
        config_path="config/default.yaml",
    )

    mock_ppo_cls.load.assert_called_once_with("/tmp/checkpoint")
    mock_evaluator_cls.return_value.evaluate.assert_called_once()
