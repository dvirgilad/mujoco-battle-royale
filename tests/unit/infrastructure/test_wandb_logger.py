from unittest.mock import MagicMock, patch
from battle_royale.infrastructure.logging.wandb_logger import WandBLogger


@patch("battle_royale.infrastructure.logging.wandb_logger.wandb")
def test_log_calls_wandb_log(mock_wandb):
    logger = WandBLogger(project="test", run_name="run1", config={})
    logger.log({"loss": 0.5, "reward": 1.2}, step=100)
    mock_wandb.init.return_value.log.assert_called_once_with(
        {"loss": 0.5, "reward": 1.2}, step=100
    )


@patch("battle_royale.infrastructure.logging.wandb_logger.wandb")
def test_save_artifact_calls_wandb(mock_wandb):
    mock_artifact = MagicMock()
    mock_wandb.Artifact.return_value = mock_artifact
    logger = WandBLogger(project="test", run_name="run1", config={})
    logger.save_artifact("/some/path/model.zip", "model_checkpoint")
    mock_wandb.Artifact.assert_called_once_with(name="model_checkpoint", type="model")
    mock_artifact.add_file.assert_called_once_with("/some/path/model.zip")
    mock_wandb.init.return_value.log_artifact.assert_called_once_with(mock_artifact)


@patch("battle_royale.infrastructure.logging.wandb_logger.wandb")
def test_init_calls_wandb_init(mock_wandb):
    _ = WandBLogger(project="myproject", run_name="exp1", config={"lr": 0.001})
    mock_wandb.init.assert_called_once_with(
        project="myproject",
        name="exp1",
        config={"lr": 0.001},
    )
