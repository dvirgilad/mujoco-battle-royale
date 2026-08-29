import wandb


class WandBLogger:
    def __init__(self, project: str, run_name: str, config: dict) -> None:
        self._run = wandb.init(project=project, name=run_name, config=config)

    def log(self, metrics: dict[str, float], step: int) -> None:
        self._run.log(metrics, step=step)

    def save_artifact(self, path: str, name: str, artifact_type: str = "model") -> None:
        artifact = wandb.Artifact(name=name, type=artifact_type)
        artifact.add_file(path)
        self._run.log_artifact(artifact)
