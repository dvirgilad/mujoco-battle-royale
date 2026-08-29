"""Loggers that need no external service, for offline runs and CI."""

from __future__ import annotations


class NullLogger:
    """Discards everything. Useful in tests and when logging is unwanted."""

    def log(self, metrics: dict[str, float], step: int) -> None:  # noqa: D102
        pass

    def save_artifact(self, path: str, name: str) -> None:  # noqa: D102
        pass


class StdoutLogger:
    """Prints metrics to stdout — an offline stand-in for WandBLogger."""

    def log(self, metrics: dict[str, float], step: int) -> None:  # noqa: D102
        rendered = " ".join(f"{k}={v:.4f}" for k, v in metrics.items())
        print(f"[step {step}] {rendered}")

    def save_artifact(self, path: str, name: str) -> None:  # noqa: D102
        print(f"[artifact] {name} -> {path}")
