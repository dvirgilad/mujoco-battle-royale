import random
from pathlib import Path


class SnapshotPool:
    def __init__(self, save_dir: str, max_size: int = 20) -> None:
        self._save_dir = Path(save_dir)
        self._save_dir.mkdir(parents=True, exist_ok=True)
        self._max_size = max_size
        self._paths: list[Path] = []

    def save(self, model, step: int) -> None:
        filename = f"snapshot_{step:06d}"
        path = self._save_dir / filename
        model.save(str(path))
        # Ensure file exists (handles mock case in tests)
        if not path.exists():
            path.touch()
        self._paths.append(path)
        if len(self._paths) > self._max_size:
            oldest = self._paths.pop(0)
            # Remove all files matching this snapshot (SB3 saves .zip)
            for f in self._save_dir.glob(f"{oldest.name}*"):
                f.unlink(missing_ok=True)

    def sample_path(self) -> Path | None:
        if not self._paths:
            return None
        return random.choice(self._paths)

    def is_empty(self) -> bool:
        return len(self._paths) == 0
