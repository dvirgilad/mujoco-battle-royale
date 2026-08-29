from __future__ import annotations

import numpy as np
import mediapy


class VideoRecorder:
    def __init__(self, output_path: str, fps: int = 30) -> None:
        self._output_path = output_path
        self._fps = fps
        self._frames: list[np.ndarray] = []

    @property
    def frame_count(self) -> int:
        return len(self._frames)

    def add_frame(self, frame: np.ndarray) -> None:
        self._frames.append(frame)

    def save(self) -> None:
        if not self._frames:
            return
        mediapy.write_video(self._output_path, self._frames, fps=self._fps)
