from typing import Protocol
import numpy as np


class IPolicy(Protocol):
    def predict(self, obs: np.ndarray) -> np.ndarray: ...
    def save(self, path: str) -> None: ...