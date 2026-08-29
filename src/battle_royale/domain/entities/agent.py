from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Agent:
    id: str
    position: np.ndarray
    velocity: np.ndarray
    alive: bool
