from dataclasses import dataclass


@dataclass(frozen=True)
class Arena:
    radius: float
