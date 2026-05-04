from collections import deque
from battle_royale.domain.interfaces.logger import ILogger
from battle_royale.application.metrics.elo import EloRatingSystem

_ROLLING_WINDOW = 100


class MetricsTracker:
    def __init__(self, logger: ILogger) -> None:
        self._logger = logger
        self._elo = EloRatingSystem()
        self._ratings: dict[str, float] = {}
        self._win_history: deque[str] = deque(maxlen=_ROLLING_WINDOW)
        self._total_episode_length: int = 0
        self.episode_count: int = 0

    def record_episode(
        self,
        winner_id: str,
        episode_length: int,
        eliminations: int,
        step: int,
    ) -> None:
        self._win_history.append(winner_id)
        self._total_episode_length += episode_length
        self.episode_count += 1

        if winner_id not in self._ratings:
            self._ratings[winner_id] = self._elo.initial_rating
        for opp_id in list(self._ratings):
            if opp_id == winner_id:
                continue
            new_w, new_opp = self._elo.update(
                self._ratings[winner_id], self._ratings[opp_id]
            )
            self._ratings[winner_id] = new_w
            self._ratings[opp_id] = new_opp

        metrics = {
            "episode_length": float(episode_length),
            "eliminations": float(eliminations),
            "episode_count": float(self.episode_count),
        }
        for agent_id, rate in self.get_win_rates().items():
            metrics[f"win_rate/{agent_id}"] = rate

        self._logger.log(metrics, step)

    def get_win_rates(self) -> dict[str, float]:
        if not self._win_history:
            return {}
        counts: dict[str, int] = {}
        for winner in self._win_history:
            counts[winner] = counts.get(winner, 0) + 1
        total = len(self._win_history)
        return {aid: count / total for aid, count in counts.items()}

    def mean_episode_length(self) -> float:
        if self.episode_count == 0:
            return 0.0
        return self._total_episode_length / self.episode_count
