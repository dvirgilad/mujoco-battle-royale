_K = 32.0


class EloRatingSystem:
    def __init__(self, initial_rating: float = 1500.0) -> None:
        self.initial_rating = initial_rating

    def expected_score(self, rating_a: float, rating_b: float) -> float:
        return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))

    def update(
        self, winner_rating: float, loser_rating: float
    ) -> tuple[float, float]:
        exp_w = self.expected_score(winner_rating, loser_rating)
        exp_l = self.expected_score(loser_rating, winner_rating)
        new_winner = winner_rating + _K * (1.0 - exp_w)
        new_loser = loser_rating + _K * (0.0 - exp_l)
        return new_winner, new_loser
