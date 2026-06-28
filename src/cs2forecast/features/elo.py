from collections import defaultdict
from dataclasses import dataclass

from collections.abc import Callable

@dataclass(frozen=True)
class EloConfig:
    initial_rating: float = 1500.0
    k_factor: float = 32.0


class EloModel:
    def __init__(self, config: EloConfig = EloConfig()):
        self.config = config
        self.ratings: defaultdict[str, float] = defaultdict(
            lambda: self.config.initial_rating
        )

    def get_rating(self, team_id: str) -> float:
        return self.ratings[team_id]

    def predict_proba(self, team_a_id: str, team_b_id: str) -> float:
        rating_a = self.get_rating(team_a_id)
        rating_b = self.get_rating(team_b_id)

        return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400.0))

    def update(self, team_a_id: str, team_b_id: str, winner_team_id: str) -> None:
        p_a = self.predict_proba(team_a_id, team_b_id)
        actual_a = 1.0 if winner_team_id == team_a_id else 0.0

        delta = self.config.k_factor * (actual_a - p_a)

        self.ratings[team_a_id] += delta
        self.ratings[team_b_id] -= delta


class MapEloModel:
    """
    Separate Elo pool for each map.

    Example:
        Faze on Nuke has a different rating from Faze on Mirage.
    """

    def __init__(self, config: EloConfig = EloConfig()):
        self.config = config
        self.ratings: defaultdict[tuple[str, str], float] = defaultdict(
            lambda: self.config.initial_rating
        )

    def get_rating(self, team_id: str, map_name: str) -> float:
        return self.ratings[(map_name, team_id)]

    def predict_proba(self, team_a_id: str, team_b_id: str, map_name: str) -> float:
        rating_a = self.get_rating(team_a_id, map_name)
        rating_b = self.get_rating(team_b_id, map_name)

        return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400.0))

    def update(
        self,
        team_a_id: str,
        team_b_id: str,
        map_name: str,
        winner_team_id: str,
    ) -> None:
        p_a = self.predict_proba(team_a_id, team_b_id, map_name)
        actual_a = 1.0 if winner_team_id == team_a_id else 0.0

        delta = self.config.k_factor * (actual_a - p_a)

        self.ratings[(map_name, team_a_id)] += delta
        self.ratings[(map_name, team_b_id)] -= delta


class DynamicKFactorEloModel:
    """
    Elo model where K depends on how much history each team has.

    New or low-history teams update faster.
    Established teams update more slowly.
    """

    def __init__(
        self,
        config: EloConfig = EloConfig(),
        k_factor_fn: Callable[[int], float] | None = None,
    ):
        self.config = config
        self.ratings: defaultdict[str, float] = defaultdict(
            lambda: self.config.initial_rating
        )
        self.match_counts: defaultdict[str, int] = defaultdict(int)
        self.k_factor_fn = k_factor_fn or self.default_k_factor

    def default_k_factor(self, prior_matches: int) -> float:
        if prior_matches < 5:
            return 48.0
        if prior_matches < 20:
            return 32.0
        return 20.0

    def get_rating(self, team_id: str) -> float:
        return self.ratings[team_id]

    def get_match_count(self, team_id: str) -> int:
        return self.match_counts[team_id]

    def predict_proba(self, team_a_id: str, team_b_id: str) -> float:
        rating_a = self.get_rating(team_a_id)
        rating_b = self.get_rating(team_b_id)

        return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400.0))

    def update(self, team_a_id: str, team_b_id: str, winner_team_id: str) -> None:
        p_a = self.predict_proba(team_a_id, team_b_id)
        actual_a = 1.0 if winner_team_id == team_a_id else 0.0

        k_a = self.k_factor_fn(self.match_counts[team_a_id])
        k_b = self.k_factor_fn(self.match_counts[team_b_id])

        self.ratings[team_a_id] += k_a * (actual_a - p_a)
        self.ratings[team_b_id] += k_b * ((1.0 - actual_a) - (1.0 - p_a))

        self.match_counts[team_a_id] += 1
        self.match_counts[team_b_id] += 1