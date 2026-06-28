from collections import defaultdict
from dataclasses import dataclass


@dataclass(frozen=True)
class MapRecentFormConfig:
    decay: float = 0.95


class MapRecentFormTracker:
    """
    Tracks opponent-adjusted recent form per team per map.

    Example:
        Vitality on Nuke has a different form score from Vitality on Mirage.
    """

    def __init__(self, config: MapRecentFormConfig = MapRecentFormConfig()):
        self.config = config
        self.scores: defaultdict[tuple[str, str], float] = defaultdict(float)

    def get_score(self, team_id: str, map_name: str) -> float:
        return self.scores[(map_name, team_id)]

    def update(
        self,
        team_a_id: str,
        team_b_id: str,
        map_name: str,
        actual_a: float,
        expected_a: float,
    ) -> None:
        residual_a = actual_a - expected_a
        residual_b = -residual_a

        key_a = (map_name, team_a_id)
        key_b = (map_name, team_b_id)

        self.scores[key_a] = (
            self.config.decay * self.scores[key_a]
            + (1.0 - self.config.decay) * residual_a
        )

        self.scores[key_b] = (
            self.config.decay * self.scores[key_b]
            + (1.0 - self.config.decay) * residual_b
        )