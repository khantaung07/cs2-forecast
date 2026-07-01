from collections import defaultdict
from dataclasses import dataclass


@dataclass(frozen=True)
class RecentFormConfig:
    decay: float = 0.85


class RecentFormTracker:
    """
    Tracks opponent-adjusted recent form.

    Positive form means the team has recently performed better than Elo expected.
    Negative form means the team has recently underperformed Elo expectation.
    """

    def __init__(self, config: RecentFormConfig = RecentFormConfig()):
        self.config = config
        self.scores: defaultdict[str, float] = defaultdict(float)

    def get_score(self, team_id: str) -> float:
        return self.scores[team_id]

    def update(
        self,
        team_a_id: str,
        team_b_id: str,
        actual_a: float,
        expected_a: float,
    ) -> None:
        residual_a = actual_a - expected_a
        residual_b = -residual_a

        self.scores[team_a_id] = (
            self.config.decay * self.scores[team_a_id]
            + (1.0 - self.config.decay) * residual_a
        )

        self.scores[team_b_id] = (
            self.config.decay * self.scores[team_b_id]
            + (1.0 - self.config.decay) * residual_b
        )
