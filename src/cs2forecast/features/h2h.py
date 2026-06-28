from collections import defaultdict
from dataclasses import dataclass


@dataclass
class H2HRecord:
    first_team_wins: int = 0
    second_team_wins: int = 0


@dataclass(frozen=True)
class H2HConfig:
    shrinkage: float = 5.0


class H2HTracker:
    """
    Tracks head-to-head results with shrinkage.

    One previous win should not heavily affect the model.
    Many repeated wins should matter more.
    """

    def __init__(self, config: H2HConfig = H2HConfig()):
        self.config = config
        self.records: defaultdict[tuple[str, str], H2HRecord] = defaultdict(H2HRecord)

    def _key(self, team_a_id: str, team_b_id: str) -> tuple[str, str]:
        return tuple(sorted((team_a_id, team_b_id)))

    def get_score(self, team_a_id: str, team_b_id: str) -> float:
        """
        Returns a score from team_a's perspective.

        Positive = team_a has historically done well against team_b.
        Negative = team_a has historically done badly against team_b.
        """
        key = self._key(team_a_id, team_b_id)
        record = self.records[key]

        games = record.first_team_wins + record.second_team_wins
        if games == 0:
            return 0.0

        first_team, second_team = key

        if team_a_id == first_team:
            net_wins = record.first_team_wins - record.second_team_wins
        else:
            net_wins = record.second_team_wins - record.first_team_wins

        return net_wins / (games + self.config.shrinkage)

    def update(self, team_a_id: str, team_b_id: str, winner_team_id: str) -> None:
        key = self._key(team_a_id, team_b_id)
        record = self.records[key]

        first_team, second_team = key

        if winner_team_id == first_team:
            record.first_team_wins += 1
        elif winner_team_id == second_team:
            record.second_team_wins += 1
