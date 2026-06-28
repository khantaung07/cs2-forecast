from dataclasses import dataclass

from cs2forecast.backtesting.metrics import BinaryMetrics, compute_binary_metrics
from cs2forecast.features.elo import EloConfig, EloModel, MapEloModel
from cs2forecast.storage.db import connect


@dataclass(frozen=True)
class BacktestResult:
    name: str
    metrics: BinaryMetrics


def load_completed_matches() -> list[dict]:
    with connect() as conn:
        rows = conn.execute(
            """
            SELECT
                match_id,
                date,
                team_a_id,
                team_b_id,
                winner_team_id
            FROM matches
            WHERE
                date IS NOT NULL
                AND winner_team_id IS NOT NULL
            ORDER BY date ASC, match_id ASC;
            """
        ).fetchall()

    return [dict(row) for row in rows]


def load_completed_maps() -> list[dict]:
    with connect() as conn:
        rows = conn.execute(
            """
            SELECT
                mr.map_result_id,
                m.date,
                m.team_a_id,
                m.team_b_id,
                mr.map_name,
                mr.map_index,
                mr.winner_team_id
            FROM map_results mr
            JOIN matches m ON m.match_id = mr.match_id
            WHERE
                m.date IS NOT NULL
                AND mr.winner_team_id IS NOT NULL
            ORDER BY m.date ASC, mr.map_index ASC, mr.map_result_id ASC;
            """
        ).fetchall()

    return [dict(row) for row in rows]


def backtest_overall_elo(k_factor: float = 32.0) -> BacktestResult:
    model = EloModel(EloConfig(k_factor=k_factor))

    y_true: list[int] = []
    y_prob: list[float] = []

    for match in load_completed_matches():
        team_a_id = match["team_a_id"]
        team_b_id = match["team_b_id"]
        winner_team_id = match["winner_team_id"]

        prob_team_a = model.predict_proba(team_a_id, team_b_id)
        actual_team_a = 1 if winner_team_id == team_a_id else 0

        y_prob.append(prob_team_a)
        y_true.append(actual_team_a)

        model.update(team_a_id, team_b_id, winner_team_id)

    return BacktestResult(
        name=f"Overall Elo K={k_factor:g}",
        metrics=compute_binary_metrics(y_true, y_prob),
    )


def backtest_map_elo(k_factor: float = 32.0) -> BacktestResult:
    model = MapEloModel(EloConfig(k_factor=k_factor))

    y_true: list[int] = []
    y_prob: list[float] = []

    for map_result in load_completed_maps():
        team_a_id = map_result["team_a_id"]
        team_b_id = map_result["team_b_id"]
        map_name = map_result["map_name"]
        winner_team_id = map_result["winner_team_id"]

        prob_team_a = model.predict_proba(team_a_id, team_b_id, map_name)
        actual_team_a = 1 if winner_team_id == team_a_id else 0

        y_prob.append(prob_team_a)
        y_true.append(actual_team_a)

        model.update(team_a_id, team_b_id, map_name, winner_team_id)

    return BacktestResult(
        name=f"Map Elo K={k_factor:g}",
        metrics=compute_binary_metrics(y_true, y_prob),
    )


def backtest_constant_baseline_on_matches() -> BacktestResult:
    y_true: list[int] = []
    y_prob: list[float] = []

    for match in load_completed_matches():
        actual_team_a = 1 if match["winner_team_id"] == match["team_a_id"] else 0
        y_true.append(actual_team_a)
        y_prob.append(0.5)

    return BacktestResult(
        name="Constant 50/50 Match",
        metrics=compute_binary_metrics(y_true, y_prob),
    )


def backtest_constant_baseline_on_maps() -> BacktestResult:
    y_true: list[int] = []
    y_prob: list[float] = []

    for map_result in load_completed_maps():
        actual_team_a = 1 if map_result["winner_team_id"] == map_result["team_a_id"] else 0
        y_true.append(actual_team_a)
        y_prob.append(0.5)

    return BacktestResult(
        name="Constant 50/50 Map",
        metrics=compute_binary_metrics(y_true, y_prob),
    )