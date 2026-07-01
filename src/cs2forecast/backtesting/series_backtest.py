from collections import defaultdict
from dataclasses import dataclass

from cs2forecast.backtesting.elo_backtest import BacktestResult
from cs2forecast.backtesting.metrics import compute_binary_metrics
from cs2forecast.features.elo import EloConfig, EloModel, MapEloModel
from cs2forecast.storage.db import connect


@dataclass(frozen=True)
class SeriesMap:
    map_index: int
    map_name: str
    winner_team_id: str


@dataclass(frozen=True)
class CompletedSeries:
    match_id: str
    date: str
    best_of: int
    team_a_id: str
    team_b_id: str
    winner_team_id: str
    maps: list[SeriesMap]


@dataclass(frozen=True)
class SeriesBacktestConfig:
    map_k_factor: float = 32.0
    min_team_matches: int = 5
    require_full_map_list: bool = False


def load_completed_series() -> list[CompletedSeries]:
    with connect() as conn:
        rows = conn.execute(
            """
            SELECT
                m.match_id,
                m.date,
                m.best_of,
                m.team_a_id,
                m.team_b_id,
                m.winner_team_id,
                mr.map_index,
                mr.map_name,
                mr.winner_team_id AS map_winner_team_id
            FROM matches m
            JOIN map_results mr ON mr.match_id = m.match_id
            WHERE
                m.date IS NOT NULL
                AND m.winner_team_id IS NOT NULL
                AND mr.winner_team_id IS NOT NULL
            ORDER BY m.date, m.match_id, mr.map_index;
            """
        ).fetchall()

    grouped: dict[str, dict] = {}

    for row in rows:
        match_id = row["match_id"]

        if match_id not in grouped:
            grouped[match_id] = {
                "match_id": match_id,
                "date": row["date"],
                "best_of": row["best_of"],
                "team_a_id": row["team_a_id"],
                "team_b_id": row["team_b_id"],
                "winner_team_id": row["winner_team_id"],
                "maps": [],
            }

        grouped[match_id]["maps"].append(
            SeriesMap(
                map_index=row["map_index"],
                map_name=row["map_name"],
                winner_team_id=row["map_winner_team_id"],
            )
        )

    series = [
        CompletedSeries(
            match_id=value["match_id"],
            date=value["date"],
            best_of=value["best_of"],
            team_a_id=value["team_a_id"],
            team_b_id=value["team_b_id"],
            winner_team_id=value["winner_team_id"],
            maps=value["maps"],
        )
        for value in grouped.values()
    ]

    return series


def normalise_best_of(best_of: int | None, map_count: int) -> int:
    if best_of in {1, 3, 5}:
        return best_of

    if map_count <= 1:
        return 1

    if map_count <= 3:
        return 3

    return 5


def has_full_map_list(series: CompletedSeries) -> bool:
    best_of = normalise_best_of(series.best_of, len(series.maps))
    return len(series.maps) >= best_of


def should_score_series_prediction(
    team_match_counts: dict[str, int],
    team_a_id: str,
    team_b_id: str,
    min_team_matches: int,
) -> bool:
    return (
        team_match_counts[team_a_id] >= min_team_matches
        and team_match_counts[team_b_id] >= min_team_matches
    )


def series_win_probability(
    map_probs: list[float],
    best_of: int,
    fallback_prob: float,
) -> float:
    """
    Converts individual map win probabilities into series win probability.

    For a Bo3:
        p1, p2, p3 -> probability Team A wins at least 2 maps before Team B.

    If the series ended 2-0 and map 3 was not played, fallback_prob is used for
    the missing possible decider map.
    """
    wins_needed = best_of // 2 + 1

    full_map_probs = list(map_probs[:best_of])

    while len(full_map_probs) < best_of:
        full_map_probs.append(fallback_prob)

    states: dict[tuple[int, int], float] = {(0, 0): 1.0}
    series_prob_a = 0.0

    for prob_a_wins_map in full_map_probs:
        next_states: defaultdict[tuple[int, int], float] = defaultdict(float)

        for (a_wins, b_wins), state_prob in states.items():
            new_a_wins = a_wins + 1
            prob_path_a = state_prob * prob_a_wins_map

            if new_a_wins >= wins_needed:
                series_prob_a += prob_path_a
            else:
                next_states[(new_a_wins, b_wins)] += prob_path_a

            new_b_wins = b_wins + 1
            prob_path_b = state_prob * (1.0 - prob_a_wins_map)

            if new_b_wins < wins_needed:
                next_states[(a_wins, new_b_wins)] += prob_path_b

        states = dict(next_states)

    return series_prob_a


def backtest_series_constant_50_50(
    config: SeriesBacktestConfig = SeriesBacktestConfig(),
) -> BacktestResult:
    team_match_counts: defaultdict[str, int] = defaultdict(int)

    y_true: list[int] = []
    y_prob: list[float] = []

    for series in load_completed_series():
        if config.require_full_map_list and not has_full_map_list(series):
            team_match_counts[series.team_a_id] += 1
            team_match_counts[series.team_b_id] += 1
            continue

        if should_score_series_prediction(
            team_match_counts,
            series.team_a_id,
            series.team_b_id,
            config.min_team_matches,
        ):
            actual_a = 1 if series.winner_team_id == series.team_a_id else 0
            y_true.append(actual_a)
            y_prob.append(0.5)

        team_match_counts[series.team_a_id] += 1
        team_match_counts[series.team_b_id] += 1

    return BacktestResult(
        name=f"Constant 50/50 Series min={config.min_team_matches}",
        metrics=compute_binary_metrics(y_true, y_prob),
    )


def backtest_series_from_overall_map_elo(
    config: SeriesBacktestConfig = SeriesBacktestConfig(),
) -> BacktestResult:
    """
    Uses one overall map-level Elo probability for each possible map in the series.

    This ignores map identity, but correctly converts map win probability into
    Bo1/Bo3/Bo5 series probability.
    """
    overall_map_elo = EloModel(EloConfig(k_factor=config.map_k_factor))
    team_match_counts: defaultdict[str, int] = defaultdict(int)

    y_true: list[int] = []
    y_prob: list[float] = []

    for series in load_completed_series():
        best_of = normalise_best_of(series.best_of, len(series.maps))

        prob_a_map = overall_map_elo.predict_proba(
            series.team_a_id,
            series.team_b_id,
        )

        map_probs = [prob_a_map for _ in series.maps]

        prob_a_series = series_win_probability(
            map_probs=map_probs,
            best_of=best_of,
            fallback_prob=prob_a_map,
        )

        if not (config.require_full_map_list and not has_full_map_list(series)):
            if should_score_series_prediction(
                team_match_counts,
                series.team_a_id,
                series.team_b_id,
                config.min_team_matches,
            ):
                actual_a = 1 if series.winner_team_id == series.team_a_id else 0
                y_true.append(actual_a)
                y_prob.append(prob_a_series)

        for map_result in series.maps:
            overall_map_elo.update(
                series.team_a_id,
                series.team_b_id,
                map_result.winner_team_id,
            )

        team_match_counts[series.team_a_id] += 1
        team_match_counts[series.team_b_id] += 1

    return BacktestResult(
        name=f"Series from Overall Map Elo min={config.min_team_matches}",
        metrics=compute_binary_metrics(y_true, y_prob),
    )


def backtest_series_from_plain_map_elo(
    config: SeriesBacktestConfig = SeriesBacktestConfig(),
) -> BacktestResult:
    """
    Uses map-specific Elo for played maps.

    For unplayed possible decider maps, falls back to overall map Elo because
    the current parser does not know the veto/decider map.
    """
    overall_map_elo = EloModel(EloConfig(k_factor=config.map_k_factor))
    map_elo = MapEloModel(EloConfig(k_factor=config.map_k_factor))
    team_match_counts: defaultdict[str, int] = defaultdict(int)

    y_true: list[int] = []
    y_prob: list[float] = []

    for series in load_completed_series():
        best_of = normalise_best_of(series.best_of, len(series.maps))

        fallback_prob = overall_map_elo.predict_proba(
            series.team_a_id,
            series.team_b_id,
        )

        map_probs = [
            map_elo.predict_proba(
                series.team_a_id,
                series.team_b_id,
                map_result.map_name,
            )
            for map_result in series.maps
        ]

        prob_a_series = series_win_probability(
            map_probs=map_probs,
            best_of=best_of,
            fallback_prob=fallback_prob,
        )

        if not (config.require_full_map_list and not has_full_map_list(series)):
            if should_score_series_prediction(
                team_match_counts,
                series.team_a_id,
                series.team_b_id,
                config.min_team_matches,
            ):
                actual_a = 1 if series.winner_team_id == series.team_a_id else 0
                y_true.append(actual_a)
                y_prob.append(prob_a_series)

        for map_result in series.maps:
            overall_map_elo.update(
                series.team_a_id,
                series.team_b_id,
                map_result.winner_team_id,
            )

            map_elo.update(
                series.team_a_id,
                series.team_b_id,
                map_result.map_name,
                map_result.winner_team_id,
            )

        team_match_counts[series.team_a_id] += 1
        team_match_counts[series.team_b_id] += 1

    return BacktestResult(
        name=f"Series from Plain Map Elo min={config.min_team_matches}",
        metrics=compute_binary_metrics(y_true, y_prob),
    )