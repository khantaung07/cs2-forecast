from collections import defaultdict
from dataclasses import dataclass

from cs2forecast.backtesting.elo_backtest import BacktestResult, load_completed_maps
from cs2forecast.backtesting.metrics import compute_binary_metrics
from cs2forecast.features.elo import EloConfig, EloModel, MapEloModel
from cs2forecast.features.map_recent_form import (
    MapRecentFormConfig,
    MapRecentFormTracker,
)


@dataclass(frozen=True)
class EnhancedMapEloConfig:
    overall_k_factor: float = 32.0
    map_k_factor: float = 32.0

    min_team_maps: int = 5

    map_elo_weight: float = 0.35

    map_form_decay: float = 0.95
    map_form_weight: float = 100.0


def probability_from_ratings(rating_a: float, rating_b: float) -> float:
    return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400.0))


def should_score_map_prediction(
    team_map_counts: dict[str, int],
    team_a_id: str,
    team_b_id: str,
    min_team_maps: int,
) -> bool:
    return (
        team_map_counts[team_a_id] >= min_team_maps
        and team_map_counts[team_b_id] >= min_team_maps
    )


def backtest_constant_50_50_maps_filtered(
    min_team_maps: int = 5,
) -> BacktestResult:
    team_map_counts: defaultdict[str, int] = defaultdict(int)

    y_true: list[int] = []
    y_prob: list[float] = []

    for map_result in load_completed_maps():
        team_a_id = map_result["team_a_id"]
        team_b_id = map_result["team_b_id"]
        winner_team_id = map_result["winner_team_id"]

        if should_score_map_prediction(
            team_map_counts,
            team_a_id,
            team_b_id,
            min_team_maps,
        ):
            actual_a = 1 if winner_team_id == team_a_id else 0
            y_true.append(actual_a)
            y_prob.append(0.5)

        team_map_counts[team_a_id] += 1
        team_map_counts[team_b_id] += 1

    return BacktestResult(
        name=f"Constant 50/50 Map min={min_team_maps}",
        metrics=compute_binary_metrics(y_true, y_prob),
    )


def backtest_overall_map_elo_filtered(
    k_factor: float = 32.0,
    min_team_maps: int = 5,
) -> BacktestResult:
    """
    Overall team Elo updated on every map result.

    This asks:
        ignoring map identity, how good is each team at winning maps?
    """
    overall_elo = EloModel(EloConfig(k_factor=k_factor))
    team_map_counts: defaultdict[str, int] = defaultdict(int)

    y_true: list[int] = []
    y_prob: list[float] = []

    for map_result in load_completed_maps():
        team_a_id = map_result["team_a_id"]
        team_b_id = map_result["team_b_id"]
        winner_team_id = map_result["winner_team_id"]

        prob_a = overall_elo.predict_proba(team_a_id, team_b_id)

        if should_score_map_prediction(
            team_map_counts,
            team_a_id,
            team_b_id,
            min_team_maps,
        ):
            actual_a = 1 if winner_team_id == team_a_id else 0
            y_true.append(actual_a)
            y_prob.append(prob_a)

        overall_elo.update(team_a_id, team_b_id, winner_team_id)

        team_map_counts[team_a_id] += 1
        team_map_counts[team_b_id] += 1

    return BacktestResult(
        name=f"Overall Map Elo min={min_team_maps}",
        metrics=compute_binary_metrics(y_true, y_prob),
    )


def backtest_plain_map_elo_filtered(
    k_factor: float = 32.0,
    min_team_maps: int = 5,
) -> BacktestResult:
    """
    Map-specific Elo only.

    This is sparse because each team-map pair gets less data.
    """
    map_elo = MapEloModel(EloConfig(k_factor=k_factor))
    team_map_counts: defaultdict[str, int] = defaultdict(int)

    y_true: list[int] = []
    y_prob: list[float] = []

    for map_result in load_completed_maps():
        team_a_id = map_result["team_a_id"]
        team_b_id = map_result["team_b_id"]
        map_name = map_result["map_name"]
        winner_team_id = map_result["winner_team_id"]

        prob_a = map_elo.predict_proba(team_a_id, team_b_id, map_name)

        if should_score_map_prediction(
            team_map_counts,
            team_a_id,
            team_b_id,
            min_team_maps,
        ):
            actual_a = 1 if winner_team_id == team_a_id else 0
            y_true.append(actual_a)
            y_prob.append(prob_a)

        map_elo.update(team_a_id, team_b_id, map_name, winner_team_id)

        team_map_counts[team_a_id] += 1
        team_map_counts[team_b_id] += 1

    return BacktestResult(
        name=f"Plain Map Elo min={min_team_maps}",
        metrics=compute_binary_metrics(y_true, y_prob),
    )


def backtest_overall_plus_map_elo(
    config: EnhancedMapEloConfig = EnhancedMapEloConfig(),
) -> BacktestResult:
    """
    Combines dense overall map Elo with sparse map-specific Elo.

    adjusted_rating =
        overall_rating
      + map_elo_weight * (map_specific_rating - 1500)
    """
    overall_elo = EloModel(EloConfig(k_factor=config.overall_k_factor))
    map_elo = MapEloModel(EloConfig(k_factor=config.map_k_factor))

    team_map_counts: defaultdict[str, int] = defaultdict(int)

    y_true: list[int] = []
    y_prob: list[float] = []

    for map_result in load_completed_maps():
        team_a_id = map_result["team_a_id"]
        team_b_id = map_result["team_b_id"]
        map_name = map_result["map_name"]
        winner_team_id = map_result["winner_team_id"]

        overall_rating_a = overall_elo.get_rating(team_a_id)
        overall_rating_b = overall_elo.get_rating(team_b_id)

        map_rating_a = map_elo.get_rating(team_a_id, map_name)
        map_rating_b = map_elo.get_rating(team_b_id, map_name)

        adjusted_rating_a = (
            overall_rating_a
            + config.map_elo_weight * (map_rating_a - 1500.0)
        )

        adjusted_rating_b = (
            overall_rating_b
            + config.map_elo_weight * (map_rating_b - 1500.0)
        )

        prob_a = probability_from_ratings(adjusted_rating_a, adjusted_rating_b)

        if should_score_map_prediction(
            team_map_counts,
            team_a_id,
            team_b_id,
            config.min_team_maps,
        ):
            actual_a = 1 if winner_team_id == team_a_id else 0
            y_true.append(actual_a)
            y_prob.append(prob_a)

        overall_elo.update(team_a_id, team_b_id, winner_team_id)
        map_elo.update(team_a_id, team_b_id, map_name, winner_team_id)

        team_map_counts[team_a_id] += 1
        team_map_counts[team_b_id] += 1

    return BacktestResult(
        name=(
            f"Overall+Map Elo min={config.min_team_maps} "
            f"map_w={config.map_elo_weight:g}"
        ),
        metrics=compute_binary_metrics(y_true, y_prob),
    )


def backtest_enhanced_map_elo(
    config: EnhancedMapEloConfig = EnhancedMapEloConfig(),
) -> BacktestResult:
    """
    Enhanced map model:

        overall map-level Elo
      + weighted map-specific Elo adjustment
      + opponent-adjusted recent map form
    """
    overall_elo = EloModel(EloConfig(k_factor=config.overall_k_factor))
    map_elo = MapEloModel(EloConfig(k_factor=config.map_k_factor))
    map_form = MapRecentFormTracker(
        MapRecentFormConfig(decay=config.map_form_decay)
    )

    team_map_counts: defaultdict[str, int] = defaultdict(int)

    y_true: list[int] = []
    y_prob: list[float] = []

    for map_result in load_completed_maps():
        team_a_id = map_result["team_a_id"]
        team_b_id = map_result["team_b_id"]
        map_name = map_result["map_name"]
        winner_team_id = map_result["winner_team_id"]

        overall_rating_a = overall_elo.get_rating(team_a_id)
        overall_rating_b = overall_elo.get_rating(team_b_id)

        map_rating_a = map_elo.get_rating(team_a_id, map_name)
        map_rating_b = map_elo.get_rating(team_b_id, map_name)

        base_prob_a = probability_from_ratings(overall_rating_a, overall_rating_b)

        map_adjustment_a = config.map_elo_weight * (map_rating_a - 1500.0)
        map_adjustment_b = config.map_elo_weight * (map_rating_b - 1500.0)

        form_adjustment_a = (
            config.map_form_weight * map_form.get_score(team_a_id, map_name)
        )
        form_adjustment_b = (
            config.map_form_weight * map_form.get_score(team_b_id, map_name)
        )

        adjusted_rating_a = (
            overall_rating_a
            + map_adjustment_a
            + form_adjustment_a
        )

        adjusted_rating_b = (
            overall_rating_b
            + map_adjustment_b
            + form_adjustment_b
        )

        prob_a = probability_from_ratings(adjusted_rating_a, adjusted_rating_b)
        actual_a = 1 if winner_team_id == team_a_id else 0

        if should_score_map_prediction(
            team_map_counts,
            team_a_id,
            team_b_id,
            config.min_team_maps,
        ):
            y_true.append(actual_a)
            y_prob.append(prob_a)

        # Update only after prediction.
        overall_elo.update(team_a_id, team_b_id, winner_team_id)
        map_elo.update(team_a_id, team_b_id, map_name, winner_team_id)

        map_form.update(
            team_a_id=team_a_id,
            team_b_id=team_b_id,
            map_name=map_name,
            actual_a=float(actual_a),
            expected_a=base_prob_a,
        )

        team_map_counts[team_a_id] += 1
        team_map_counts[team_b_id] += 1

    return BacktestResult(
        name=(
            f"Enhanced Map Elo min={config.min_team_maps} "
            f"map_w={config.map_elo_weight:g} "
            f"form={config.map_form_weight:g}"
        ),
        metrics=compute_binary_metrics(y_true, y_prob),
    )