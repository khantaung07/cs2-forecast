from collections import defaultdict
from dataclasses import dataclass

from cs2forecast.backtesting.elo_backtest import BacktestResult
from cs2forecast.backtesting.enhanced_backtest import probability_from_ratings
from cs2forecast.backtesting.metrics import compute_binary_metrics
from cs2forecast.backtesting.series_backtest import (
    load_completed_series,
    normalise_best_of,
    series_win_probability,
    should_score_series_prediction,
)
from cs2forecast.features.elo import (
    DynamicKFactorEloModel,
    EloConfig,
    EloModel,
)
from cs2forecast.features.h2h import H2HConfig, H2HTracker
from cs2forecast.features.recent_form import (
    RecentFormConfig,
    RecentFormTracker,
)


@dataclass(frozen=True)
class BlendedSeriesBacktestConfig:
    match_k_factor: float = 32.0
    map_k_factor: float = 32.0

    min_team_matches: int = 5

    form_decay: float = 0.95
    form_weight: float = 100.0

    h2h_shrinkage: float = 3.0
    h2h_weight: float = 50.0


def enhanced_dynamic_match_probability(
    match_elo: DynamicKFactorEloModel,
    recent_form: RecentFormTracker,
    h2h: H2HTracker,
    team_a_id: str,
    team_b_id: str,
    config: BlendedSeriesBacktestConfig,
) -> float:
    rating_a = match_elo.get_rating(team_a_id)
    rating_b = match_elo.get_rating(team_b_id)

    form_a = recent_form.get_score(team_a_id)
    form_b = recent_form.get_score(team_b_id)

    h2h_score_a = h2h.get_score(team_a_id, team_b_id)

    adjusted_rating_a = (
        rating_a
        + config.form_weight * form_a
        + config.h2h_weight * h2h_score_a
    )

    adjusted_rating_b = (
        rating_b
        + config.form_weight * form_b
        - config.h2h_weight * h2h_score_a
    )

    return probability_from_ratings(adjusted_rating_a, adjusted_rating_b)


def update_match_models(
    match_elo: DynamicKFactorEloModel,
    recent_form: RecentFormTracker,
    h2h: H2HTracker,
    team_a_id: str,
    team_b_id: str,
    winner_team_id: str,
) -> None:
    base_prob_a = match_elo.predict_proba(team_a_id, team_b_id)
    actual_a = 1.0 if winner_team_id == team_a_id else 0.0

    match_elo.update(team_a_id, team_b_id, winner_team_id)

    recent_form.update(
        team_a_id=team_a_id,
        team_b_id=team_b_id,
        actual_a=actual_a,
        expected_a=base_prob_a,
    )

    h2h.update(team_a_id, team_b_id, winner_team_id)


def series_probability_from_overall_map_elo(
    overall_map_elo: EloModel,
    team_a_id: str,
    team_b_id: str,
    best_of: int,
    played_map_count: int,
) -> float:
    prob_a_map = overall_map_elo.predict_proba(team_a_id, team_b_id)

    map_probs = [prob_a_map for _ in range(played_map_count)]

    return series_win_probability(
        map_probs=map_probs,
        best_of=best_of,
        fallback_prob=prob_a_map,
    )


def update_map_models(
    overall_map_elo: EloModel,
    team_a_id: str,
    team_b_id: str,
    map_winner_team_ids: list[str],
) -> None:
    for winner_team_id in map_winner_team_ids:
        overall_map_elo.update(team_a_id, team_b_id, winner_team_id)


def backtest_blended_match_series(
    config: BlendedSeriesBacktestConfig = BlendedSeriesBacktestConfig(),
    match_weight: float = 0.5,
) -> BacktestResult:
    """
    Blends:
        Enhanced Dynamic Match Elo
    with:
        Series probability from Overall Map Elo

    match_weight = 1.0 means pure match model.
    match_weight = 0.0 means pure map-series model.
    """
    if not 0.0 <= match_weight <= 1.0:
        raise ValueError("match_weight must be between 0 and 1.")

    match_elo = DynamicKFactorEloModel(EloConfig(k_factor=config.match_k_factor))
    recent_form = RecentFormTracker(RecentFormConfig(decay=config.form_decay))
    h2h = H2HTracker(H2HConfig(shrinkage=config.h2h_shrinkage))

    overall_map_elo = EloModel(EloConfig(k_factor=config.map_k_factor))

    team_match_counts: defaultdict[str, int] = defaultdict(int)

    y_true: list[int] = []
    y_prob: list[float] = []

    for series in load_completed_series():
        best_of = normalise_best_of(series.best_of, len(series.maps))

        match_prob_a = enhanced_dynamic_match_probability(
            match_elo=match_elo,
            recent_form=recent_form,
            h2h=h2h,
            team_a_id=series.team_a_id,
            team_b_id=series.team_b_id,
            config=config,
        )

        series_map_prob_a = series_probability_from_overall_map_elo(
            overall_map_elo=overall_map_elo,
            team_a_id=series.team_a_id,
            team_b_id=series.team_b_id,
            best_of=best_of,
            played_map_count=len(series.maps),
        )

        blended_prob_a = (
            match_weight * match_prob_a
            + (1.0 - match_weight) * series_map_prob_a
        )

        if should_score_series_prediction(
            team_match_counts,
            series.team_a_id,
            series.team_b_id,
            config.min_team_matches,
        ):
            actual_a = 1 if series.winner_team_id == series.team_a_id else 0
            y_true.append(actual_a)
            y_prob.append(blended_prob_a)

        update_match_models(
            match_elo=match_elo,
            recent_form=recent_form,
            h2h=h2h,
            team_a_id=series.team_a_id,
            team_b_id=series.team_b_id,
            winner_team_id=series.winner_team_id,
        )

        update_map_models(
            overall_map_elo=overall_map_elo,
            team_a_id=series.team_a_id,
            team_b_id=series.team_b_id,
            map_winner_team_ids=[
                map_result.winner_team_id
                for map_result in series.maps
            ],
        )

        team_match_counts[series.team_a_id] += 1
        team_match_counts[series.team_b_id] += 1

    return BacktestResult(
        name=(
            f"Blend Match+Map min={config.min_team_matches} "
            f"match_w={match_weight:g}"
        ),
        metrics=compute_binary_metrics(y_true, y_prob),
    )