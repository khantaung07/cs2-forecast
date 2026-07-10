from collections import defaultdict
from dataclasses import dataclass

from cs2forecast.backtesting.blended_series_backtest import (
    BlendedSeriesBacktestConfig,
    enhanced_dynamic_match_probability,
    series_probability_from_overall_map_elo,
    update_map_models,
    update_match_models,
)
from cs2forecast.backtesting.series_backtest import (
    load_completed_series,
    normalise_best_of,
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


FEATURE_NAMES = (
    "match_probability_a",
    "series_probability_a",
    "match_rating_diff",
    "form_diff",
    "h2h_score_a",
    "overall_map_rating_diff",
    "min_match_count",
    "match_count_diff",
    "best_of",
)


@dataclass(frozen=True)
class MLFeatureRow:
    """
    Represents the pre-match features and target for one historical match.

    All feature values are computed using only information available before 
    the match, preventing look-ahead bias during chronological backtesting.
    """
    match_id: str
    date: str
    team_a_id: str
    team_b_id: str

    match_probability_a: float
    series_probability_a: float
    blended_probability_a: float

    match_rating_diff: float
    form_diff: float
    h2h_score_a: float
    overall_map_rating_diff: float

    min_match_count: int
    match_count_diff: int
    best_of: int

    actual_a: int


def feature_vector(row: MLFeatureRow) -> list[float]:
    return [
        row.match_probability_a,
        row.series_probability_a,
        row.match_rating_diff,
        row.form_diff,
        row.h2h_score_a,
        row.overall_map_rating_diff,
        float(row.min_match_count),
        float(row.match_count_diff),
        float(row.best_of),
    ]


def build_ml_feature_rows(
    config: BlendedSeriesBacktestConfig = BlendedSeriesBacktestConfig(),
    baseline_match_weight: float = 0.5,
) -> list[MLFeatureRow]:
    """
    Generate one pre-match feature row per eligible historical series.

    Every feature is calculated before the current match result is used to
    update the model state.
    """
    if not 0.0 <= baseline_match_weight <= 1.0:
        raise ValueError("baseline_match_weight must be between 0 and 1.")

    match_elo = DynamicKFactorEloModel(
        EloConfig(k_factor=config.match_k_factor)
    )
    recent_form = RecentFormTracker(
        RecentFormConfig(decay=config.form_decay)
    )
    h2h = H2HTracker(
        H2HConfig(shrinkage=config.h2h_shrinkage)
    )
    overall_map_elo = EloModel(
        EloConfig(k_factor=config.map_k_factor)
    )

    team_match_counts: defaultdict[str, int] = defaultdict(int)
    rows: list[MLFeatureRow] = []

    for series in load_completed_series():
        team_a_id = series.team_a_id
        team_b_id = series.team_b_id

        team_a_count = team_match_counts[team_a_id]
        team_b_count = team_match_counts[team_b_id]

        best_of = normalise_best_of(
            series.best_of,
            len(series.maps),
        )

        match_probability_a = enhanced_dynamic_match_probability(
            match_elo=match_elo,
            recent_form=recent_form,
            h2h=h2h,
            team_a_id=team_a_id,
            team_b_id=team_b_id,
            config=config,
        )

        # Overall map Elo uses the same probability for every map slot.
        # Passing best_of avoids relying on how many maps were actually played.
        series_probability_a = series_probability_from_overall_map_elo(
            overall_map_elo=overall_map_elo,
            team_a_id=team_a_id,
            team_b_id=team_b_id,
            best_of=best_of,
            played_map_count=best_of,
        )

        blended_probability_a = (
            baseline_match_weight * match_probability_a
            + (1.0 - baseline_match_weight) * series_probability_a
        )

        if should_score_series_prediction(
            team_match_counts,
            team_a_id,
            team_b_id,
            config.min_team_matches,
        ):
            rows.append(
                MLFeatureRow(
                    match_id=series.match_id,
                    date=series.date,
                    team_a_id=team_a_id,
                    team_b_id=team_b_id,
                    match_probability_a=match_probability_a,
                    series_probability_a=series_probability_a,
                    blended_probability_a=blended_probability_a,
                    match_rating_diff=(
                        match_elo.get_rating(team_a_id)
                        - match_elo.get_rating(team_b_id)
                    ),
                    form_diff=(
                        recent_form.get_score(team_a_id)
                        - recent_form.get_score(team_b_id)
                    ),
                    h2h_score_a=h2h.get_score(team_a_id, team_b_id),
                    overall_map_rating_diff=(
                        overall_map_elo.get_rating(team_a_id)
                        - overall_map_elo.get_rating(team_b_id)
                    ),
                    min_match_count=min(team_a_count, team_b_count),
                    match_count_diff=team_a_count - team_b_count,
                    best_of=best_of,
                    actual_a=int(
                        series.winner_team_id == team_a_id
                    ),
                )
            )

        # Update only after extracting the pre-match features.
        update_match_models(
            match_elo=match_elo,
            recent_form=recent_form,
            h2h=h2h,
            team_a_id=team_a_id,
            team_b_id=team_b_id,
            winner_team_id=series.winner_team_id,
        )

        update_map_models(
            overall_map_elo=overall_map_elo,
            team_a_id=team_a_id,
            team_b_id=team_b_id,
            map_winner_team_ids=[
                map_result.winner_team_id
                for map_result in series.maps
            ],
        )

        team_match_counts[team_a_id] += 1
        team_match_counts[team_b_id] += 1

    return rows
