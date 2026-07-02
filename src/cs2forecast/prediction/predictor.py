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
)
from cs2forecast.features.elo import DynamicKFactorEloModel, EloConfig, EloModel
from cs2forecast.features.h2h import H2HConfig, H2HTracker
from cs2forecast.features.recent_form import RecentFormConfig, RecentFormTracker
from cs2forecast.parsing.normalization import canonical_team_id
from cs2forecast.storage.db import connect


@dataclass(frozen=True)
class PredictionConfig:
    match_k_factor: float = 32.0
    map_k_factor: float = 32.0

    form_decay: float = 0.95
    form_weight: float = 100.0

    h2h_shrinkage: float = 3.0
    h2h_weight: float = 50.0

    match_weight: float = 0.5


@dataclass(frozen=True)
class PredictionState:
    match_elo: DynamicKFactorEloModel
    recent_form: RecentFormTracker
    h2h: H2HTracker
    overall_map_elo: EloModel
    latest_match_date: str | None
    completed_series_count: int


@dataclass(frozen=True)
class MapPrediction:
    map_name: str
    team_a_probability: float
    team_b_probability: float


@dataclass(frozen=True)
class MatchPrediction:
    team_a_input: str
    team_b_input: str

    team_a_id: str
    team_b_id: str

    team_a_name: str
    team_b_name: str

    match_probability_a: float
    match_probability_b: float

    series_probability_a: float | None
    series_probability_b: float | None

    final_probability_a: float
    final_probability_b: float

    match_weight: float
    best_of: int | None
    maps: list[MapPrediction]

    team_a_rating: float
    team_b_rating: float
    team_a_form: float
    team_b_form: float
    h2h_score_a: float

    latest_match_date: str | None
    completed_series_count: int


def to_backtest_config(config: PredictionConfig) -> BlendedSeriesBacktestConfig:
    return BlendedSeriesBacktestConfig(
        match_k_factor=config.match_k_factor,
        map_k_factor=config.map_k_factor,
        form_decay=config.form_decay,
        form_weight=config.form_weight,
        h2h_shrinkage=config.h2h_shrinkage,
        h2h_weight=config.h2h_weight,
    )


def build_prediction_state(config: PredictionConfig = PredictionConfig()) -> PredictionState:
    model_config = to_backtest_config(config)

    match_elo = DynamicKFactorEloModel(EloConfig(k_factor=config.match_k_factor))
    recent_form = RecentFormTracker(RecentFormConfig(decay=config.form_decay))
    h2h = H2HTracker(H2HConfig(shrinkage=config.h2h_shrinkage))
    overall_map_elo = EloModel(EloConfig(k_factor=config.map_k_factor))

    latest_match_date: str | None = None
    completed_series_count = 0

    for series in load_completed_series():
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

        latest_match_date = series.date
        completed_series_count += 1

    # Keep the variable used so future edits don't accidentally remove the shared
    # config conversion and drift from the backtest implementation.
    _ = model_config

    return PredictionState(
        match_elo=match_elo,
        recent_form=recent_form,
        h2h=h2h,
        overall_map_elo=overall_map_elo,
        latest_match_date=latest_match_date,
        completed_series_count=completed_series_count,
    )


def load_team_name(team_id: str) -> str:
    with connect() as conn:
        row = conn.execute(
            """
            SELECT canonical_name
            FROM teams
            WHERE team_id = ?;
            """,
            (team_id,),
        ).fetchone()

    if row is None:
        return team_id

    return row["canonical_name"]


def resolve_best_of(best_of: int | None, maps: list[str]) -> int | None:
    if best_of is not None:
        if best_of not in {1, 3, 5}:
            raise ValueError("best_of must be one of: 1, 3, 5.")
        return best_of

    if not maps:
        return None

    return normalise_best_of(best_of=None, map_count=len(maps))


def predict_match(
    team_a: str,
    team_b: str,
    maps: list[str] | None = None,
    best_of: int | None = None,
    config: PredictionConfig = PredictionConfig(),
) -> MatchPrediction:
    if not 0.0 <= config.match_weight <= 1.0:
        raise ValueError("match_weight must be between 0 and 1.")

    cleaned_maps = [
        map_name.strip()
        for map_name in (maps or [])
        if map_name.strip()
    ]

    resolved_best_of = resolve_best_of(best_of=best_of, maps=cleaned_maps)

    team_a_id = canonical_team_id(team_a)
    team_b_id = canonical_team_id(team_b)

    state = build_prediction_state(config)
    model_config = to_backtest_config(config)

    match_probability_a = enhanced_dynamic_match_probability(
        match_elo=state.match_elo,
        recent_form=state.recent_form,
        h2h=state.h2h,
        team_a_id=team_a_id,
        team_b_id=team_b_id,
        config=model_config,
    )

    match_probability_b = 1.0 - match_probability_a

    series_probability_a: float | None = None
    series_probability_b: float | None = None
    map_predictions: list[MapPrediction] = []

    if resolved_best_of is not None:
        played_map_count = len(cleaned_maps) if cleaned_maps else resolved_best_of

        series_probability_a = series_probability_from_overall_map_elo(
            overall_map_elo=state.overall_map_elo,
            team_a_id=team_a_id,
            team_b_id=team_b_id,
            best_of=resolved_best_of,
            played_map_count=played_map_count,
        )
        series_probability_b = 1.0 - series_probability_a

        map_probability_a = state.overall_map_elo.predict_proba(team_a_id, team_b_id)

        map_predictions = [
            MapPrediction(
                map_name=map_name,
                team_a_probability=map_probability_a,
                team_b_probability=1.0 - map_probability_a,
            )
            for map_name in cleaned_maps
        ]

        final_probability_a = (
            config.match_weight * match_probability_a
            + (1.0 - config.match_weight) * series_probability_a
        )
    else:
        final_probability_a = match_probability_a

    final_probability_b = 1.0 - final_probability_a

    return MatchPrediction(
        team_a_input=team_a,
        team_b_input=team_b,
        team_a_id=team_a_id,
        team_b_id=team_b_id,
        team_a_name=load_team_name(team_a_id),
        team_b_name=load_team_name(team_b_id),
        match_probability_a=match_probability_a,
        match_probability_b=match_probability_b,
        series_probability_a=series_probability_a,
        series_probability_b=series_probability_b,
        final_probability_a=final_probability_a,
        final_probability_b=final_probability_b,
        match_weight=config.match_weight,
        best_of=resolved_best_of,
        maps=map_predictions,
        team_a_rating=state.match_elo.get_rating(team_a_id),
        team_b_rating=state.match_elo.get_rating(team_b_id),
        team_a_form=state.recent_form.get_score(team_a_id),
        team_b_form=state.recent_form.get_score(team_b_id),
        h2h_score_a=state.h2h.get_score(team_a_id, team_b_id),
        latest_match_date=state.latest_match_date,
        completed_series_count=state.completed_series_count,
    )
