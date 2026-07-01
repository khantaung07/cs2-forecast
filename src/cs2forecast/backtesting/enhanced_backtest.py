from collections import defaultdict
from dataclasses import dataclass

from cs2forecast.backtesting.elo_backtest import BacktestResult, load_completed_matches
from cs2forecast.backtesting.metrics import compute_binary_metrics
from cs2forecast.features.elo import DynamicKFactorEloModel, EloConfig, EloModel
from cs2forecast.features.h2h import H2HConfig, H2HTracker
from cs2forecast.features.recent_form import RecentFormConfig, RecentFormTracker


@dataclass(frozen=True)
class EnhancedEloConfig:
    k_factor: float = 32.0
    min_team_matches: int = 5

    form_decay: float = 0.85
    form_weight: float = 150.0

    h2h_shrinkage: float = 5.0
    h2h_weight: float = 75.0


def probability_from_ratings(rating_a: float, rating_b: float) -> float:
    return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400.0))


def should_score_prediction(
    team_match_counts: dict[str, int],
    team_a_id: str,
    team_b_id: str,
    min_team_matches: int,
) -> bool:
    return (
        team_match_counts[team_a_id] >= min_team_matches
        and team_match_counts[team_b_id] >= min_team_matches
    )


def backtest_constant_50_50_filtered(
    min_team_matches: int = 5,
) -> BacktestResult:
    team_match_counts: defaultdict[str, int] = defaultdict(int)

    y_true: list[int] = []
    y_prob: list[float] = []

    for match in load_completed_matches():
        team_a_id = match["team_a_id"]
        team_b_id = match["team_b_id"]
        winner_team_id = match["winner_team_id"]

        if should_score_prediction(
            team_match_counts,
            team_a_id,
            team_b_id,
            min_team_matches,
        ):
            actual_a = 1 if winner_team_id == team_a_id else 0
            y_true.append(actual_a)
            y_prob.append(0.5)

        team_match_counts[team_a_id] += 1
        team_match_counts[team_b_id] += 1

    return BacktestResult(
        name=f"Constant 50/50 min={min_team_matches}",
        metrics=compute_binary_metrics(y_true, y_prob),
    )


def backtest_overall_elo_filtered(
    k_factor: float = 32.0,
    min_team_matches: int = 5,
) -> BacktestResult:
    elo = EloModel(EloConfig(k_factor=k_factor))
    team_match_counts: defaultdict[str, int] = defaultdict(int)

    y_true: list[int] = []
    y_prob: list[float] = []

    for match in load_completed_matches():
        team_a_id = match["team_a_id"]
        team_b_id = match["team_b_id"]
        winner_team_id = match["winner_team_id"]

        prob_a = elo.predict_proba(team_a_id, team_b_id)

        if should_score_prediction(
            team_match_counts,
            team_a_id,
            team_b_id,
            min_team_matches,
        ):
            actual_a = 1 if winner_team_id == team_a_id else 0
            y_true.append(actual_a)
            y_prob.append(prob_a)

        elo.update(team_a_id, team_b_id, winner_team_id)

        team_match_counts[team_a_id] += 1
        team_match_counts[team_b_id] += 1

    return BacktestResult(
        name=f"Overall Elo min={min_team_matches}",
        metrics=compute_binary_metrics(y_true, y_prob),
    )


def backtest_enhanced_elo(
    config: EnhancedEloConfig = EnhancedEloConfig(),
) -> BacktestResult:
    elo = EloModel(EloConfig(k_factor=config.k_factor))
    recent_form = RecentFormTracker(RecentFormConfig(decay=config.form_decay))
    h2h = H2HTracker(H2HConfig(shrinkage=config.h2h_shrinkage))

    team_match_counts: defaultdict[str, int] = defaultdict(int)

    y_true: list[int] = []
    y_prob: list[float] = []

    for match in load_completed_matches():
        team_a_id = match["team_a_id"]
        team_b_id = match["team_b_id"]
        winner_team_id = match["winner_team_id"]

        base_rating_a = elo.get_rating(team_a_id)
        base_rating_b = elo.get_rating(team_b_id)

        base_prob_a = probability_from_ratings(base_rating_a, base_rating_b)

        form_adjustment_a = config.form_weight * recent_form.get_score(team_a_id)
        form_adjustment_b = config.form_weight * recent_form.get_score(team_b_id)

        h2h_adjustment_a = config.h2h_weight * h2h.get_score(team_a_id, team_b_id)

        adjusted_rating_a = (
            base_rating_a
            + form_adjustment_a
            + h2h_adjustment_a
        )

        adjusted_rating_b = (
            base_rating_b
            + form_adjustment_b
            - h2h_adjustment_a
        )

        enhanced_prob_a = probability_from_ratings(
            adjusted_rating_a,
            adjusted_rating_b,
        )

        actual_a = 1 if winner_team_id == team_a_id else 0

        if should_score_prediction(
            team_match_counts,
            team_a_id,
            team_b_id,
            config.min_team_matches,
        ):
            y_true.append(actual_a)
            y_prob.append(enhanced_prob_a)

        # Update only after prediction.
        elo.update(team_a_id, team_b_id, winner_team_id)

        recent_form.update(
            team_a_id=team_a_id,
            team_b_id=team_b_id,
            actual_a=float(actual_a),
            expected_a=base_prob_a,
        )

        h2h.update(team_a_id, team_b_id, winner_team_id)

        team_match_counts[team_a_id] += 1
        team_match_counts[team_b_id] += 1

    return BacktestResult(
        name=(
            f"Enhanced Elo min={config.min_team_matches} "
            f"form={config.form_weight:g} "
            f"h2h={config.h2h_weight:g}"
        ),
        metrics=compute_binary_metrics(y_true, y_prob),
    )

@dataclass(frozen=True)
class GridSearchRow:
    min_team_matches: int
    form_decay: float
    form_weight: float
    h2h_shrinkage: float
    h2h_weight: float
    n: int
    accuracy: float
    log_loss: float
    brier_score: float


def grid_search_enhanced_elo(
    min_team_matches: int = 5,
) -> list[GridSearchRow]:
    rows: list[GridSearchRow] = []

    form_decay_values = [0.75, 0.85, 0.95]
    form_weight_values = [0.0, 25.0, 50.0, 100.0, 150.0]
    h2h_shrinkage_values = [3.0, 5.0, 10.0]
    h2h_weight_values = [0.0, 25.0, 50.0, 75.0, 100.0]

    for form_decay in form_decay_values:
        for form_weight in form_weight_values:
            for h2h_shrinkage in h2h_shrinkage_values:
                for h2h_weight in h2h_weight_values:
                    config = EnhancedEloConfig(
                        min_team_matches=min_team_matches,
                        form_decay=form_decay,
                        form_weight=form_weight,
                        h2h_shrinkage=h2h_shrinkage,
                        h2h_weight=h2h_weight,
                    )

                    result = backtest_enhanced_elo(config)
                    metrics = result.metrics

                    rows.append(
                        GridSearchRow(
                            min_team_matches=min_team_matches,
                            form_decay=form_decay,
                            form_weight=form_weight,
                            h2h_shrinkage=h2h_shrinkage,
                            h2h_weight=h2h_weight,
                            n=metrics.n,
                            accuracy=metrics.accuracy,
                            log_loss=metrics.log_loss,
                            brier_score=metrics.brier_score,
                        )
                    )

    rows.sort(key=lambda row: row.log_loss)
    return rows


def backtest_dynamic_elo_filtered(
    min_team_matches: int = 5,
) -> BacktestResult:
    elo = DynamicKFactorEloModel()
    team_match_counts: defaultdict[str, int] = defaultdict(int)

    y_true: list[int] = []
    y_prob: list[float] = []

    for match in load_completed_matches():
        team_a_id = match["team_a_id"]
        team_b_id = match["team_b_id"]
        winner_team_id = match["winner_team_id"]

        prob_a = elo.predict_proba(team_a_id, team_b_id)

        if should_score_prediction(
            team_match_counts,
            team_a_id,
            team_b_id,
            min_team_matches,
        ):
            actual_a = 1 if winner_team_id == team_a_id else 0
            y_true.append(actual_a)
            y_prob.append(prob_a)

        elo.update(team_a_id, team_b_id, winner_team_id)

        team_match_counts[team_a_id] += 1
        team_match_counts[team_b_id] += 1

    return BacktestResult(
        name=f"Dynamic Elo min={min_team_matches}",
        metrics=compute_binary_metrics(y_true, y_prob),
    )


def backtest_enhanced_dynamic_elo(
    config: EnhancedEloConfig = EnhancedEloConfig(),
) -> BacktestResult:
    elo = DynamicKFactorEloModel()
    recent_form = RecentFormTracker(RecentFormConfig(decay=config.form_decay))
    h2h = H2HTracker(H2HConfig(shrinkage=config.h2h_shrinkage))

    team_match_counts: defaultdict[str, int] = defaultdict(int)

    y_true: list[int] = []
    y_prob: list[float] = []

    for match in load_completed_matches():
        team_a_id = match["team_a_id"]
        team_b_id = match["team_b_id"]
        winner_team_id = match["winner_team_id"]

        base_rating_a = elo.get_rating(team_a_id)
        base_rating_b = elo.get_rating(team_b_id)

        base_prob_a = probability_from_ratings(base_rating_a, base_rating_b)

        form_adjustment_a = config.form_weight * recent_form.get_score(team_a_id)
        form_adjustment_b = config.form_weight * recent_form.get_score(team_b_id)

        h2h_adjustment_a = config.h2h_weight * h2h.get_score(team_a_id, team_b_id)

        adjusted_rating_a = (
            base_rating_a
            + form_adjustment_a
            + h2h_adjustment_a
        )

        adjusted_rating_b = (
            base_rating_b
            + form_adjustment_b
            - h2h_adjustment_a
        )

        enhanced_prob_a = probability_from_ratings(
            adjusted_rating_a,
            adjusted_rating_b,
        )

        actual_a = 1 if winner_team_id == team_a_id else 0

        if should_score_prediction(
            team_match_counts,
            team_a_id,
            team_b_id,
            config.min_team_matches,
        ):
            y_true.append(actual_a)
            y_prob.append(enhanced_prob_a)

        elo.update(team_a_id, team_b_id, winner_team_id)

        recent_form.update(
            team_a_id=team_a_id,
            team_b_id=team_b_id,
            actual_a=float(actual_a),
            expected_a=base_prob_a,
        )

        h2h.update(team_a_id, team_b_id, winner_team_id)

        team_match_counts[team_a_id] += 1
        team_match_counts[team_b_id] += 1

    return BacktestResult(
        name=(
            f"Enhanced Dynamic Elo min={config.min_team_matches} "
            f"form={config.form_weight:g} "
            f"h2h={config.h2h_weight:g}"
        ),
        metrics=compute_binary_metrics(y_true, y_prob),
    )
