from dataclasses import dataclass
from typing import TypeVar

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from cs2forecast.backtesting.elo_backtest import BacktestResult
from cs2forecast.backtesting.metrics import compute_binary_metrics
from cs2forecast.backtesting.blended_series_backtest import (
    BlendedSeriesBacktestConfig,
)
from cs2forecast.features.ml_dataset import (
    FEATURE_NAMES,
    build_ml_feature_rows,
    feature_vector,
)


T = TypeVar("T")


@dataclass(frozen=True)
class MLBacktestReport:
    results: list[BacktestResult]
    train_size: int
    test_size: int
    first_test_date: str
    logistic_coefficients: tuple[tuple[str, float], ...]


def chronological_split(
    rows: list[T],
    train_fraction: float,
) -> tuple[list[T], list[T]]:
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be between 0 and 1.")

    if len(rows) < 2:
        raise ValueError("At least two rows are required.")

    split_index = int(len(rows) * train_fraction)
    split_index = max(1, min(split_index, len(rows) - 1))

    return rows[:split_index], rows[split_index:]


def positive_class_probabilities(
    model: object,
    features: list[list[float]],
) -> list[float]:
    probabilities = model.predict_proba(features)
    classes = list(model.classes_)

    try:
        positive_index = classes.index(1)
    except ValueError as error:
        raise ValueError(
            "The fitted classifier does not contain positive class 1."
        ) from error

    return [
        float(row[positive_index])
        for row in probabilities
    ]


def run_ml_backtest(
    min_team_matches: int = 5,
    train_fraction: float = 0.7,
    logistic_c: float = 1.0,
) -> MLBacktestReport:
    config = BlendedSeriesBacktestConfig(
        min_team_matches=min_team_matches,
    )

    rows = build_ml_feature_rows(
        config=config,
        baseline_match_weight=0.5,
    )

    train_rows, test_rows = chronological_split(
        rows,
        train_fraction=train_fraction,
    )

    x_train = [feature_vector(row) for row in train_rows]
    y_train = [row.actual_a for row in train_rows]

    x_test = [feature_vector(row) for row in test_rows]
    y_test = [row.actual_a for row in test_rows]

    if len(set(y_train)) < 2:
        raise ValueError(
            "Training period must contain both outcome classes."
        )

    logistic = Pipeline(
        steps=[
            ("scale", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    C=logistic_c,
                    max_iter=1000,
                    solver="lbfgs",
                ),
            ),
        ]
    )

    gradient_boosting = HistGradientBoostingClassifier(
        loss="log_loss",
        learning_rate=0.05,
        max_iter=150,
        max_leaf_nodes=7,
        min_samples_leaf=20,
        l2_regularization=1.0,
        early_stopping=False,
        random_state=42,
    )

    logistic.fit(x_train, y_train)
    gradient_boosting.fit(x_train, y_train)

    logistic_probabilities = positive_class_probabilities(
        logistic,
        x_test,
    )
    gradient_boosting_probabilities = positive_class_probabilities(
        gradient_boosting,
        x_test,
    )

    results = [
        BacktestResult(
            name="Constant 50/50 Holdout",
            metrics=compute_binary_metrics(
                y_test,
                [0.5] * len(y_test),
            ),
        ),
        BacktestResult(
            name="Enhanced Dynamic Match Holdout",
            metrics=compute_binary_metrics(
                y_test,
                [
                    row.match_probability_a
                    for row in test_rows
                ],
            ),
        ),
        BacktestResult(
            name="Blended Match+Map Holdout",
            metrics=compute_binary_metrics(
                y_test,
                [
                    row.blended_probability_a
                    for row in test_rows
                ],
            ),
        ),
        BacktestResult(
            name="Logistic Regression",
            metrics=compute_binary_metrics(
                y_test,
                logistic_probabilities,
            ),
        ),
        BacktestResult(
            name="Histogram Gradient Boosting",
            metrics=compute_binary_metrics(
                y_test,
                gradient_boosting_probabilities,
            ),
        ),
    ]

    logistic_model = logistic.named_steps["model"]

    logistic_coefficients = tuple(
        (
            feature_name,
            float(coefficient),
        )
        for feature_name, coefficient in zip(
            FEATURE_NAMES,
            logistic_model.coef_[0],
            strict=True,
        )
    )

    return MLBacktestReport(
        results=results,
        train_size=len(train_rows),
        test_size=len(test_rows),
        first_test_date=test_rows[0].date,
        logistic_coefficients=logistic_coefficients,
    )
