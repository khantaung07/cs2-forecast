import math

import pytest

from cs2forecast.backtesting.metrics import compute_binary_metrics


def test_empty_metrics_are_zero() -> None:
    metrics = compute_binary_metrics([], [])

    assert metrics.n == 0
    assert metrics.accuracy == 0.0
    assert metrics.log_loss == 0.0
    assert metrics.brier_score == 0.0


def test_binary_metrics_for_simple_correct_predictions() -> None:
    metrics = compute_binary_metrics(
        y_true=[1, 0],
        y_prob=[0.75, 0.25],
    )

    assert metrics.n == 2
    assert metrics.accuracy == pytest.approx(1.0)
    assert metrics.log_loss == pytest.approx(-math.log(0.75))
    assert metrics.brier_score == pytest.approx(0.0625)


def test_metrics_reject_mismatched_lengths() -> None:
    with pytest.raises(ValueError):
        compute_binary_metrics([1, 0], [0.7])