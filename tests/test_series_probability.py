import pytest

from cs2forecast.backtesting.series_backtest import series_win_probability


def test_bo1_probability_is_single_map_probability() -> None:
    result = series_win_probability(
        map_probs=[0.6],
        best_of=1,
        fallback_prob=0.5,
    )

    assert result == pytest.approx(0.6)


def test_bo3_with_equal_50_50_maps_is_50_50() -> None:
    result = series_win_probability(
        map_probs=[0.5, 0.5, 0.5],
        best_of=3,
        fallback_prob=0.5,
    )

    assert result == pytest.approx(0.5)


def test_bo3_with_same_60_percent_map_prob() -> None:
    result = series_win_probability(
        map_probs=[0.6, 0.6, 0.6],
        best_of=3,
        fallback_prob=0.5,
    )

    # P(2-0) + P(2-1)
    # = 0.6*0.6 + 0.6*0.4*0.6 + 0.4*0.6*0.6
    assert result == pytest.approx(0.648)


def test_bo3_uses_fallback_probability_for_unplayed_decider() -> None:
    result = series_win_probability(
        map_probs=[0.6, 0.7],
        best_of=3,
        fallback_prob=0.5,
    )

    # P(win maps 1 and 2)
    # + P(win map 1, lose map 2, win fallback map 3)
    # + P(lose map 1, win map 2, win fallback map 3)
    expected = (0.6 * 0.7) + (0.6 * 0.3 * 0.5) + (0.4 * 0.7 * 0.5)

    assert result == pytest.approx(expected)


def test_bo5_with_equal_50_50_maps_is_50_50() -> None:
    result = series_win_probability(
        map_probs=[0.5, 0.5, 0.5, 0.5, 0.5],
        best_of=5,
        fallback_prob=0.5,
    )

    assert result == pytest.approx(0.5)


def test_bo3_certain_two_zero_win() -> None:
    result = series_win_probability(
        map_probs=[1.0, 1.0, 0.0],
        best_of=3,
        fallback_prob=0.5,
    )

    assert result == pytest.approx(1.0)
