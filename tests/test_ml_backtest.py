import pytest

from cs2forecast.backtesting.ml_backtest import chronological_split


def test_chronological_split_preserves_order() -> None:
    train, test = chronological_split(
        [1, 2, 3, 4, 5],
        train_fraction=0.6,
    )

    assert train == [1, 2, 3]
    assert test == [4, 5]


def test_chronological_split_rejects_invalid_fraction() -> None:
    with pytest.raises(ValueError):
        chronological_split(
            [1, 2, 3],
            train_fraction=1.0,
        )


def test_chronological_split_requires_multiple_rows() -> None:
    with pytest.raises(ValueError):
        chronological_split(
            [1],
            train_fraction=0.7,
        )