import pytest

from cs2forecast.prediction.predictor import PredictionConfig, resolve_best_of


def test_prediction_config_default_match_weight() -> None:
    assert PredictionConfig().match_weight == 0.5


def test_resolve_best_of_uses_explicit_value() -> None:
    assert resolve_best_of(best_of=1, maps=[]) == 1
    assert resolve_best_of(best_of=3, maps=[]) == 3
    assert resolve_best_of(best_of=5, maps=[]) == 5


def test_resolve_best_of_rejects_invalid_explicit_value() -> None:
    with pytest.raises(ValueError):
        resolve_best_of(best_of=2, maps=[])


def test_resolve_best_of_from_map_count() -> None:
    assert resolve_best_of(best_of=None, maps=[]) is None
    assert resolve_best_of(best_of=None, maps=["dust2"]) == 1
    assert resolve_best_of(best_of=None, maps=["dust2", "mirage"]) == 3
    assert resolve_best_of(best_of=None, maps=["dust2", "mirage", "inferno"]) == 3
    assert resolve_best_of(
        best_of=None,
        maps=["dust2", "mirage", "inferno", "nuke"],
    ) == 5
