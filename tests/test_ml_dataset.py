from cs2forecast.features.ml_dataset import (
    FEATURE_NAMES,
    MLFeatureRow,
    feature_vector,
)


def test_feature_vector_matches_feature_names() -> None:
    row = MLFeatureRow(
        match_id="match-1",
        date="2026-01-01T00:00:00+00:00",
        team_a_id="a",
        team_b_id="b",
        match_probability_a=0.6,
        series_probability_a=0.65,
        blended_probability_a=0.625,
        match_rating_diff=100.0,
        form_diff=0.1,
        h2h_score_a=0.2,
        overall_map_rating_diff=50.0,
        min_match_count=10,
        match_count_diff=2,
        best_of=3,
        actual_a=1,
    )

    values = feature_vector(row)

    assert len(values) == len(FEATURE_NAMES)
    assert values == [
        0.6,
        0.65,
        100.0,
        0.1,
        0.2,
        50.0,
        10.0,
        2.0,
        3.0,
    ]
