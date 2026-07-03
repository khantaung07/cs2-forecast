from cs2forecast.parsing.normalization import canonical_team_id


def test_team_vitality_aliases_normalize_to_vit() -> None:
    assert canonical_team_id("Vitality") == "vit"
    assert canonical_team_id("Team Vitality") == "vit"


def test_team_liquid_aliases_normalize_to_tl() -> None:
    assert canonical_team_id("Liquid") == "tl"
    assert canonical_team_id("Team Liquid") == "tl"


def test_natus_vincere_aliases_normalize_to_navi() -> None:
    assert canonical_team_id("Natus Vincere") == "navi"
    assert canonical_team_id("NAVI") == "navi"


def test_unknown_team_gets_stable_slug_id() -> None:
    assert canonical_team_id("Some New Team") == "some_new_team"