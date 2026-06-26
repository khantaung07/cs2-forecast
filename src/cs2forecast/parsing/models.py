from dataclasses import dataclass


@dataclass(frozen=True)
class ParsedEvent:
    event_id: str
    name: str
    source_page: str


@dataclass(frozen=True)
class ParsedTeam:
    team_id: str
    canonical_name: str


@dataclass(frozen=True)
class ParsedMatch:
    match_id: str
    event_id: str
    date: str | None
    best_of: int | None
    team_a_id: str
    team_b_id: str
    winner_team_id: str | None
    source_page: str
    raw_template_name: str
    raw_template_text: str


@dataclass(frozen=True)
class ParsedMapResult:
    map_result_id: str
    match_id: str
    map_index: int
    map_name: str
    team_a_score: int | None
    team_b_score: int | None
    winner_team_id: str | None