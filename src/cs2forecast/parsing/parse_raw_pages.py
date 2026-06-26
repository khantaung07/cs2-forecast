from rich.console import Console

from cs2forecast.parsing.match_parser import parse_match_templates_from_page
from cs2forecast.storage.db import init_db
from cs2forecast.storage.repositories import (
    list_raw_pages_with_text,
    upsert_event,
    upsert_map_result,
    upsert_match,
    upsert_team,
)

console = Console()


def parse_raw_pages() -> None:
    init_db()

    raw_pages = list_raw_pages_with_text()

    total_events = 0
    total_teams = 0
    total_matches = 0
    total_maps = 0

    for raw_page in raw_pages:
        title = raw_page["title"]
        wikitext = raw_page["wikitext"]

        events, teams, matches, map_results = parse_match_templates_from_page(
            source_page=title,
            wikitext=wikitext,
        )

        for event in events:
            upsert_event(event)

        for team in teams:
            upsert_team(team)

        for match in matches:
            upsert_match(match)

        for map_result in map_results:
            upsert_map_result(map_result)

        total_events += len(events)
        total_teams += len(teams)
        total_matches += len(matches)
        total_maps += len(map_results)

        console.print(
            f"[green]parsed[/green] {title}: "
            f"events={len(events)} teams={len(teams)} "
            f"matches={len(matches)} maps={len(map_results)}"
        )

    console.print(
        f"[bold]Done.[/bold] "
        f"events={total_events} teams={total_teams} "
        f"matches={total_matches} maps={total_maps}"
    )