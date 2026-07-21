import re

import mwparserfromhell
from mwparserfromhell.nodes import Template

from cs2forecast.parsing.models import ParsedEvent, ParsedMapResult, ParsedMatch, ParsedTeam
from cs2forecast.parsing.normalization import clean_text, stable_id, canonical_team_id

from datetime import datetime, timedelta, timezone

MATCH_TEMPLATE_NAMES = {"match"}


def get_param(template: Template, *names: str) -> str | None:
    for name in names:
        if template.has(name):
            value = str(template.get(name).value).strip()
            if value:
                return value
    return None


def clean_param(template: Template, *names: str) -> str | None:
    value = get_param(template, *names)
    return clean_text(value) if value is not None else None


def parse_int(value: str | None) -> int | None:
    if value is None:
        return None

    match = re.search(r"-?\d+", value)
    if not match:
        return None

    return int(match.group())


def extract_team_id_from_opponent(value: str | None) -> str | None:
    """
    Parse values like:
        {{TeamOpponent|tl}}
        {{TeamOpponent|heroic}}

    For now, we use the first positional parameter as the team identifier.
    """
    if value is None:
        return None

    wikicode = mwparserfromhell.parse(value)
    templates = wikicode.filter_templates(recursive=True)

    for template in templates:
        name = clean_text(template.name).lower()

        if name != "teamopponent":
            continue

        # Positional parameter 1 in {{TeamOpponent|tl}}
        if template.has("1"):
            team = clean_text(template.get("1").value)
            return canonical_team_id(team)

    # Fallback if it somehow is plain text.
    cleaned = clean_text(value)
    return canonical_team_id(cleaned) if cleaned else None


def extract_team_display_name_from_opponent(value: str | None) -> str | None:
    """
    For now, the display name is just the TeamOpponent code.
    Later we can resolve aliases to full names.
    """
    team_id = extract_team_id_from_opponent(value)
    return team_id if team_id else None


def parse_map_template(value: str | None) -> Template | None:
    if value is None:
        return None

    wikicode = mwparserfromhell.parse(value)

    for template in wikicode.filter_templates(recursive=True):
        name = clean_text(template.name).lower()

        if name == "map":
            return template

    return None


def get_map_param(map_template: Template, name: str) -> str | None:
    if not map_template.has(name):
        return None

    value = str(map_template.get(name).value).strip()
    return clean_text(value) if value else None


def score_component(map_template: Template, name: str) -> int:
    value = get_map_param(map_template, name)
    return parse_int(value) or 0


def compute_team_score(map_template: Template, team_number: int) -> int:
    """
    Compute score from regulation and overtime side scores.

    Regulation fields:
        t1t, t1ct, t2t, t2ct

    Overtime fields can look like:
        o1t1t, o1t1ct, o1t2t, o1t2ct
        o2t1t, o2t1ct, o2t2t, o2t2ct
    """
    prefix = f"t{team_number}"

    total = score_component(map_template, f"{prefix}t")
    total += score_component(map_template, f"{prefix}ct")

    # Support multiple overtimes: o1, o2, o3...
    for overtime_index in range(1, 10):
        overtime_prefix = f"o{overtime_index}t{team_number}"

        has_any_overtime_field = (
            map_template.has(f"{overtime_prefix}t")
            or map_template.has(f"{overtime_prefix}ct")
        )

        if not has_any_overtime_field:
            continue

        total += score_component(map_template, f"{overtime_prefix}t")
        total += score_component(map_template, f"{overtime_prefix}ct")

    return total

TIMEZONE_OFFSETS = {
    "UTC": 0,
    "GMT": 0,
    "CET": 1,
    "CEST": 2,
    "EET": 2,
    "EEST": 3,
    "ET": -5,
    "EST": -5,
    "EDT": -4,
    "CT": -6,
    "CST": -6,
    "CDT": -5,
    "PT": -8,
    "PST": -8,
    "PDT": -7,
}


def extract_timezone_name(raw_date: str) -> str | None:
    """
    Extract timezone from values like:
        January 29, 2025 - 18:55 {{Abbr/CET}}
        January 29, 2025 - 18:55 {{Abbr|CET}}
    """
    slash_match = re.search(r"\{\{\s*Abbr/([^}|]+)", raw_date)
    if slash_match:
        return slash_match.group(1).strip().upper()

    pipe_match = re.search(r"\{\{\s*Abbr\s*\|\s*([^}|]+)", raw_date)
    if pipe_match:
        return pipe_match.group(1).strip().upper()

    return None


def clean_date_string(raw_date: str) -> str:
    """
    Convert:
        January 29, 2025 - 18:55 {{Abbr/CET}}

    into:
        January 29, 2025 18:55
    """
    without_templates = re.sub(r"\{\{[^{}]+\}\}", "", raw_date)
    without_dash = without_templates.replace(" - ", " ")
    cleaned = clean_text(without_dash)
    return re.sub(r"\s+", " ", cleaned).strip()


def parse_date(raw_date: str | None) -> str | None:
    """
    Normalize Liquipedia match dates to UTC ISO-8601 strings.

    Example:
        January 29, 2025 - 18:55 {{Abbr/CET}}
    becomes:
        2025-01-29T17:55:00+00:00
    """
    if raw_date is None:
        return None

    timezone_name = extract_timezone_name(raw_date) or "UTC"
    offset_hours = TIMEZONE_OFFSETS.get(timezone_name)

    if offset_hours is None:
        # Conservative fallback: keep timezone as UTC if unknown.
        offset_hours = 0

    cleaned = clean_date_string(raw_date)

    formats = [
        "%B %d, %Y %H:%M",
        "%b %d, %Y %H:%M",
        "%B %d, %Y",
        "%b %d, %Y",
    ]

    parsed: datetime | None = None

    for fmt in formats:
        try:
            parsed = datetime.strptime(cleaned, fmt)
            break
        except ValueError:
            continue

    if parsed is None:
        # Fallback: store cleaned raw-ish value rather than crashing the whole parse.
        return cleaned

    local_tz = timezone(timedelta(hours=offset_hours))
    local_dt = parsed.replace(tzinfo=local_tz)
    utc_dt = local_dt.astimezone(timezone.utc)

    return utc_dt.isoformat()


def parse_match_templates_from_page(
    *,
    source_page: str,
    wikitext: str,
) -> tuple[
    list[ParsedEvent],
    list[ParsedTeam],
    list[ParsedMatch],
    list[ParsedMapResult],
]:
    wikicode = mwparserfromhell.parse(wikitext)

    events: list[ParsedEvent] = []
    teams: list[ParsedTeam] = []
    matches: list[ParsedMatch] = []
    map_results: list[ParsedMapResult] = []

    event_id = stable_id(source_page)

    events.append(
        ParsedEvent(
            event_id=event_id,
            name=source_page,
            source_page=source_page,
        )
    )

    for template in wikicode.filter_templates(recursive=True):
        template_name = clean_text(template.name).lower()

        if template_name not in MATCH_TEMPLATE_NAMES:
            continue

        finished = clean_param(template, "finished")

        explicitly_finished = finished == "true"
        infer_finished_from_maps = finished is None

        # Reject matches explicitly marked unfinished, such as:
        # |finished=false
        #
        # When |finished= is blank, we may still accept the match if its
        # completed maps establish a valid series winner.
        if not explicitly_finished and not infer_finished_from_maps:
            continue

        opponent1_raw = get_param(template, "opponent1")
        opponent2_raw = get_param(template, "opponent2")

        team_a_id = extract_team_id_from_opponent(opponent1_raw)
        team_b_id = extract_team_id_from_opponent(opponent2_raw)

        if not team_a_id or not team_b_id:
            continue

        team_a_name = (
            extract_team_display_name_from_opponent(opponent1_raw)
            or team_a_id
        )
        team_b_name = (
            extract_team_display_name_from_opponent(opponent2_raw)
            or team_b_id
        )

        teams.append(
            ParsedTeam(
                team_id=team_a_id,
                canonical_name=team_a_name,
            )
        )
        teams.append(
            ParsedTeam(
                team_id=team_b_id,
                canonical_name=team_b_name,
            )
        )

        date = parse_date(get_param(template, "date"))
        hltv_id = clean_param(template, "hltv")

        # Prefer the HLTV ID because it is stable and unique.
        match_id = (
            f"hltv_{hltv_id}"
            if hltv_id
            else stable_id(
                source_page,
                date or "unknown_date",
                team_a_id,
                team_b_id,
            )
        )

        # Infer the intended series format from the map slots.
        #
        # A BO3 normally contains map1-map3, even when the final map
        # is skipped because one team wins the first two maps.
        best_of: int | None = None

        if template.has("map5"):
            best_of = 5
        elif template.has("map3"):
            best_of = 3
        elif template.has("map1"):
            best_of = 1

        wins_required = (
            best_of // 2 + 1
            if best_of is not None
            else None
        )

        parsed_maps_for_match: list[ParsedMapResult] = []

        team_a_maps_won = 0
        team_b_maps_won = 0

        for map_index in range(1, 8):
            raw_map = get_param(template, f"map{map_index}")
            map_template = parse_map_template(raw_map)

            if map_template is None:
                continue

            map_finished = get_map_param(map_template, "finished")

            if map_finished == "skip":
                continue

            # If the outer match completion flag is blank, only use maps
            # that are themselves explicitly marked as completed.
            #
            # This prevents an ongoing match from being inferred as finished.
            if infer_finished_from_maps and map_finished != "true":
                continue

            map_name = get_map_param(map_template, "map")

            if not map_name:
                continue

            team_a_score = compute_team_score(
                map_template,
                team_number=1,
            )
            team_b_score = compute_team_score(
                map_template,
                team_number=2,
            )

            map_winner_team_id: str | None = None

            if team_a_score > team_b_score:
                map_winner_team_id = team_a_id
                team_a_maps_won += 1
            elif team_b_score > team_a_score:
                map_winner_team_id = team_b_id
                team_b_maps_won += 1

            map_result_id = stable_id(
                match_id,
                str(map_index),
                map_name,
            )

            parsed_maps_for_match.append(
                ParsedMapResult(
                    map_result_id=map_result_id,
                    match_id=match_id,
                    map_index=map_index,
                    map_name=map_name,
                    team_a_score=team_a_score,
                    team_b_score=team_b_score,
                    winner_team_id=map_winner_team_id,
                )
            )

        winner_team_id: str | None = None

        if team_a_maps_won > team_b_maps_won:
            winner_team_id = team_a_id
        elif team_b_maps_won > team_a_maps_won:
            winner_team_id = team_b_id

        # When the outer |finished= value was blank, only accept the match
        # if its completed maps establish a valid series victory.
        #
        # Examples:
        # BO1 requires 1 map win.
        # BO3 requires 2 map wins.
        # BO5 requires 3 map wins.
        if infer_finished_from_maps:
            completed_from_maps = (
                wins_required is not None
                and winner_team_id is not None
                and max(team_a_maps_won, team_b_maps_won)
                >= wins_required
            )

            if not completed_from_maps:
                continue

        # Avoid reporting a series format when no maps were successfully
        # parsed. This preserves the previous behaviour for forfeits.
        parsed_best_of = best_of if parsed_maps_for_match else None

        matches.append(
            ParsedMatch(
                match_id=match_id,
                event_id=event_id,
                date=date,
                best_of=parsed_best_of,
                team_a_id=team_a_id,
                team_b_id=team_b_id,
                winner_team_id=winner_team_id,
                source_page=source_page,
                raw_template_name=str(template.name).strip(),
                raw_template_text=str(template),
            )
        )

        map_results.extend(parsed_maps_for_match)

    return events, teams, matches, map_results
