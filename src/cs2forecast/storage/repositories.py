from datetime import datetime, timezone
from typing import Any

from cs2forecast.storage.db import connect

from cs2forecast.parsing.models import (
    ParsedEvent,
    ParsedMapResult,
    ParsedMatch,
    ParsedTeam,
)

def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def upsert_raw_page(
    *,
    title: str,
    source: str,
    revid: int | None,
    page_timestamp: str | None,
    wikitext: str,
) -> None:
    with connect() as conn:
        conn.execute(
            """
            INSERT INTO raw_pages (
                title,
                source,
                fetched_at,
                revid,
                page_timestamp,
                wikitext
            )
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(title) DO UPDATE SET
                source = excluded.source,
                fetched_at = excluded.fetched_at,
                revid = excluded.revid,
                page_timestamp = excluded.page_timestamp,
                wikitext = excluded.wikitext;
            """,
            (
                title,
                source,
                utc_now_iso(),
                revid,
                page_timestamp,
                wikitext,
            ),
        )


def get_raw_page(title: str) -> dict[str, Any] | None:
    with connect() as conn:
        row = conn.execute(
            """
            SELECT title, source, fetched_at, revid, page_timestamp, wikitext
            FROM raw_pages
            WHERE title = ?;
            """,
            (title,),
        ).fetchone()

    return dict(row) if row else None


def list_raw_pages() -> list[dict[str, Any]]:
    with connect() as conn:
        rows = conn.execute(
            """
            SELECT title, source, fetched_at, revid, page_timestamp, LENGTH(wikitext) AS chars
            FROM raw_pages
            ORDER BY fetched_at DESC;
            """
        ).fetchall()

    return [dict(row) for row in rows]

def list_raw_pages_with_text() -> list[dict[str, Any]]:
    with connect() as conn:
        rows = conn.execute(
            """
            SELECT title, wikitext
            FROM raw_pages
            ORDER BY title;
            """
        ).fetchall()

    return [dict(row) for row in rows]


def create_scrape_run(*, pages_requested: int) -> int:
    with connect() as conn:
        cursor = conn.execute(
            """
            INSERT INTO scrape_runs (
                started_at,
                status,
                pages_requested
            )
            VALUES (?, ?, ?);
            """,
            (utc_now_iso(), "running", pages_requested),
        )
        return int(cursor.lastrowid)


def finish_scrape_run(
    *,
    scrape_run_id: int,
    status: str,
    pages_fetched: int,
    pages_skipped: int,
    error_message: str | None = None,
) -> None:
    with connect() as conn:
        conn.execute(
            """
            UPDATE scrape_runs
            SET
                finished_at = ?,
                status = ?,
                pages_fetched = ?,
                pages_skipped = ?,
                error_message = ?
            WHERE scrape_run_id = ?;
            """,
            (
                utc_now_iso(),
                status,
                pages_fetched,
                pages_skipped,
                error_message,
                scrape_run_id,
            ),
        )


def upsert_event(event: ParsedEvent) -> None:
    with connect() as conn:
        conn.execute(
            """
            INSERT INTO events (event_id, name, source_page)
            VALUES (?, ?, ?)
            ON CONFLICT(event_id) DO UPDATE SET
                name = excluded.name,
                source_page = excluded.source_page;
            """,
            (event.event_id, event.name, event.source_page),
        )


def upsert_team(team: ParsedTeam) -> None:
    with connect() as conn:
        conn.execute(
            """
            INSERT INTO teams (team_id, canonical_name)
            VALUES (?, ?)
            ON CONFLICT(team_id) DO UPDATE SET
                canonical_name = excluded.canonical_name;
            """,
            (team.team_id, team.canonical_name),
        )


def upsert_match(match: ParsedMatch) -> None:
    with connect() as conn:
        conn.execute(
            """
            INSERT INTO matches (
                match_id,
                event_id,
                date,
                best_of,
                team_a_id,
                team_b_id,
                winner_team_id,
                source_page,
                raw_template_name,
                raw_template_text
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(match_id) DO UPDATE SET
                event_id = excluded.event_id,
                date = excluded.date,
                best_of = excluded.best_of,
                team_a_id = excluded.team_a_id,
                team_b_id = excluded.team_b_id,
                winner_team_id = excluded.winner_team_id,
                source_page = excluded.source_page,
                raw_template_name = excluded.raw_template_name,
                raw_template_text = excluded.raw_template_text;
            """,
            (
                match.match_id,
                match.event_id,
                match.date,
                match.best_of,
                match.team_a_id,
                match.team_b_id,
                match.winner_team_id,
                match.source_page,
                match.raw_template_name,
                match.raw_template_text,
            ),
        )


def upsert_map_result(map_result: ParsedMapResult) -> None:
    with connect() as conn:
        conn.execute(
            """
            INSERT INTO map_results (
                map_result_id,
                match_id,
                map_index,
                map_name,
                team_a_score,
                team_b_score,
                winner_team_id
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(map_result_id) DO UPDATE SET
                match_id = excluded.match_id,
                map_index = excluded.map_index,
                map_name = excluded.map_name,
                team_a_score = excluded.team_a_score,
                team_b_score = excluded.team_b_score,
                winner_team_id = excluded.winner_team_id;
            """,
            (
                map_result.map_result_id,
                map_result.match_id,
                map_result.map_index,
                map_result.map_name,
                map_result.team_a_score,
                map_result.team_b_score,
                map_result.winner_team_id,
            ),
        )