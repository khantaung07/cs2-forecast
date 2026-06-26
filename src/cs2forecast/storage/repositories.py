from datetime import datetime, timezone
from typing import Any

from cs2forecast.storage.db import connect


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