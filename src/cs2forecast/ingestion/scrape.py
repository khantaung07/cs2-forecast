from rich.console import Console

from cs2forecast.ingestion.liquipedia_client import LiquipediaClient
from cs2forecast.storage.db import init_db
from cs2forecast.storage.repositories import (
    create_scrape_run,
    finish_scrape_run,
    get_raw_page,
    upsert_raw_page,
)

console = Console()


def scrape_pages(
    titles: list[str],
    *,
    refresh: bool = False,
    continue_on_error: bool = False,
) -> None:
    """
    Fetch Liquipedia pages through the MediaWiki API and store raw wikitext locally.

    This intentionally does not access generated HTML pages.
    """
    init_db()

    scrape_run_id = create_scrape_run(pages_requested=len(titles))
    client = LiquipediaClient()

    pages_fetched = 0
    pages_skipped = 0
    errors: list[str] = []

    try:
        for title in titles:
            try:
                existing = get_raw_page(title)

                if existing is not None and not refresh:
                    pages_skipped += 1
                    console.print(f"[yellow]skip[/yellow] {title} already cached")
                    continue

                console.print(f"[cyan]fetch[/cyan] {title}")

                page = client.get_page_wikitext(title)

                upsert_raw_page(
                    title=page.title,
                    source="liquipedia_mediawiki",
                    revid=page.revid,
                    page_timestamp=page.page_timestamp,
                    wikitext=page.wikitext,
                )

                pages_fetched += 1
                console.print(
                    f"[green]stored[/green] {page.title} "
                    f"revid={page.revid} chars={len(page.wikitext)}"
                )

            except Exception as exc:
                message = f"{title}: {exc}"
                errors.append(message)
                console.print(f"[red]error[/red] {message}")

                if not continue_on_error:
                    raise

        status = "success" if not errors else "partial_success"

        finish_scrape_run(
            scrape_run_id=scrape_run_id,
            status=status,
            pages_fetched=pages_fetched,
            pages_skipped=pages_skipped,
            error_message="\n".join(errors) if errors else None,
        )

        if errors:
            console.print("[yellow]Completed with errors:[/yellow]")
            for error in errors:
                console.print(f"  - {error}")

    except Exception as exc:
        finish_scrape_run(
            scrape_run_id=scrape_run_id,
            status="failed",
            pages_fetched=pages_fetched,
            pages_skipped=pages_skipped,
            error_message=str(exc),
        )
        raise
