import typer
from rich.console import Console
from rich.table import Table

from cs2forecast.ingestion.scrape import scrape_pages
from cs2forecast.storage.db import init_db
from cs2forecast.storage.repositories import get_raw_page, list_raw_pages

app = typer.Typer(help="CS2 forecasting pipeline.")
console = Console()


@app.command("init-db")
def init_database() -> None:
    """Initialise the local SQLite database."""
    init_db()
    console.print("[green]Database initialised.[/green]")


@app.command("scrape")
def scrape(
    titles: list[str] = typer.Argument(..., help="Liquipedia page titles to fetch."),
    refresh: bool = typer.Option(False, help="Re-fetch pages even if cached locally."),
) -> None:
    """Fetch Liquipedia page wikitext through the MediaWiki API."""
    scrape_pages(titles, refresh=refresh)


@app.command("list-pages")
def list_pages() -> None:
    """List locally stored raw pages."""
    init_db()
    rows = list_raw_pages()

    table = Table(title="Raw Pages")
    table.add_column("Title")
    table.add_column("Source")
    table.add_column("Fetched At")
    table.add_column("Revision")
    table.add_column("Chars", justify="right")

    for row in rows:
        table.add_row(
            row["title"],
            row["source"],
            row["fetched_at"],
            str(row["revid"]),
            str(row["chars"]),
        )

    console.print(table)


@app.command("show-page")
def show_page(
    title: str,
    chars: int = typer.Option(1000, help="Number of characters to print."),
) -> None:
    """Print the start of a cached raw page."""
    init_db()
    page = get_raw_page(title)

    if page is None:
        console.print(f"[red]No cached page found for title:[/red] {title}")
        raise typer.Exit(code=1)

    console.print(f"[bold]{page['title']}[/bold]")
    console.print(f"revid={page['revid']} fetched_at={page['fetched_at']}")
    console.print("-" * 80)
    console.print(page["wikitext"][:chars])