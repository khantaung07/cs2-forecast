import typer
from rich.console import Console
from rich.table import Table

from cs2forecast.ingestion.scrape import scrape_pages
from cs2forecast.storage.db import init_db
from cs2forecast.storage.repositories import get_raw_page, list_raw_pages

from cs2forecast.parsing.template_inspector import (
    count_templates,
    format_template_examples,
)

from cs2forecast.parsing.parse_raw_pages import parse_raw_pages
from cs2forecast.storage.db import connect

from cs2forecast.backtesting.elo_backtest import (
    backtest_constant_baseline_on_maps,
    backtest_constant_baseline_on_matches,
    backtest_map_elo,
    backtest_overall_elo,
)

from pathlib import Path
from cs2forecast.ingestion.seeds import read_seed_titles

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
    continue_on_error: bool = typer.Option(False, help="Continue if one page fails."),
) -> None:
    """Fetch Liquipedia page wikitext through the MediaWiki API."""
    scrape_pages(titles, refresh=refresh, continue_on_error=continue_on_error)

@app.command("scrape-events")
def scrape_events(
    seed_file: Path = typer.Option(
        Path("seeds/tournaments.txt"), # default
        "--seed-file",
        "-f",
        help="File containing Liquipedia tournament page titles.",
    ),
    refresh: bool = typer.Option(False, help="Re-fetch pages even if cached locally."),
    continue_on_error: bool = typer.Option(
        True,
        help="Continue scraping if one page title fails.",
    ),
) -> None:
    """Fetch tournament pages listed in a seed file."""
    if not seed_file.exists():
        console.print(f"[red]Seed file not found:[/red] {seed_file}")
        raise typer.Exit(code=1)

    titles = read_seed_titles(seed_file)

    if not titles:
        console.print(f"[yellow]No tournament titles found in:[/yellow] {seed_file}")
        raise typer.Exit(code=1)

    console.print(f"[bold]Scraping {len(titles)} tournament pages from {seed_file}[/bold]")

    scrape_pages(
        titles,
        refresh=refresh,
        continue_on_error=continue_on_error,
    )


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


@app.command("inspect-templates")
def inspect_templates(
    title: str,
    limit: int = typer.Option(40, help="Number of template names to show."),
) -> None:
    """Show template names used in a cached raw page."""
    init_db()
    page = get_raw_page(title)

    if page is None:
        console.print(f"[red]No cached page found for title:[/red] {title}")
        raise typer.Exit(code=1)

    counts = count_templates(page["wikitext"])

    table = Table(title=f"Templates in {title}")
    table.add_column("Template")
    table.add_column("Count", justify="right")

    for name, count in counts.most_common(limit):
        table.add_row(name, str(count))

    console.print(table)


@app.command("show-template")
def show_template(
    title: str,
    template_name: str,
    limit: int = typer.Option(3, help="Number of examples to show."),
) -> None:
    """Show examples of a specific template from a cached page."""
    init_db()
    page = get_raw_page(title)

    if page is None:
        console.print(f"[red]No cached page found for title:[/red] {title}")
        raise typer.Exit(code=1)

    examples = format_template_examples(
        page["wikitext"],
        template_name=template_name,
        limit=limit,
    )

    if not examples:
        console.print(f"[yellow]No templates named {template_name!r} found.[/yellow]")
        raise typer.Exit()

    for example in examples:
        console.print("-" * 80)
        console.print(example)


@app.command("parse")
def parse() -> None:
    """Parse cached raw pages into clean match and map tables."""
    parse_raw_pages()

@app.command("list-matches")
def list_matches(
    limit: int = typer.Option(20, help="Maximum number of matches to show."),
) -> None:
    """List parsed matches."""
    init_db()

    with connect() as conn:
        rows = conn.execute(
            """
            SELECT
                m.date,
                ta.canonical_name AS team_a,
                tb.canonical_name AS team_b,
                tw.canonical_name AS winner,
                m.best_of,
                m.source_page,
                SUM(CASE WHEN mr.winner_team_id = m.team_a_id THEN 1 ELSE 0 END) AS team_a_maps,
                SUM(CASE WHEN mr.winner_team_id = m.team_b_id THEN 1 ELSE 0 END) AS team_b_maps
            FROM matches m
            JOIN teams ta ON ta.team_id = m.team_a_id
            JOIN teams tb ON tb.team_id = m.team_b_id
            LEFT JOIN teams tw ON tw.team_id = m.winner_team_id
            LEFT JOIN map_results mr ON mr.match_id = m.match_id
            GROUP BY
                m.match_id,
                m.date,
                ta.canonical_name,
                tb.canonical_name,
                tw.canonical_name,
                m.best_of,
                m.source_page
            ORDER BY m.date
            LIMIT ?;
            """,
            (limit,),
        ).fetchall()

    table = Table(title="Parsed Matches")
    table.add_column("Date")
    table.add_column("Team A")
    table.add_column("Team B")
    table.add_column("Score")
    table.add_column("Winner")
    table.add_column("BO")
    table.add_column("Source")

    for row in rows:
        team_a_maps = row["team_a_maps"] or 0
        team_b_maps = row["team_b_maps"] or 0
        score = f"{team_a_maps}-{team_b_maps}"

        table.add_row(
            str(row["date"]),
            row["team_a"],
            row["team_b"],
            score,
            row["winner"] or "",
            str(row["best_of"] or ""),
            row["source_page"],
        )

    console.print(table)


@app.command("list-maps")
def list_maps(
    limit: int = typer.Option(30, help="Maximum number of maps to show."),
) -> None:
    """List parsed map results."""
    init_db()

    with connect() as conn:
        rows = conn.execute(
            """
            SELECT
                m.date,
                ta.canonical_name AS team_a,
                tb.canonical_name AS team_b,
                mr.map_index,
                mr.map_name,
                mr.team_a_score,
                mr.team_b_score,
                tw.canonical_name AS winner
            FROM map_results mr
            JOIN matches m ON m.match_id = mr.match_id
            JOIN teams ta ON ta.team_id = m.team_a_id
            JOIN teams tb ON tb.team_id = m.team_b_id
            LEFT JOIN teams tw ON tw.team_id = mr.winner_team_id
            ORDER BY m.date, mr.map_index
            LIMIT ?;
            """,
            (limit,),
        ).fetchall()

    table = Table(title="Parsed Map Results")
    table.add_column("Date")
    table.add_column("Team A")
    table.add_column("Team B")
    table.add_column("Map #")
    table.add_column("Map")
    table.add_column("Score")
    table.add_column("Winner")

    for row in rows:
        score = f"{row['team_a_score']}-{row['team_b_score']}"

        table.add_row(
            str(row["date"]),
            row["team_a"],
            row["team_b"],
            str(row["map_index"]),
            row["map_name"],
            score,
            row["winner"] or "",
        )

    console.print(table)


@app.command("list-teams")
def list_teams(
    source_page: str | None = typer.Option(
        None,
        "--source-page",
        "-s",
        help="Only show teams from one Liquipedia source page.",
    ),
) -> None:
    """List parsed teams, optionally filtered by source page."""
    init_db()

    with connect() as conn:
        if source_page is None:
            rows = conn.execute(
                """
                WITH team_appearances AS (
                    SELECT source_page, team_a_id AS team_id
                    FROM matches

                    UNION ALL

                    SELECT source_page, team_b_id AS team_id
                    FROM matches
                )
                SELECT
                    ta.source_page,
                    t.team_id,
                    t.canonical_name,
                    COUNT(*) AS appearances
                FROM team_appearances ta
                JOIN teams t ON t.team_id = ta.team_id
                GROUP BY ta.source_page, t.team_id, t.canonical_name
                ORDER BY ta.source_page, appearances DESC, t.canonical_name;
                """
            ).fetchall()
        else:
            rows = conn.execute(
                """
                WITH team_appearances AS (
                    SELECT source_page, team_a_id AS team_id
                    FROM matches
                    WHERE source_page = ?

                    UNION ALL

                    SELECT source_page, team_b_id AS team_id
                    FROM matches
                    WHERE source_page = ?
                )
                SELECT
                    ta.source_page,
                    t.team_id,
                    t.canonical_name,
                    COUNT(*) AS appearances
                FROM team_appearances ta
                JOIN teams t ON t.team_id = ta.team_id
                GROUP BY ta.source_page, t.team_id, t.canonical_name
                ORDER BY appearances DESC, t.canonical_name;
                """,
                (source_page, source_page),
            ).fetchall()

    table = Table(title="Parsed Teams")
    table.add_column("Source")
    table.add_column("Team ID")
    table.add_column("Name")
    table.add_column("Appearances", justify="right")

    for row in rows:
        table.add_row(
            row["source_page"],
            row["team_id"],
            row["canonical_name"],
            str(row["appearances"]),
        )

    console.print(table)


@app.command("backtest-elo")
def backtest_elo(
    k_factor: float = typer.Option(32.0, help="Elo K-factor."),
) -> None:
    """Run chronological Elo backtests on parsed matches and maps."""
    init_db()

    results = [
        backtest_constant_baseline_on_matches(),
        backtest_overall_elo(k_factor=k_factor),
        backtest_constant_baseline_on_maps(),
        backtest_map_elo(k_factor=k_factor),
    ]

    table = Table(title="Elo Backtest")
    table.add_column("Model")
    table.add_column("N", justify="right")
    table.add_column("Accuracy", justify="right")
    table.add_column("Log Loss", justify="right")
    table.add_column("Brier", justify="right")

    for result in results:
        metrics = result.metrics

        table.add_row(
            result.name,
            str(metrics.n),
            f"{metrics.accuracy:.3f}",
            f"{metrics.log_loss:.3f}",
            f"{metrics.brier_score:.3f}",
        )

    console.print(table)