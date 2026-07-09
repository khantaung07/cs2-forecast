import typer
from rich.console import Console
from rich.table import Table
from pathlib import Path

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

from cs2forecast.backtesting.enhanced_backtest import (
    EnhancedEloConfig,
    backtest_constant_50_50_filtered,
    backtest_dynamic_elo_filtered,
    backtest_enhanced_dynamic_elo,
    backtest_enhanced_elo,
    backtest_overall_elo_filtered,
    grid_search_enhanced_elo,
)

from cs2forecast.backtesting.enhanced_map_backtest import (
    EnhancedMapEloConfig,
    backtest_constant_50_50_maps_filtered,
    backtest_enhanced_map_elo,
    backtest_overall_map_elo_filtered,
    backtest_overall_plus_map_elo,
    backtest_plain_map_elo_filtered,
)

from cs2forecast.backtesting.series_backtest import (
    SeriesBacktestConfig,
    backtest_series_constant_50_50,
    backtest_series_from_overall_map_elo,
    backtest_series_from_plain_map_elo,
)

from cs2forecast.backtesting.blended_series_backtest import (
    BlendedSeriesBacktestConfig,
    backtest_blended_match_series,
)

from cs2forecast.backtesting.ml_backtest import run_ml_backtest

from cs2forecast.prediction.predictor import PredictionConfig, predict_match

from cs2forecast.ingestion.seeds import read_seed_titles

app = typer.Typer(help="CS2 forecasting pipeline.")
console = Console()


def print_backtest_results(title: str, results: list) -> None:
    table = Table(title=title)
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


def print_summary(title: str, rows: list[tuple[str, object]]) -> None:
    table = Table(title=title)
    table.add_column("Metric")
    table.add_column("Value", justify="right")

    for label, value in rows:
        table.add_row(label, str(value))

    console.print(table)

def format_probability(probability: float) -> str:
    return f"{probability * 100:.1f}%"


def parse_map_inputs(
    maps_csv: str | None,
    map_options: list[str] | None,
) -> list[str]:
    maps: list[str] = []

    if maps_csv:
        maps.extend(maps_csv.replace(",", " ").split())

    if map_options:
        maps.extend(map_options)

    return maps


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
        summary = conn.execute(
            """
            SELECT
                (SELECT COUNT(*) FROM matches) AS total_matches,
                (
                    SELECT SUM(CASE WHEN winner_team_id IS NOT NULL THEN 1 ELSE 0 END)
                    FROM matches
                )
                    AS matches_with_winner,
                (SELECT COUNT(DISTINCT source_page) FROM matches) AS source_pages,
                (
                    SELECT COUNT(DISTINCT team_id)
                    FROM (
                        SELECT team_a_id AS team_id FROM matches
                        UNION
                        SELECT team_b_id AS team_id FROM matches
                    )
                ) AS unique_teams,
                (SELECT COUNT(DISTINCT map_name) FROM map_results) AS unique_maps,
                (SELECT COUNT(DISTINCT match_id) FROM map_results) AS matches_with_maps,
                (SELECT COUNT(*) FROM map_results) AS total_maps;
            """
        ).fetchone()

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

    print_summary(
        "Match Summary",
        [
            ("Total matches", summary["total_matches"] or 0),
            ("Matches with winner", summary["matches_with_winner"] or 0),
            ("Matches with maps", summary["matches_with_maps"] or 0),
            ("Total maps", summary["total_maps"] or 0),
            ("Unique teams", summary["unique_teams"] or 0),
            ("Unique maps", summary["unique_maps"] or 0),
            ("Source pages", summary["source_pages"] or 0),
        ],
    )

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
        summary = conn.execute(
            """
            SELECT
                (SELECT COUNT(*) FROM map_results) AS total_maps,
                (
                    SELECT SUM(CASE WHEN winner_team_id IS NOT NULL THEN 1 ELSE 0 END)
                    FROM map_results
                )
                    AS maps_with_winner,
                (SELECT COUNT(DISTINCT map_name) FROM map_results) AS unique_maps,
                (SELECT COUNT(DISTINCT match_id) FROM map_results) AS matches_with_maps,
                (
                    SELECT COUNT(DISTINCT team_id)
                    FROM (
                        SELECT m.team_a_id AS team_id
                        FROM map_results mr
                        JOIN matches m ON m.match_id = mr.match_id
                        UNION
                        SELECT m.team_b_id AS team_id
                        FROM map_results mr
                        JOIN matches m ON m.match_id = mr.match_id
                    )
                ) AS unique_teams,
                (
                    SELECT COUNT(DISTINCT m.source_page)
                    FROM map_results mr
                    JOIN matches m ON m.match_id = mr.match_id
                ) AS source_pages;
            """
        ).fetchone()

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

    print_summary(
        "Map Summary",
        [
            ("Total maps", summary["total_maps"] or 0),
            ("Maps with winner", summary["maps_with_winner"] or 0),
            ("Matches with maps", summary["matches_with_maps"] or 0),
            ("Unique teams", summary["unique_teams"] or 0),
            ("Unique maps", summary["unique_maps"] or 0),
            ("Source pages", summary["source_pages"] or 0),
        ],
    )

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

    print_backtest_results("Elo Backtest", results)


@app.command("backtest-enhanced")
def backtest_enhanced(
    k_factor: float = typer.Option(32.0, help="Elo K-factor."),
    min_team_matches: int = typer.Option(
        5,
        help="Only score predictions once both teams have this many prior matches.",
    ),
    form_decay: float = typer.Option(0.95, help="Recent form decay."),
    form_weight: float = typer.Option(
        100.0,
        help="Rating-point weight for recent form.",
    ),
    h2h_shrinkage: float = typer.Option(
        3.0,
        help="Shrinkage for head-to-head results.",
    ),
    h2h_weight: float = typer.Option(
        50.0,
        help="Rating-point weight for head-to-head score.",
    ),
) -> None:
    """Run enhanced pre-ML match backtest."""
    init_db()

    config = EnhancedEloConfig(
        k_factor=k_factor,
        min_team_matches=min_team_matches,
        form_decay=form_decay,
        form_weight=form_weight,
        h2h_shrinkage=h2h_shrinkage,
        h2h_weight=h2h_weight,
    )

    results = [
        backtest_constant_50_50_filtered(
            min_team_matches=min_team_matches,
        ),
        backtest_overall_elo_filtered(
            k_factor=k_factor,
            min_team_matches=min_team_matches,
        ),
        backtest_dynamic_elo_filtered(
            min_team_matches=min_team_matches,
        ),
        backtest_enhanced_elo(config),
        backtest_enhanced_dynamic_elo(config),
    ]

    print_backtest_results("Enhanced Match Backtest", results)


@app.command("backtest-enhanced-map")
def backtest_enhanced_map(
    overall_k_factor: float = typer.Option(
        32.0,
        help="K-factor for overall map-level Elo.",
    ),
    map_k_factor: float = typer.Option(
        32.0,
        help="K-factor for map-specific Elo.",
    ),
    min_team_maps: int = typer.Option(
        5,
        help="Only score predictions once both teams have this many prior maps.",
    ),
    map_elo_weight: float = typer.Option(
        0.35,
        help="Weight applied to map-specific Elo deviation from 1500.",
    ),
    map_form_decay: float = typer.Option(
        0.95,
        help="Decay for map-specific recent form.",
    ),
    map_form_weight: float = typer.Option(
        100.0,
        help="Rating-point weight for map-specific recent form.",
    ),
) -> None:
    """Run enhanced pre-ML map backtest."""
    init_db()

    config = EnhancedMapEloConfig(
        overall_k_factor=overall_k_factor,
        map_k_factor=map_k_factor,
        min_team_maps=min_team_maps,
        map_elo_weight=map_elo_weight,
        map_form_decay=map_form_decay,
        map_form_weight=map_form_weight,
    )

    results = [
        backtest_constant_50_50_maps_filtered(
            min_team_maps=min_team_maps,
        ),
        backtest_overall_map_elo_filtered(
            k_factor=overall_k_factor,
            min_team_maps=min_team_maps,
        ),
        backtest_plain_map_elo_filtered(
            k_factor=map_k_factor,
            min_team_maps=min_team_maps,
        ),
        backtest_overall_plus_map_elo(config),
        backtest_enhanced_map_elo(config),
    ]

    print_backtest_results("Enhanced Map Backtest", results)


@app.command("grid-search-enhanced")
def grid_search_enhanced(
    min_team_matches: int = typer.Option(
        5,
        help="Only score predictions once both teams have this many prior matches.",
    ),
    limit: int = typer.Option(20, help="Number of best configurations to show."),
) -> None:
    """Grid search enhanced Elo hyperparameters."""
    init_db()

    rows = grid_search_enhanced_elo(min_team_matches=min_team_matches)

    table = Table(title=f"Enhanced Elo Grid Search min={min_team_matches}")
    table.add_column("Rank", justify="right")
    table.add_column("N", justify="right")
    table.add_column("Form Decay", justify="right")
    table.add_column("Form Weight", justify="right")
    table.add_column("H2H Shrink", justify="right")
    table.add_column("H2H Weight", justify="right")
    table.add_column("Accuracy", justify="right")
    table.add_column("Log Loss", justify="right")
    table.add_column("Brier", justify="right")

    for index, row in enumerate(rows[:limit], start=1):
        table.add_row(
            str(index),
            str(row.n),
            f"{row.form_decay:.2f}",
            f"{row.form_weight:g}",
            f"{row.h2h_shrinkage:g}",
            f"{row.h2h_weight:g}",
            f"{row.accuracy:.4f}",
            f"{row.log_loss:.5f}",
            f"{row.brier_score:.5f}",
        )

    console.print(table)


@app.command("backtest-series")
def backtest_series(
    map_k_factor: float = typer.Option(
        32.0,
        help="K-factor for map-level Elo models.",
    ),
    min_team_matches: int = typer.Option(
        5,
        help="Only score predictions once both teams have this many prior matches.",
    ),
    require_full_map_list: bool = typer.Option(
        False,
        help="Only score matches where every possible map in the series is observed.",
    ),
) -> None:
    """Run series-level backtests using map probabilities."""
    init_db()

    config = SeriesBacktestConfig(
        map_k_factor=map_k_factor,
        min_team_matches=min_team_matches,
        require_full_map_list=require_full_map_list,
    )

    results = [
        backtest_series_constant_50_50(config),
        backtest_series_from_overall_map_elo(config),
        backtest_series_from_plain_map_elo(config),
    ]

    print_backtest_results("Series Backtest", results)


@app.command("backtest-blended-series")
def backtest_blended_series(
    match_k_factor: float = typer.Option(
        32.0,
        help="K-factor for match-level dynamic Elo.",
    ),
    map_k_factor: float = typer.Option(
        32.0,
        help="K-factor for map-level Elo.",
    ),
    min_team_matches: int = typer.Option(
        5,
        help="Only score predictions once both teams have this many prior matches.",
    ),
    form_decay: float = typer.Option(
        0.95,
        help="Recent form decay.",
    ),
    form_weight: float = typer.Option(
        100.0,
        help="Rating-point weight for recent form.",
    ),
    h2h_shrinkage: float = typer.Option(
        3.0,
        help="Shrinkage for head-to-head results.",
    ),
    h2h_weight: float = typer.Option(
        50.0,
        help="Rating-point weight for head-to-head score.",
    ),
) -> None:
    """Backtest blended match-level and map-series probabilities."""
    init_db()

    config = BlendedSeriesBacktestConfig(
        match_k_factor=match_k_factor,
        map_k_factor=map_k_factor,
        min_team_matches=min_team_matches,
        form_decay=form_decay,
        form_weight=form_weight,
        h2h_shrinkage=h2h_shrinkage,
        h2h_weight=h2h_weight,
    )

    results = [
        backtest_blended_match_series(config, match_weight=1.0),
        backtest_blended_match_series(config, match_weight=0.9),
        backtest_blended_match_series(config, match_weight=0.75),
        backtest_blended_match_series(config, match_weight=0.5),
        backtest_blended_match_series(config, match_weight=0.25),
        backtest_blended_match_series(config, match_weight=0.0),
    ]

    results = [
        backtest_blended_match_series(config, match_weight=1.0),
        backtest_blended_match_series(config, match_weight=0.9),
        backtest_blended_match_series(config, match_weight=0.75),
        backtest_blended_match_series(config, match_weight=0.65),
        backtest_blended_match_series(config, match_weight=0.6),
        backtest_blended_match_series(config, match_weight=0.55),
        backtest_blended_match_series(config, match_weight=0.5),
        backtest_blended_match_series(config, match_weight=0.45),
        backtest_blended_match_series(config, match_weight=0.4),
        backtest_blended_match_series(config, match_weight=0.25),
        backtest_blended_match_series(config, match_weight=0.0),
    ]

    print_backtest_results("Blended Series Backtest", results)


@app.command("predict-match")
def predict_match_command(
    team_a: str = typer.Argument(..., help="First team name or ID."),
    team_b: str = typer.Argument(..., help="Second team name or ID."),
    best_of: int | None = typer.Option(
        None,
        "--best-of",
        help="Optional series length: 1, 3, or 5.",
    ),
    maps_csv: str | None = typer.Option(
        None,
        "--maps",
        help="Optional comma/space-separated map list, e.g. 'nuke,mirage,ancient'.",
    ),
    map_options: list[str] | None = typer.Option(
        None,
        "--map",
        "-m",
        help="Optional repeatable map name, e.g. -m nuke -m mirage -m ancient.",
    ),
    match_weight: float = typer.Option(
        0.5,
        help="Blend weight for match model when series context is available.",
    ),
) -> None:
    """Predict a CS2 match using the replayed local model state."""
    init_db()

    maps = parse_map_inputs(maps_csv=maps_csv, map_options=map_options)

    config = PredictionConfig(match_weight=match_weight)

    try:
        prediction = predict_match(
            team_a=team_a,
            team_b=team_b,
            maps=maps,
            best_of=best_of,
            config=config,
        )
    except ValueError as error:
        console.print(f"[red]{error}[/red]")
        raise typer.Exit(code=1) from error

    title = f"{prediction.team_a_name} vs {prediction.team_b_name}"

    summary_table = Table(title=title)
    summary_table.add_column("Model")
    summary_table.add_column(prediction.team_a_name, justify="right")
    summary_table.add_column(prediction.team_b_name, justify="right")

    summary_table.add_row(
        "Enhanced Dynamic Match Elo",
        format_probability(prediction.match_probability_a),
        format_probability(prediction.match_probability_b),
    )

    if prediction.series_probability_a is not None:
        summary_table.add_row(
            "Series from Overall Map Elo",
            format_probability(prediction.series_probability_a),
            format_probability(prediction.series_probability_b or 0.0),
        )

        summary_table.add_row(
            f"Blended Final Probability (match_w={prediction.match_weight:g})",
            f"[bold]{format_probability(prediction.final_probability_a)}[/bold]",
            f"[bold]{format_probability(prediction.final_probability_b)}[/bold]",
        )
    else:
        summary_table.add_row(
            "Final Probability",
            f"[bold]{format_probability(prediction.final_probability_a)}[/bold]",
            f"[bold]{format_probability(prediction.final_probability_b)}[/bold]",
        )

    console.print(summary_table)

    signal_table = Table(title="Signals")
    signal_table.add_column("Signal")
    signal_table.add_column(prediction.team_a_name, justify="right")
    signal_table.add_column(prediction.team_b_name, justify="right")

    signal_table.add_row(
        "Match Elo Rating",
        f"{prediction.team_a_rating:.1f}",
        f"{prediction.team_b_rating:.1f}",
    )

    signal_table.add_row(
        "Recent Form Score",
        f"{prediction.team_a_form:.4f}",
        f"{prediction.team_b_form:.4f}",
    )

    signal_table.add_row(
        "H2H Score",
        f"{prediction.h2h_score_a:.4f}",
        f"{-prediction.h2h_score_a:.4f}",
    )

    console.print(signal_table)

    if prediction.maps:
        map_table = Table(title=f"Map Slots from Overall Map Elo, Bo{prediction.best_of}")
        map_table.add_column("Map")
        map_table.add_column(prediction.team_a_name, justify="right")
        map_table.add_column(prediction.team_b_name, justify="right")

        for map_prediction in prediction.maps:
            map_table.add_row(
                map_prediction.map_name,
                format_probability(map_prediction.team_a_probability),
                format_probability(map_prediction.team_b_probability),
            )

        console.print(map_table)
        console.print(
            "[dim]Note: map names are displayed for series context; probabilities use "
            "overall map Elo because map-specific Elo underperformed in backtesting.[/dim]"
        )

    console.print(
        f"[dim]Model replayed {prediction.completed_series_count} completed series"
        f" through {prediction.latest_match_date}.[/dim]"
    )

@app.command("backtest-ml")
def backtest_ml(
    min_team_matches: int = typer.Option(
        5,
        help="Only include rows once both teams have this many prior matches.",
    ),
    train_fraction: float = typer.Option(
        0.7,
        help="Chronological fraction used to train the ML models.",
    ),
    logistic_c: float = typer.Option(
        1.0,
        help="Inverse regularization strength for logistic regression.",
    ),
) -> None:
    """Backtest logistic regression and gradient boosting chronologically."""
    init_db()

    try:
        report = run_ml_backtest(
            min_team_matches=min_team_matches,
            train_fraction=train_fraction,
            logistic_c=logistic_c,
        )
    except ValueError as error:
        console.print(f"[red]{error}[/red]")
        raise typer.Exit(code=1) from error

    print_backtest_results(
        "Chronological ML Holdout Backtest",
        report.results,
    )

    console.print(
        f"[dim]Training rows: {report.train_size}; "
        f"test rows: {report.test_size}; "
        f"test period begins: {report.first_test_date}.[/dim]"
    )

    coefficient_table = Table(
        title="Logistic Regression Coefficients"
    )
    coefficient_table.add_column("Feature")
    coefficient_table.add_column(
        "Coefficient",
        justify="right",
    )

    sorted_coefficients = sorted(
        report.logistic_coefficients,
        key=lambda item: abs(item[1]),
        reverse=True,
    )

    for feature_name, coefficient in sorted_coefficients:
        coefficient_table.add_row(
            feature_name,
            f"{coefficient:+.4f}",
        )

    console.print(coefficient_table)
    console.print(
        "[dim]Coefficients are based on standardized features. "
        "Positive values favour Team A.[/dim]"
    )
