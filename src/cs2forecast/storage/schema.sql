CREATE TABLE IF NOT EXISTS raw_pages (
    title TEXT PRIMARY KEY,
    source TEXT NOT NULL,
    fetched_at TEXT NOT NULL,
    revid INTEGER,
    page_timestamp TEXT,
    wikitext TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS scrape_runs (
    scrape_run_id INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at TEXT NOT NULL,
    finished_at TEXT,
    status TEXT NOT NULL,
    pages_requested INTEGER NOT NULL DEFAULT 0,
    pages_fetched INTEGER NOT NULL DEFAULT 0,
    pages_skipped INTEGER NOT NULL DEFAULT 0,
    error_message TEXT
);

CREATE TABLE IF NOT EXISTS events (
    event_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    source_page TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS teams (
    team_id TEXT PRIMARY KEY,
    canonical_name TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS team_aliases (
    alias TEXT PRIMARY KEY,
    team_id TEXT NOT NULL,
    FOREIGN KEY (team_id) REFERENCES teams(team_id)
);

CREATE TABLE IF NOT EXISTS matches (
    match_id TEXT PRIMARY KEY,
    event_id TEXT,
    date TEXT,
    best_of INTEGER,
    team_a_id TEXT NOT NULL,
    team_b_id TEXT NOT NULL,
    winner_team_id TEXT,
    source_page TEXT NOT NULL,
    raw_template_name TEXT,
    raw_template_text TEXT,
    FOREIGN KEY (event_id) REFERENCES events(event_id),
    FOREIGN KEY (team_a_id) REFERENCES teams(team_id),
    FOREIGN KEY (team_b_id) REFERENCES teams(team_id),
    FOREIGN KEY (winner_team_id) REFERENCES teams(team_id)
);

CREATE TABLE IF NOT EXISTS map_results (
    map_result_id TEXT PRIMARY KEY,
    match_id TEXT NOT NULL,
    map_index INTEGER NOT NULL,
    map_name TEXT NOT NULL,
    team_a_score INTEGER,
    team_b_score INTEGER,
    winner_team_id TEXT,
    FOREIGN KEY (match_id) REFERENCES matches(match_id),
    FOREIGN KEY (winner_team_id) REFERENCES teams(team_id)
);