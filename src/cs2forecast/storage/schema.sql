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