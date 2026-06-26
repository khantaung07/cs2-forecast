import sqlite3
from pathlib import Path

from cs2forecast.config import DATA_DIR, DB_PATH


SCHEMA_PATH = Path(__file__).with_name("schema.sql")


def connect(db_path: Path = DB_PATH) -> sqlite3.Connection:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")

    return conn


def init_db(db_path: Path = DB_PATH) -> None:
    with connect(db_path) as conn:
        schema = SCHEMA_PATH.read_text(encoding="utf-8")
        conn.executescript(schema)