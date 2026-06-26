from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
DB_PATH = DATA_DIR / "cs2forecast.db"

LIQUIPEDIA_API_URL = "https://liquipedia.net/counterstrike/api.php"

USER_AGENT = (
    "cs2-forecast/0.1 "
    "(github.com/khantaung07/cs2-forecast; khantaung07@gmail.com)"
)

DEFAULT_REQUEST_DELAY_SECONDS = 2.1