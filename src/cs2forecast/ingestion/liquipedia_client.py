import time
from dataclasses import dataclass

import requests

from cs2forecast.config import (
    DEFAULT_REQUEST_DELAY_SECONDS,
    LIQUIPEDIA_API_URL,
    USER_AGENT,
)


@dataclass(frozen=True)
class LiquipediaPage:
    title: str
    revid: int | None
    page_timestamp: str | None
    wikitext: str


class LiquipediaClient:
    def __init__(self, min_delay_seconds: float = DEFAULT_REQUEST_DELAY_SECONDS):
        self.min_delay_seconds = min_delay_seconds
        self._last_request_at: float | None = None

        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": USER_AGENT,
                "Accept-Encoding": "gzip",
            }
        )

    def _wait_for_rate_limit(self) -> None:
        if self._last_request_at is None:
            return

        elapsed = time.monotonic() - self._last_request_at
        remaining = self.min_delay_seconds - elapsed

        if remaining > 0:
            time.sleep(remaining)

    def _get(self, params: dict[str, str]) -> dict:
        self._wait_for_rate_limit()

        response = self.session.get(
            LIQUIPEDIA_API_URL,
            params=params,
            timeout=30,
        )

        self._last_request_at = time.monotonic()

        response.raise_for_status()
        return response.json()

    def get_page_wikitext(self, title: str) -> LiquipediaPage:
        params = {
            "action": "query",
            "format": "json",
            "prop": "revisions",
            "rvprop": "ids|timestamp|content",
            "rvslots": "main",
            "titles": title,
        }

        data = self._get(params)

        pages = data.get("query", {}).get("pages", {})
        if not pages:
            raise RuntimeError(f"No pages returned for title: {title}")

        page = next(iter(pages.values()))

        if "missing" in page:
            raise ValueError(f"Liquipedia page not found: {title}")

        revisions = page.get("revisions", [])
        if not revisions:
            raise RuntimeError(f"No revisions returned for title: {title}")

        revision = revisions[0]
        slot = revision.get("slots", {}).get("main", {})

        wikitext = slot.get("*")
        if wikitext is None:
            raise RuntimeError(f"No wikitext found for title: {title}")

        return LiquipediaPage(
            title=page["title"],
            revid=revision.get("revid"),
            page_timestamp=revision.get("timestamp"),
            wikitext=wikitext,
        )
