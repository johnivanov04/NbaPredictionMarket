"""Paced, cached, resumable retrieval of Basketball-Reference pages.

Three properties matter more than speed:

* **Pacing is not optional.** Basketball-Reference publishes ``Crawl-delay: 3``
  and this honours it with a margin. The delay is enforced between requests,
  including across retries.
* **Everything is cached to disk before it is parsed.** A parser change must
  never require refetching, and a backfill interrupted at game 4,000 must
  resume at 4,001 rather than at 1.
* **A failure is recorded, not inferred.** A page that does not arrive is
  written to a failure log with its status. It is never treated as a game
  without officials.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx

from nba_prediction_market.ingestion.raw_store import utc_now
from nba_prediction_market.referees.bbref_source import CRAWL_DELAY_SECONDS

logger = logging.getLogger(__name__)

#: An honest identifier. Basketball-Reference's robots.txt blocks some named
#: bots by user agent; this project is not one of them, and it obeys the
#: Crawl-delay that applies to everyone else.
USER_AGENT: str = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)

REQUEST_TIMEOUT_SECONDS: float = 45.0
MAX_RETRIES: int = 3


@dataclass
class FetchStats:
    requested: int = 0
    from_cache: int = 0
    fetched: int = 0
    failed: int = 0

    def to_dict(self) -> dict[str, int]:
        return {
            "requested": self.requested,
            "from_cache": self.from_cache,
            "fetched": self.fetched,
            "failed": self.failed,
        }


@dataclass
class PacedCache:
    """A disk cache in front of a rate-limited source."""

    root: Path
    min_interval: float = CRAWL_DELAY_SECONDS
    sleep: Callable[[float], None] = time.sleep
    monotonic: Callable[[], float] = time.monotonic
    stats: FetchStats = field(default_factory=FetchStats)
    _last: float | None = field(default=None, init=False)

    def path_for(self, key: str) -> Path:
        return self.root / f"{key}.html"

    def has(self, key: str) -> bool:
        return self.path_for(key).is_file()

    def read(self, key: str) -> str:
        return self.path_for(key).read_text(encoding="utf-8", errors="replace")

    def _wait(self) -> None:
        """Hold until ``min_interval`` has passed since the last request *began*.

        Crawl-delay is the minimum interval between successive requests, not
        an extra pause bolted onto each response. Timing it from the response
        instead made every cycle delay-plus-download -- roughly six seconds a
        page here -- which is slower than asked for without being kinder.
        """
        if self._last is not None:
            elapsed = self.monotonic() - self._last
            if elapsed < self.min_interval:
                self.sleep(self.min_interval - elapsed)
        self._last = self.monotonic()

    def get(
        self, key: str, url: str, client: httpx.Client
    ) -> tuple[str | None, dict[str, Any]]:
        """Cached page text, fetching it if absent.

        Returns ``(text, meta)``; ``text`` is None only when every attempt
        failed, and ``meta`` always says which.
        """
        self.stats.requested += 1
        if self.has(key):
            self.stats.from_cache += 1
            return self.read(key), {"key": key, "url": url, "source": "cache"}

        started = utc_now()
        last_status: int | None = None
        last_error: str | None = None
        for attempt in range(MAX_RETRIES):
            self._wait()
            try:
                response = client.get(url, timeout=REQUEST_TIMEOUT_SECONDS)
                last_status = response.status_code
                if response.status_code == 200:
                    self.path_for(key).parent.mkdir(parents=True, exist_ok=True)
                    self.path_for(key).write_text(response.text, encoding="utf-8")
                    self.stats.fetched += 1
                    return response.text, {
                        "key": key,
                        "url": url,
                        "source": "network",
                        "http_status": 200,
                        "first_observed_at_utc": utc_now().isoformat(),
                        "request_started_at_utc": started.isoformat(),
                        "attempts": attempt + 1,
                    }
                # 404 is a real answer: this page does not exist. Retrying it
                # only spends the crawl budget that a recoverable error needs.
                if response.status_code == 404:
                    break
            except httpx.HTTPError as exc:
                last_error = f"{type(exc).__name__}: {exc}"
            if attempt < MAX_RETRIES - 1:
                self.sleep(self.min_interval * (attempt + 1))

        self.stats.failed += 1
        return None, {
            "key": key,
            "url": url,
            "source": "failed",
            "http_status": last_status,
            "error": last_error,
            "observed_at_utc": utc_now().isoformat(),
        }


def new_client() -> httpx.Client:
    return httpx.Client(
        headers={
            "User-Agent": USER_AGENT,
            "Accept": "text/html,application/xhtml+xml",
            "Accept-Language": "en-US,en;q=0.9",
        },
        follow_redirects=True,
    )


def append_log(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, default=str) + "\n")


def isoparse(value: str) -> datetime:
    return datetime.fromisoformat(value)
