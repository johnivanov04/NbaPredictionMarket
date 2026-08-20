"""Resumable on-disk cache for raw candlestick responses.

One JSON file per market, sharded by game date so no directory (and no single
file) becomes unmanageable::

    data/raw/kalshi/candlesticks/<config slug>/<game date>/<market ticker>.json

The config slug encodes the request geometry (offset from tipoff, lookback,
period interval), so changing any of those writes to a *different* tree instead
of silently reusing candles fetched for a different window. Within a tree, a
cached entry is only reused when its recorded request matches the one being made
byte for byte -- a cache that can return the wrong window is worse than none.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

from nba_prediction_market.ingestion.raw_store import utc_now

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CandleRequest:
    """The exact request a cached payload must correspond to."""

    market_ticker: str
    start_ts: int
    end_ts: int
    period_interval: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "market_ticker": self.market_ticker,
            "start_ts": self.start_ts,
            "end_ts": self.end_ts,
            "period_interval": self.period_interval,
        }


def cache_slug(
    *, minutes_before_tip: int, lookback_minutes: int, period_interval: int
) -> str:
    """Directory name encoding the request geometry (``t30_lb60_p1``)."""
    return f"t{minutes_before_tip}_lb{lookback_minutes}_p{period_interval}"


@dataclass
class CacheStats:
    """Counts for the run summary."""

    hits: int = 0
    misses: int = 0
    stale: int = 0
    corrupt: int = 0
    writes: int = 0

    def to_dict(self) -> dict[str, int]:
        return {
            "hits": self.hits,
            "misses": self.misses,
            "invalidated_request_changed": self.stale,
            "invalidated_corrupt": self.corrupt,
            "writes": self.writes,
        }


class CandleCache:
    """Reads and writes raw candlestick payloads, keyed by request."""

    def __init__(self, root: Path, slug: str) -> None:
        self.root = Path(root) / slug
        self.slug = slug
        self.stats = CacheStats()

    def path_for(self, market_ticker: str, game_date: date | str) -> Path:
        shard = game_date.isoformat() if isinstance(game_date, date) else str(game_date)
        # Tickers are already filename-safe (A-Z, digits, dashes); assert rather
        # than sanitise, so an unexpected shape surfaces instead of colliding.
        if "/" in market_ticker or "\\" in market_ticker:
            raise ValueError(f"market ticker {market_ticker!r} is not filename-safe")
        return self.root / shard / f"{market_ticker}.json"

    def load(self, request: CandleRequest, game_date: date | str) -> dict[str, Any] | None:
        """Return the cached payload when it matches ``request``, else ``None``."""
        path = self.path_for(request.market_ticker, game_date)
        if not path.is_file():
            self.stats.misses += 1
            return None
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            logger.warning("Discarding unreadable candle cache file %s", path)
            self.stats.corrupt += 1
            return None
        if document.get("request") != request.to_dict():
            self.stats.stale += 1
            return None
        payload = document.get("response")
        if not isinstance(payload, dict):
            self.stats.corrupt += 1
            return None
        self.stats.hits += 1
        return payload

    def store(
        self,
        request: CandleRequest,
        game_date: date | str,
        payload: dict[str, Any],
        *,
        endpoint: str,
    ) -> Path:
        """Persist a raw payload verbatim, alongside the request that produced it."""
        path = self.path_for(request.market_ticker, game_date)
        path.parent.mkdir(parents=True, exist_ok=True)
        document = {
            "request": request.to_dict(),
            "endpoint": endpoint,
            "fetched_at_utc": utc_now().isoformat(),
            "candle_count": len(payload.get("candlesticks") or []),
            "response": payload,
        }
        # Write via a temp file so an interrupted run cannot leave a half-written
        # file that a later run would treat as a cache hit.
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(document, indent=2, sort_keys=True), encoding="utf-8")
        tmp.replace(path)
        self.stats.writes += 1
        return path
