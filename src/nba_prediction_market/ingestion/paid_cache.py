"""Resumable cache for paid BALLDONTLIE feeds.

One file per (feed, season)::

    data/raw/balldontlie/<feed>/season_<year>.json

A season is written only once it has been fetched *completely*, atomically via a
temp file and rename, so an interrupted run can never leave a partial season that
a later run mistakes for a finished one. Resume granularity is one season.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from nba_prediction_market.ingestion.raw_store import utc_now

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PaidRequest:
    """The exact request a cached payload must correspond to."""

    feed: str
    endpoint: str
    season: int
    per_page: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "feed": self.feed,
            "endpoint": self.endpoint,
            "season": self.season,
            "per_page": self.per_page,
        }


@dataclass
class PaidCacheStats:
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


class PaidCache:
    """Reads and writes complete per-season pulls for one paid feed."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.stats = PaidCacheStats()

    def path_for(self, feed: str, season: int) -> Path:
        if "/" in feed or "\\" in feed:
            raise ValueError(f"feed name {feed!r} is not filename-safe")
        return self.root / feed / f"season_{int(season)}.json"

    def load(self, request: PaidRequest) -> list[dict[str, Any]] | None:
        """Cached records when they match ``request``, else ``None``."""
        path = self.path_for(request.feed, request.season)
        if not path.is_file():
            self.stats.misses += 1
            return None
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            logger.warning("Discarding unreadable paid cache file %s", path)
            self.stats.corrupt += 1
            return None
        if document.get("request") != request.to_dict():
            self.stats.stale += 1
            return None
        if not document.get("complete"):
            self.stats.corrupt += 1
            return None
        records = document.get("records")
        if not isinstance(records, list):
            self.stats.corrupt += 1
            return None
        self.stats.hits += 1
        return records

    def store(self, request: PaidRequest, records: list[dict[str, Any]]) -> Path:
        """Persist a complete season pull verbatim, atomically."""
        path = self.path_for(request.feed, request.season)
        path.parent.mkdir(parents=True, exist_ok=True)
        document = {
            "request": request.to_dict(),
            "fetched_at_utc": utc_now().isoformat(),
            "complete": True,
            "record_count": len(records),
            "records": records,
        }
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(document, sort_keys=True), encoding="utf-8")
        tmp.replace(path)
        self.stats.writes += 1
        logger.info(
            "Cached %s season %s: %d records -> %s",
            request.feed, request.season, len(records), path,
        )
        return path

    def cached_seasons(self, feed: str) -> list[int]:
        directory = self.root / feed
        if not directory.is_dir():
            return []
        seasons = []
        for path in directory.glob("season_*.json"):
            try:
                seasons.append(int(path.stem.split("_")[1]))
            except (IndexError, ValueError):
                continue
        return sorted(seasons)
