"""Resumable on-disk cache for raw BALLDONTLIE season pulls.

One JSON file per season::

    data/raw/nba/seasons/<season>.json

A season is cached only once it has been fetched *completely*, and the file is
written atomically (temp file + rename), so an interrupted run can never leave a
partial season that a later run would mistake for a finished one. Resume
granularity is therefore one season: completed seasons are never refetched, and
only the season in flight when a run stops is repeated.

A cached entry is reused only when its recorded request matches the one being
made. ``--refresh`` bypasses the cache entirely.
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
class SeasonRequest:
    """The exact request a cached season payload must correspond to."""

    season: int
    per_page: int
    endpoint: str = "/games"

    def to_dict(self) -> dict[str, Any]:
        return {"season": self.season, "per_page": self.per_page, "endpoint": self.endpoint}


@dataclass
class SeasonCacheStats:
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


class SeasonCache:
    """Reads and writes complete raw season pulls, keyed by request."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.stats = SeasonCacheStats()

    def path_for(self, season: int) -> Path:
        return self.root / f"season_{int(season)}.json"

    def load(self, request: SeasonRequest) -> list[dict[str, Any]] | None:
        """Return cached pages when they match ``request``, else ``None``."""
        path = self.path_for(request.season)
        if not path.is_file():
            self.stats.misses += 1
            return None
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            logger.warning("Discarding unreadable season cache file %s", path)
            self.stats.corrupt += 1
            return None
        if document.get("request") != request.to_dict():
            self.stats.stale += 1
            return None
        pages = document.get("pages")
        if not isinstance(pages, list) or not document.get("complete"):
            self.stats.corrupt += 1
            return None
        self.stats.hits += 1
        return pages

    def store(self, request: SeasonRequest, pages: list[dict[str, Any]]) -> Path:
        """Persist a *complete* season pull verbatim, atomically."""
        path = self.path_for(request.season)
        path.parent.mkdir(parents=True, exist_ok=True)
        record_count = sum(len(p.get("data") or []) for p in pages)
        document = {
            "request": request.to_dict(),
            "fetched_at_utc": utc_now().isoformat(),
            "complete": True,
            "page_count": len(pages),
            "record_count": record_count,
            "pages": pages,
        }
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(document, indent=2, sort_keys=True), encoding="utf-8")
        tmp.replace(path)
        self.stats.writes += 1
        logger.info(
            "Cached season %s: %d games across %d pages -> %s",
            request.season, record_count, len(pages), path,
        )
        return path

    def cached_seasons(self) -> list[int]:
        """Seasons with a cache file present, ascending."""
        seasons = []
        for path in self.root.glob("season_*.json"):
            try:
                seasons.append(int(path.stem.split("_")[1]))
            except (IndexError, ValueError):
                continue
        return sorted(seasons)
