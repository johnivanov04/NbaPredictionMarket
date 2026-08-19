"""Verbatim persistence of API responses.

Raw payloads are written before any parsing so every processed table can be
rebuilt (or a parsing bug re-diagnosed) without re-hitting the APIs.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def utc_now() -> datetime:
    """Current time, UTC and timezone-aware."""
    return datetime.now(UTC)


def utc_stamp(moment: datetime | None = None) -> str:
    """Compact UTC timestamp usable in filenames (``20260819T211354Z``)."""
    return (moment or utc_now()).strftime("%Y%m%dT%H%M%SZ")


@dataclass
class RawSnapshot:
    """Provenance for one persisted raw pull."""

    name: str
    path: Path
    page_count: int
    record_count: int
    fetched_at_utc: str
    params: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "path": str(self.path),
            "page_count": self.page_count,
            "record_count": self.record_count,
            "fetched_at_utc": self.fetched_at_utc,
            "params": self.params,
        }


class RawStore:
    """Writes raw JSON documents under a directory, one file per pull."""

    def __init__(self, directory: Path, *, run_stamp: str | None = None) -> None:
        self.directory = Path(directory)
        self.run_stamp = run_stamp or utc_stamp()
        self.directory.mkdir(parents=True, exist_ok=True)

    def write(
        self,
        name: str,
        pages: list[dict[str, Any]],
        *,
        record_count: int,
        params: dict[str, Any] | None = None,
    ) -> RawSnapshot:
        """Persist every page of a paginated pull, plus its request context."""
        fetched_at = utc_now().isoformat()
        path = self.directory / f"{name}_{self.run_stamp}.json"
        document = {
            "name": name,
            "fetched_at_utc": fetched_at,
            "params": params or {},
            "page_count": len(pages),
            "record_count": record_count,
            "pages": pages,
        }
        path.write_text(json.dumps(document, indent=2, sort_keys=True), encoding="utf-8")
        logger.info("Wrote raw %s: %d records across %d pages -> %s",
                    name, record_count, len(pages), path)
        return RawSnapshot(
            name=name,
            path=path,
            page_count=len(pages),
            record_count=record_count,
            fetched_at_utc=fetched_at,
            params=params or {},
        )

    def latest(self, name: str) -> Path | None:
        """Most recent raw file for ``name``, or ``None`` if none exist."""
        candidates = sorted(self.directory.glob(f"{name}_*.json"))
        return candidates[-1] if candidates else None

    @staticmethod
    def load(path: Path) -> dict[str, Any]:
        """Read a previously written raw document."""
        return json.loads(Path(path).read_text(encoding="utf-8"))

    @staticmethod
    def records(document: dict[str, Any], items_key: str) -> list[dict[str, Any]]:
        """Flatten a raw document's pages back into a record list."""
        out: list[dict[str, Any]] = []
        for page in document.get("pages", []):
            out.extend(page.get(items_key, []))
        return out
