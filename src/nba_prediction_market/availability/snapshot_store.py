"""Append-only, immutable raw availability snapshots.

Every capture is written once and never modified. A newer state is a *new*
snapshot beside the old one, never an overwrite -- otherwise the record of what
was known at an earlier time is destroyed, which is precisely the information
this whole effort exists to preserve.

Layout::

    data/raw/availability/<source>/<YYYY-MM-DD>/<observed_at>__<slug>.json

The observed timestamp is in the filename so the archive is inspectable and
sortable without opening files. Writes are atomic (temp file + rename), and a
re-capture of an identical payload at the same instant is a no-op rather than a
duplicate.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_SLUG = re.compile(r"[^A-Za-z0-9_.-]+")


def _slugify(value: str) -> str:
    return _SLUG.sub("-", value).strip("-")[:80] or "snapshot"


@dataclass
class SnapshotStats:
    written: int = 0
    duplicates: int = 0

    def to_dict(self) -> dict[str, int]:
        return {"written": self.written, "duplicate_skipped": self.duplicates}


@dataclass(frozen=True)
class Snapshot:
    """One immutable raw capture."""

    source: str
    retrieved_at_utc: datetime
    request: str
    payload: Any
    source_report_timestamp: datetime | None = None
    source_effective_date: str | None = None
    content_type: str = "application/json"
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.retrieved_at_utc.tzinfo is None:
            raise ValueError("retrieved_at_utc must be timezone-aware")

    def to_document(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "retrieved_at_utc": self.retrieved_at_utc.astimezone(UTC).isoformat(),
            "source_report_timestamp": (
                self.source_report_timestamp.astimezone(UTC).isoformat()
                if self.source_report_timestamp else None
            ),
            "source_effective_date": self.source_effective_date,
            "request": self.request,
            "content_type": self.content_type,
            "metadata": self.metadata,
            "payload": self.payload,
        }

    def fingerprint(self) -> str:
        """Content hash, used to skip an identical re-capture."""
        body = json.dumps(
            {"source": self.source, "request": self.request, "payload": self.payload},
            sort_keys=True, default=str,
        )
        return hashlib.sha256(body.encode("utf-8")).hexdigest()[:16]


class SnapshotStore:
    """Append-only store of raw availability captures."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.stats = SnapshotStats()

    def directory_for(self, source: str, moment: datetime) -> Path:
        return self.root / _slugify(source) / moment.astimezone(UTC).strftime("%Y-%m-%d")

    def path_for(self, snapshot: Snapshot) -> Path:
        stamp = snapshot.retrieved_at_utc.astimezone(UTC).strftime("%Y%m%dT%H%M%SZ")
        name = f"{stamp}__{_slugify(snapshot.request)}__{snapshot.fingerprint()}.json"
        return self.directory_for(snapshot.source, snapshot.retrieved_at_utc) / name

    def append(self, snapshot: Snapshot) -> Path:
        """Write a snapshot. Never overwrites an existing capture."""
        path = self.path_for(snapshot)
        if path.exists():
            # Same source, instant, request and content: nothing new was learned.
            self.stats.duplicates += 1
            return path
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(
            json.dumps(snapshot.to_document(), indent=2, sort_keys=True, default=str),
            encoding="utf-8",
        )
        tmp.replace(path)
        self.stats.written += 1
        return path

    def iter_snapshots(
        self, source: str, *, since: datetime | None = None, until: datetime | None = None
    ) -> list[dict[str, Any]]:
        """Every stored snapshot for a source, oldest first.

        ``until`` is what makes replay honest: passing a past instant yields
        exactly the snapshots that existed then.
        """
        directory = self.root / _slugify(source)
        if not directory.is_dir():
            return []
        out: list[dict[str, Any]] = []
        for path in sorted(directory.rglob("*.json")):
            try:
                document = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                logger.warning("Skipping unreadable snapshot %s", path)
                continue
            retrieved = datetime.fromisoformat(document["retrieved_at_utc"])
            if since is not None and retrieved < since:
                continue
            if until is not None and retrieved > until:
                continue
            document["_path"] = str(path)
            out.append(document)
        return sorted(out, key=lambda d: d["retrieved_at_utc"])

    def sources(self) -> list[str]:
        if not self.root.is_dir():
            return []
        return sorted(p.name for p in self.root.iterdir() if p.is_dir())
