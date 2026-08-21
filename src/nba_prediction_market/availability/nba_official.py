"""Official NBA injury-report archive: URL construction and immutable capture.

The league publishes a PDF every 30 minutes, around the clock, at::

    https://ak-static.cms.nba.com/referee/injury/Injury-Report_{YYYY-MM-DD}_{hh}_{mm}{AM|PM}.pdf

The filename time is **Eastern**, and it is authoritative: fetching the 6:30
report at 6:59 does not make its contents a 6:59 observation. Both timestamps
are therefore preserved -- ``report_timestamp`` from the filename and
``retrieved_at_utc`` from us.

A missing report returns **403**, not 404 (verified by requesting an invalid
minute on a valid date), so 403 means "not available" rather than a transient
server fault.

The CDN retains only a rolling window of roughly eight months, which is why
capture is urgent: older reports are being deleted as time passes.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any, Final
from zoneinfo import ZoneInfo

logger = logging.getLogger(__name__)

BASE_URL: Final = "https://ak-static.cms.nba.com/referee/injury"
#: The league publishes on this grid, in minutes past the hour.
SLOT_MINUTES: Final[tuple[int, ...]] = (0, 30)
EASTERN: Final = ZoneInfo("America/New_York")
USER_AGENT: Final = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
)
#: HTTP status the CDN returns for a report that does not exist.
NOT_AVAILABLE_STATUS: Final = 403


@dataclass(frozen=True)
class ReportSlot:
    """One publication slot, identified by its Eastern wall-clock time."""

    report_date: date
    hour_12: int
    minute: int
    meridiem: str

    @property
    def filename(self) -> str:
        return (
            f"Injury-Report_{self.report_date.isoformat()}_"
            f"{self.hour_12:02d}_{self.minute:02d}{self.meridiem}.pdf"
        )

    @property
    def url(self) -> str:
        return f"{BASE_URL}/{self.filename}"

    @property
    def hour_24(self) -> int:
        if self.meridiem == "AM":
            return 0 if self.hour_12 == 12 else self.hour_12
        return 12 if self.hour_12 == 12 else self.hour_12 + 12

    @property
    def report_timestamp_et(self) -> datetime:
        """The slot's Eastern wall-clock instant."""
        return datetime(
            self.report_date.year, self.report_date.month, self.report_date.day,
            self.hour_24, self.minute, tzinfo=EASTERN,
        )

    @property
    def report_timestamp_utc(self) -> datetime:
        return self.report_timestamp_et.astimezone(UTC)

    def to_dict(self) -> dict[str, Any]:
        return {
            "report_date": self.report_date.isoformat(),
            "report_slot": f"{self.hour_12:02d}:{self.minute:02d}{self.meridiem}",
            "report_timestamp_et": self.report_timestamp_et.isoformat(),
            "report_timestamp_utc": self.report_timestamp_utc.isoformat(),
            "filename": self.filename,
            "url": self.url,
        }


def slots_for_date(report_date: date) -> list[ReportSlot]:
    """All 48 half-hourly slots for one calendar date, in chronological order."""
    out: list[ReportSlot] = []
    for hour_24 in range(24):
        meridiem = "AM" if hour_24 < 12 else "PM"
        hour_12 = hour_24 % 12 or 12
        for minute in SLOT_MINUTES:
            out.append(ReportSlot(report_date, hour_12, minute, meridiem))
    return sorted(out, key=lambda s: (s.hour_24, s.minute))


def slot_from_filename(filename: str) -> ReportSlot | None:
    """Parse a slot back out of a filename, or ``None`` if it does not match."""
    stem = Path(filename).stem
    if not stem.startswith("Injury-Report_"):
        return None
    try:
        _, day, clock = stem.split("_", 2)
        hour_text, rest = clock.split("_", 1)
        minute_text, meridiem = rest[:2], rest[2:]
        return ReportSlot(
            date.fromisoformat(day), int(hour_text), int(minute_text), meridiem
        )
    except (ValueError, IndexError):
        return None


def latest_slot_at_or_before(anchor: datetime) -> ReportSlot:
    """The newest publication slot at or before ``anchor``.

    The anchor is converted to Eastern first, because the grid is defined in
    Eastern wall-clock time and shifts with daylight saving.
    """
    if anchor.tzinfo is None:
        raise ValueError("anchor must be timezone-aware")
    eastern = anchor.astimezone(EASTERN).replace(second=0, microsecond=0)
    eastern -= timedelta(minutes=eastern.minute % 30)
    hour_12 = eastern.hour % 12 or 12
    meridiem = "AM" if eastern.hour < 12 else "PM"
    return ReportSlot(eastern.date(), hour_12, eastern.minute, meridiem)


# --- immutable archive -----------------------------------------------------


@dataclass
class ArchiveStats:
    checked: int = 0
    archived: int = 0
    unavailable: int = 0
    already_present: int = 0
    hash_conflicts: int = 0
    errors: int = 0

    def to_dict(self) -> dict[str, int]:
        return {
            "slots_checked": self.checked,
            "reports_archived": self.archived,
            "slots_unavailable": self.unavailable,
            "already_present": self.already_present,
            "hash_conflicts": self.hash_conflicts,
            "errors": self.errors,
        }


class ReportArchive:
    """Immutable on-disk archive of official injury-report PDFs.

    Layout ``<root>/YYYY/MM/DD/<filename>`` with a JSON sidecar carrying
    provenance. An archived report is never overwritten: an identical re-download
    is deduplicated, and a *differing* download for the same identifier is stored
    beside the original and flagged rather than replacing it.
    """

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.stats = ArchiveStats()

    def directory_for(self, slot: ReportSlot) -> Path:
        d = slot.report_date
        return self.root / f"{d.year:04d}" / f"{d.month:02d}" / f"{d.day:02d}"

    def pdf_path(self, slot: ReportSlot) -> Path:
        return self.directory_for(slot) / slot.filename

    def sidecar_path(self, slot: ReportSlot) -> Path:
        return self.directory_for(slot) / f"{slot.filename}.meta.json"

    def has(self, slot: ReportSlot) -> bool:
        return self.pdf_path(slot).is_file() and self.sidecar_path(slot).is_file()

    def store(
        self,
        slot: ReportSlot,
        content: bytes,
        *,
        http_status: int,
        headers: dict[str, str],
        retrieved_at_utc: datetime,
    ) -> dict[str, Any]:
        """Archive one report immutably, returning its inventory row."""
        digest = hashlib.sha256(content).hexdigest()
        path = self.pdf_path(slot)
        path.parent.mkdir(parents=True, exist_ok=True)

        if path.is_file():
            existing = hashlib.sha256(path.read_bytes()).hexdigest()
            if existing == digest:
                self.stats.already_present += 1
                return self._row(slot, digest, http_status, headers, retrieved_at_utc, path)
            # Same identifier, different bytes: keep both and flag it.
            self.stats.hash_conflicts += 1
            path = path.with_name(f"{slot.filename}.conflict-{digest[:12]}.pdf")
            logger.warning(
                "Hash conflict for %s: archived %s, new %s -- preserving both",
                slot.filename, existing[:12], digest[:12],
            )

        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_bytes(content)
        tmp.replace(path)
        row = self._row(slot, digest, http_status, headers, retrieved_at_utc, path)
        self.sidecar_path(slot).write_text(json.dumps(row, indent=2), encoding="utf-8")
        self.stats.archived += 1
        return row

    def _row(
        self, slot: ReportSlot, digest: str, http_status: int,
        headers: dict[str, str], retrieved_at_utc: datetime, path: Path,
    ) -> dict[str, Any]:
        return {
            **slot.to_dict(),
            "available": True,
            "http_status": http_status,
            "last_modified": headers.get("last-modified"),
            "content_length": headers.get("content-length"),
            "etag": headers.get("etag"),
            "retrieved_at_utc": retrieved_at_utc.astimezone(UTC).isoformat(),
            "sha256": digest,
            "local_path": str(path),
        }

    def unavailable_row(
        self, slot: ReportSlot, http_status: int, retrieved_at_utc: datetime
    ) -> dict[str, Any]:
        """An inventory row for a slot the CDN does not hold.

        A missing report means the artefact is unavailable -- never that there
        were no injuries that day.
        """
        self.stats.unavailable += 1
        return {
            **slot.to_dict(),
            "available": False,
            "http_status": http_status,
            "last_modified": None,
            "content_length": None,
            "etag": None,
            "retrieved_at_utc": retrieved_at_utc.astimezone(UTC).isoformat(),
            "sha256": None,
            "local_path": None,
        }

    def archived_slots(self) -> list[ReportSlot]:
        """Every slot present on disk, chronological."""
        if not self.root.is_dir():
            return []
        slots = []
        for path in self.root.rglob("Injury-Report_*.pdf"):
            if ".conflict-" in path.name:
                continue
            slot = slot_from_filename(path.name)
            if slot is not None:
                slots.append(slot)
        return sorted(slots, key=lambda s: s.report_timestamp_utc)
