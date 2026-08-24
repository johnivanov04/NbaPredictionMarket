"""Official NBA injury-report archive: URL construction and immutable capture.

The league has published under **two filename conventions**, and telling them
apart is what makes the historical archive reachable at all::

    modern  Injury-Report_{YYYY-MM-DD}_{hh}_{mm}{AM|PM}.pdf   every 30 minutes
    legacy  Injury-Report_{YYYY-MM-DD}_{hh}{AM|PM}.pdf        hourly, at :30

The cutover is sharp and was measured, not assumed: on 2025-12-22 the legacy
name still serves the 08:30 ET report while the modern name first serves the
09:00 ET one. Requests in the wrong convention return 403.

**This corrects an earlier misreading.** Phase 3A3B0 probed 2025-26-era names
against older dates, got 403, and concluded the CDN retained only ~8 months.
That boundary was the convention change, not a retention limit: reports remain
fetchable back to 2018-12-17 under the legacy name.

Two properties of the legacy convention matter for correctness:

* the filename carries only the hour, but the report inside is stamped at
  **:30 past** it -- ``05PM`` is the 5:30 PM report, so reading the filename as
  5:00 would understate the timestamp by half an hour and risk selecting a
  report published *after* an anchor;
* the publication cadence itself changed. Three slots a day (01PM/05PM/08PM)
  through 2020-21, hourly from 2021-22, half-hourly from the cutover.

The filename time is **Eastern**, and it is authoritative: fetching the 6:30
report at 6:59 does not make its contents a 6:59 observation. Both timestamps
are preserved -- ``report_timestamp`` from the slot and ``retrieved_at_utc``
from us. The parser additionally reads the timestamp printed inside the PDF,
which is the authority if the two ever disagree.

A missing report returns **403**, not 404, so 403 means "not available" rather
than a transient server fault -- and, unhelpfully, it is also what a throttled
client sees, which is why callers pair it with a canary.
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
#: Minute past the hour at which every legacy-convention report is stamped.
LEGACY_SLOT_MINUTE: Final = 30
USER_AGENT: Final = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
)
#: HTTP status the CDN returns for a report that does not exist.
NOT_AVAILABLE_STATUS: Final = 403
#: First report published under the modern half-hourly filename. Measured: the
#: 08:30 ET report that day is legacy-named, the 09:00 ET one is modern-named.
CONVENTION_CUTOVER_ET: Final = datetime(2025, 12, 22, 9, 0, tzinfo=EASTERN)
#: Earliest report the CDN still serves, found by bisection.
EARLIEST_AVAILABLE_REPORT_DATE: Final = date(2018, 12, 17)


@dataclass(frozen=True)
class ReportSlot:
    """One publication slot, identified by its Eastern wall-clock time."""

    report_date: date
    hour_12: int
    minute: int
    meridiem: str

    @property
    def is_legacy(self) -> bool:
        """True when this slot predates the filename cutover."""
        return self.report_timestamp_et < CONVENTION_CUTOVER_ET

    @property
    def filename(self) -> str:
        stem = f"Injury-Report_{self.report_date.isoformat()}_{self.hour_12:02d}"
        if self.is_legacy:
            return f"{stem}{self.meridiem}.pdf"
        return f"{stem}_{self.minute:02d}{self.meridiem}.pdf"

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


def _slot(report_date: date, hour_24: int, minute: int) -> ReportSlot:
    return ReportSlot(
        report_date,
        hour_24 % 12 or 12,
        minute,
        "AM" if hour_24 < 12 else "PM",
    )


def slots_for_date(report_date: date) -> list[ReportSlot]:
    """Every candidate slot for one calendar date, in chronological order.

    Which grid applies depends on where the date sits relative to the filename
    cutover: hourly (at :30) before it, half-hourly after. The cutover day
    itself carries both, split at the cutover instant, which is why it has 39
    candidates rather than 24 or 48.

    These are *candidates*. Cadence varied by era -- only three slots a day were
    published through 2020-21 -- so an absent slot answers 403 and is recorded
    as unavailable rather than treated as an error.
    """
    out: list[ReportSlot] = []
    for hour_24 in range(24):
        legacy = _slot(report_date, hour_24, LEGACY_SLOT_MINUTE)
        if legacy.is_legacy:
            out.append(legacy)
            continue
        for minute in SLOT_MINUTES:
            candidate = _slot(report_date, hour_24, minute)
            if not candidate.is_legacy:
                out.append(candidate)
    return sorted(out, key=lambda s: (s.hour_24, s.minute))


def slot_from_filename(filename: str) -> ReportSlot | None:
    """Parse a slot back out of a filename, or ``None`` if it does not match."""
    stem = Path(filename).stem
    if not stem.startswith("Injury-Report_"):
        return None
    try:
        _, day, clock = stem.split("_", 2)
        if "_" in clock:
            hour_text, rest = clock.split("_", 1)
            minute_text, meridiem = rest[:2], rest[2:]
            minute = int(minute_text)
        else:
            # Legacy name carries only the hour; the report is stamped at :30.
            hour_text, meridiem = clock[:2], clock[2:]
            minute = LEGACY_SLOT_MINUTE
        if meridiem not in ("AM", "PM"):
            return None
        return ReportSlot(date.fromisoformat(day), int(hour_text), minute, meridiem)
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
    candidate = eastern - timedelta(minutes=eastern.minute % 30)
    slot = _slot(candidate.date(), candidate.hour, candidate.minute)
    if not slot.is_legacy:
        return slot
    # Legacy reports exist only at :30, so an anchor in the first half of an
    # hour resolves to the previous hour's report, crossing midnight if needed.
    if eastern.minute < LEGACY_SLOT_MINUTE:
        eastern -= timedelta(hours=1)
    return _slot(eastern.date(), eastern.hour, LEGACY_SLOT_MINUTE)


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
