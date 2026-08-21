"""What the salvaged NBA injury-report archive actually contains.

The archive is the evidentiary base for every retrospective availability
feature, so its gaps have to be *stated*, not discovered later. A missing slot
is never interpolated: the salvage pipeline reports the anchor as uncovered and
moves on.

One caveat this module deliberately preserves: the CDN answers 403 both for
"this report was never published" and for "you are being rate limited". A run
that was throttled would look, in the sidecars alone, exactly like a stretch of
non-publication. Days flagged ``suspect_blocked`` below are how that shows up.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any

from nba_prediction_market.availability.nba_official import (
    SLOT_MINUTES,
    ReportSlot,
    slots_for_date,
)

#: Slots published per calendar day (every 30 minutes, around the clock).
SLOTS_PER_DAY: int = 24 * len(SLOT_MINUTES)

#: A day missing at least this fraction of its slots, while its neighbours are
#: complete, is more likely a blocked run than a real publication gap.
SUSPECT_BLOCKED_MISSING_FRACTION: float = 0.9


@dataclass(frozen=True)
class DayCoverage:
    """One calendar day's slot coverage."""

    report_date: date
    archived: int
    expected: int = SLOTS_PER_DAY
    missing_slots: tuple[str, ...] = ()

    @property
    def is_complete(self) -> bool:
        return self.archived == self.expected

    @property
    def is_empty(self) -> bool:
        return self.archived == 0

    @property
    def missing_fraction(self) -> float:
        return 0.0 if not self.expected else (self.expected - self.archived) / self.expected

    def to_dict(self) -> dict[str, Any]:
        return {
            "report_date": self.report_date.isoformat(),
            "archived": self.archived,
            "expected": self.expected,
            "complete": self.is_complete,
            "missing_slots": list(self.missing_slots),
        }


@dataclass(frozen=True)
class GapRun:
    """A maximal run of consecutive calendar days with nothing archived."""

    start: date
    end: date

    @property
    def days(self) -> int:
        return (self.end - self.start).days + 1

    def to_dict(self) -> dict[str, Any]:
        return {"start": self.start.isoformat(), "end": self.end.isoformat(), "days": self.days}


@dataclass(frozen=True)
class ArchiveInventory:
    """Full coverage picture for the archive."""

    earliest_report: date | None
    latest_report: date | None
    total_reports: int
    days_observed: int
    complete_days: int
    partial_days: int
    empty_days: int
    expected_slots: int
    coverage_fraction: float
    gaps: tuple[GapRun, ...] = ()
    partial_day_details: tuple[DayCoverage, ...] = ()
    suspect_blocked_days: tuple[date, ...] = ()
    per_day: tuple[DayCoverage, ...] = field(default=(), repr=False)

    def to_dict(self) -> dict[str, Any]:
        return {
            "earliest_report_date": (
                self.earliest_report.isoformat() if self.earliest_report else None
            ),
            "latest_report_date": (
                self.latest_report.isoformat() if self.latest_report else None
            ),
            "total_reports_archived": self.total_reports,
            "expected_slots_over_span": self.expected_slots,
            "coverage_fraction": round(self.coverage_fraction, 6),
            "days_in_span": self.days_observed,
            "complete_days": self.complete_days,
            "partial_days": self.partial_days,
            "empty_days": self.empty_days,
            "gap_runs": [g.to_dict() for g in self.gaps],
            "partial_days_detail": [d.to_dict() for d in self.partial_day_details],
            "suspect_blocked_days": [d.isoformat() for d in self.suspect_blocked_days],
            "caveat": (
                "A 403 from the CDN means either 'never published' or 'rate limited'. "
                "suspect_blocked_days lists near-empty days sitting between complete "
                "days, which is the signature of throttling rather than non-publication."
            ),
        }


def _daterange(start: date, end: date) -> Iterable[date]:
    current = start
    while current <= end:
        yield current
        current += timedelta(days=1)


def _gap_runs(per_day: Sequence[DayCoverage]) -> list[GapRun]:
    runs: list[GapRun] = []
    start: date | None = None
    previous: date | None = None
    for day in per_day:
        if day.is_empty:
            if start is None:
                start = day.report_date
            previous = day.report_date
        elif start is not None and previous is not None:
            runs.append(GapRun(start, previous))
            start = None
    if start is not None and previous is not None:
        runs.append(GapRun(start, previous))
    return runs


def _suspect_blocked(per_day: Sequence[DayCoverage]) -> list[date]:
    """Near-empty days bracketed by complete days on both sides look like throttling."""
    suspects: list[date] = []
    for index, day in enumerate(per_day):
        if day.missing_fraction < SUSPECT_BLOCKED_MISSING_FRACTION:
            continue
        # Both sides must be complete. A thin day at either edge of the span is
        # the retention boundary running out, not a throttled run, and must not
        # be blamed on rate limiting.
        if index == 0 or index + 1 >= len(per_day):
            continue
        if per_day[index - 1].is_complete and per_day[index + 1].is_complete:
            suspects.append(day.report_date)
    return suspects


def build_inventory(slots: Iterable[ReportSlot]) -> ArchiveInventory:
    """Summarise archived slots into a coverage report.

    The span runs from the earliest to the latest archived report; days inside
    that span with nothing archived are reported as gaps, never as zero-report
    days that simply did not exist.
    """
    by_date: dict[date, set[str]] = {}
    total = 0
    for slot in slots:
        by_date.setdefault(slot.report_date, set()).add(slot.filename)
        total += 1

    if not by_date:
        return ArchiveInventory(
            earliest_report=None, latest_report=None, total_reports=0, days_observed=0,
            complete_days=0, partial_days=0, empty_days=0, expected_slots=0,
            coverage_fraction=0.0,
        )

    earliest, latest = min(by_date), max(by_date)
    per_day: list[DayCoverage] = []
    for day in _daterange(earliest, latest):
        present = by_date.get(day, set())
        missing = tuple(
            sorted(s.filename for s in slots_for_date(day) if s.filename not in present)
        )
        per_day.append(DayCoverage(day, archived=len(present), missing_slots=missing))

    expected = len(per_day) * SLOTS_PER_DAY
    return ArchiveInventory(
        earliest_report=earliest,
        latest_report=latest,
        total_reports=total,
        days_observed=len(per_day),
        complete_days=sum(1 for d in per_day if d.is_complete),
        partial_days=sum(1 for d in per_day if not d.is_complete and not d.is_empty),
        empty_days=sum(1 for d in per_day if d.is_empty),
        expected_slots=expected,
        coverage_fraction=total / expected if expected else 0.0,
        gaps=tuple(_gap_runs(per_day)),
        partial_day_details=tuple(d for d in per_day if not d.is_complete and not d.is_empty),
        suspect_blocked_days=tuple(_suspect_blocked(per_day)),
        per_day=tuple(per_day),
    )
