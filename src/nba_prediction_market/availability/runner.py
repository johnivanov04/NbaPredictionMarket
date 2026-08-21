"""Prospective availability capture runner.

Executes the Phase 3A3B0 capture plan against live sources and stores every
result immutably. Designed to be restarted at any moment: all state lives on
disk, a completed capture is recognised by its presence in the archive, and a
restart neither loses history nor double-writes.

Two timestamps are always kept apart:

* ``report_timestamp`` -- what the *source* says the state is. Authoritative.
* ``retrieved_at_utc`` -- when *we* fetched it.

Fetching the 6:30 report at 6:59 does not make its contents a 6:59 observation.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Final

from nba_prediction_market.availability.capture_schedule import CaptureTask, plan_captures
from nba_prediction_market.availability.nba_official import (
    NOT_AVAILABLE_STATUS,
    ReportArchive,
    ReportSlot,
    latest_slot_at_or_before,
)
from nba_prediction_market.availability.snapshot_store import Snapshot, SnapshotStore

logger = logging.getLogger(__name__)

#: A capture is considered fresh enough to satisfy its anchor if the source
#: report is no older than this. Reports publish every 30 minutes, so anything
#: beyond one hour means slots were missing.
MAX_ACCEPTABLE_REPORT_AGE_MINUTES: Final = 60.0


@dataclass
class RunnerStats:
    attempted: int = 0
    captured: int = 0
    already_present: int = 0
    unavailable: int = 0
    failed: int = 0

    def to_dict(self) -> dict[str, int]:
        return {
            "attempted": self.attempted,
            "captured": self.captured,
            "already_present": self.already_present,
            "source_unavailable": self.unavailable,
            "failed": self.failed,
        }


@dataclass
class CaptureResult:
    """Outcome of one capture attempt."""

    task: CaptureTask
    outcome: str
    slot: ReportSlot | None = None
    report_timestamp_utc: datetime | None = None
    retrieved_at_utc: datetime | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.task.to_dict(),
            "outcome": self.outcome,
            "report_slot_filename": self.slot.filename if self.slot else None,
            "report_timestamp_utc": (
                self.report_timestamp_utc.isoformat() if self.report_timestamp_utc else None
            ),
            "retrieved_at_utc": (
                self.retrieved_at_utc.isoformat() if self.retrieved_at_utc else None
            ),
            "error": self.error,
        }


class AvailabilityRunner:
    """Executes capture tasks against sources, storing results immutably."""

    def __init__(
        self,
        archive: ReportArchive,
        store: SnapshotStore,
        *,
        fetch: Callable[[str], tuple[int, bytes, dict[str, str]]],
        max_retries: int = 3,
        retry_backoff_seconds: float = 2.0,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.archive = archive
        self.store = store
        self._fetch = fetch
        self.max_retries = max_retries
        self.retry_backoff_seconds = retry_backoff_seconds
        self._sleep = sleep
        self.stats = RunnerStats()

    def _fetch_with_retry(self, url: str) -> tuple[int, bytes, dict[str, str]]:
        last: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                return self._fetch(url)
            except Exception as exc:
                last = exc
                if attempt < self.max_retries - 1:
                    self._sleep(self.retry_backoff_seconds * (attempt + 1))
        raise RuntimeError(f"fetch failed after {self.max_retries} attempts: {last}")

    def run_task(self, task: CaptureTask, *, now: datetime | None = None) -> CaptureResult:
        """Execute one capture. Idempotent: an already-archived slot is skipped."""
        self.stats.attempted += 1
        anchor = task.anchor_utc
        slot = latest_slot_at_or_before(anchor)

        if self.archive.has(slot):
            self.stats.already_present += 1
            return CaptureResult(
                task, "already_present", slot, slot.report_timestamp_utc
            )

        try:
            status, content, headers = self._fetch_with_retry(slot.url)
        except RuntimeError as exc:
            self.stats.failed += 1
            logger.warning("capture failed for %s: %s", slot.filename, exc)
            return CaptureResult(task, "failed", slot, error=str(exc))

        retrieved = (now or datetime.now(UTC)).astimezone(UTC)
        if status == NOT_AVAILABLE_STATUS:
            self.stats.unavailable += 1
            return CaptureResult(task, "source_unavailable", slot, retrieved_at_utc=retrieved)
        if status != 200:
            self.stats.failed += 1
            return CaptureResult(
                task, "failed", slot, retrieved_at_utc=retrieved,
                error=f"unexpected HTTP {status}",
            )

        self.archive.store(
            slot, content, http_status=status, headers=headers, retrieved_at_utc=retrieved
        )
        # A parallel immutable record, so provenance survives independently.
        self.store.append(
            Snapshot(
                source="nba_official",
                retrieved_at_utc=retrieved,
                request=slot.filename,
                payload={"bytes": len(content), "sha256_in_archive": True},
                source_report_timestamp=slot.report_timestamp_utc,
                source_effective_date=slot.report_date.isoformat(),
                content_type="application/pdf",
                metadata={"anchor_utc": anchor.isoformat(), "game_id": task.game_id},
            )
        )
        self.stats.captured += 1
        return CaptureResult(task, "captured", slot, slot.report_timestamp_utc, retrieved)

    def run(
        self, tasks: Sequence[CaptureTask], *, now: datetime | None = None
    ) -> list[CaptureResult]:
        """Execute a plan in chronological order."""
        return [self.run_task(task, now=now) for task in sorted(
            tasks, key=lambda t: t.capture_at_utc
        )]


def plan_for_slate(
    games: list[dict[str, Any]], sources: list[str] | None = None
) -> list[CaptureTask]:
    """Capture plan for one day's games."""
    return plan_captures(games, sources or ["nba_official_injury_report"])


def anchor_health(results: Sequence[CaptureResult]) -> dict[str, Any]:
    """Whether each game's anchor ended up covered by a fresh enough report.

    Run this before relying on a night's capture: a plan that executed without
    errors can still leave an anchor uncovered if the source had no report.
    """
    by_game: dict[Any, list[CaptureResult]] = {}
    for result in results:
        by_game.setdefault(result.task.game_id, []).append(result)

    covered, uncovered, stale = [], [], []
    for game_id, game_results in by_game.items():
        usable = [
            r for r in game_results
            if r.outcome in {"captured", "already_present"} and r.report_timestamp_utc
        ]
        if not usable:
            uncovered.append(game_id)
            continue
        best = max(usable, key=lambda r: r.report_timestamp_utc)
        age = (best.task.anchor_utc - best.report_timestamp_utc).total_seconds() / 60.0
        (stale if age > MAX_ACCEPTABLE_REPORT_AGE_MINUTES else covered).append(game_id)
    return {
        "games": len(by_game),
        "anchors_covered": len(covered),
        "anchors_stale": len(stale),
        "anchors_uncovered": len(uncovered),
        "uncovered_game_ids": uncovered,
        "stale_game_ids": stale,
        "healthy": not uncovered and not stale,
    }
