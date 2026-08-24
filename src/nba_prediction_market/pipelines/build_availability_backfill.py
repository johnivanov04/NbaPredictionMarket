"""Archive historical official NBA injury reports.

Phase 3A3B0 concluded that historical availability was unrecoverable. That was
wrong, and the reason is worth stating plainly: the probe used the 2025-26
filename convention against older dates, got 403, and read the resulting
boundary as a retention limit. It was a *naming* change. Under the legacy name
the CDN still serves reports back to 2018-12-17.

This pipeline walks a date range, fetches every candidate slot in whichever
convention applies, and stores the PDFs in the same append-only archive the
prospective capture writes to, so salvaged and captured reports are
indistinguishable downstream except by provenance.

Safety properties, all of them learned the hard way:

* **Sequential and paced.** Concurrency gets the CDN to answer 403, which is
  the same thing it says for "never published".
* **Canary-guarded.** A run that sees 403s re-checks a URL known to exist. If
  that fails too, the run was throttled and its verdicts are untrustworthy, so
  it stops rather than recording real reports as missing.
* **Restart-safe.** An already-archived slot is never refetched.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import httpx

from nba_prediction_market.availability.nba_official import (
    BASE_URL,
    EARLIEST_AVAILABLE_REPORT_DATE,
    NOT_AVAILABLE_STATUS,
    USER_AGENT,
    ReportArchive,
    ReportSlot,
    slots_for_date,
)
from nba_prediction_market.config import ConfigError, Settings, load_settings
from nba_prediction_market.ingestion.raw_store import utc_now

logger = logging.getLogger(__name__)

#: Floor on the interval between requests.
MIN_REQUEST_INTERVAL_SECONDS: float = 0.35
#: A slot known to exist, used to tell "not published" from "blocked".
CANARY_FILENAME: str = "Injury-Report_2026-04-10_04_00PM.pdf"
#: Consecutive 403s that trigger a canary check.
CANARY_AFTER_CONSECUTIVE_MISSES: int = 12
REQUEST_TIMEOUT_SECONDS: float = 30.0


@dataclass
class BackfillStats:
    slots_checked: int = 0
    archived: int = 0
    already_present: int = 0
    unavailable: int = 0
    errors: int = 0
    canary_checks: int = 0
    days_completed: int = 0
    conflicts: int = 0
    error_examples: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "slots_checked": self.slots_checked,
            "reports_archived": self.archived,
            "already_present": self.already_present,
            "slots_unavailable": self.unavailable,
            "errors": self.errors,
            "hash_conflicts": self.conflicts,
            "canary_checks": self.canary_checks,
            "days_completed": self.days_completed,
            "error_examples": self.error_examples[:10],
        }


class ThrottleDetected(RuntimeError):
    """Raised when the canary fails, meaning 403s cannot be trusted."""


class BackfillRunner:
    """Walks dates, archiving every candidate slot exactly once."""

    def __init__(
        self,
        archive: ReportArchive,
        client: httpx.Client,
        *,
        min_interval: float = MIN_REQUEST_INTERVAL_SECONDS,
        sleep: Any = time.sleep,
        monotonic: Any = time.monotonic,
    ) -> None:
        self.archive = archive
        self._client = client
        self._min_interval = min_interval
        self._sleep = sleep
        self._monotonic = monotonic
        self._last: float | None = None
        self._consecutive_misses = 0
        self.stats = BackfillStats()

    def _paced_get(self, url: str) -> httpx.Response:
        if self._last is not None:
            waited = self._monotonic() - self._last
            if waited < self._min_interval:
                self._sleep(self._min_interval - waited)
        try:
            return self._client.get(url, timeout=REQUEST_TIMEOUT_SECONDS)
        finally:
            self._last = self._monotonic()

    def canary_reachable(self) -> bool:
        self.stats.canary_checks += 1
        try:
            return self._paced_get(f"{BASE_URL}/{CANARY_FILENAME}").status_code == 200
        except httpx.HTTPError:
            return False

    def fetch_slot(self, slot: ReportSlot, *, now: datetime | None = None) -> str:
        """Fetch and archive one slot. Returns its outcome."""
        if self.archive.has(slot):
            self.stats.already_present += 1
            return "already_present"

        self.stats.slots_checked += 1
        try:
            response = self._paced_get(slot.url)
        except httpx.HTTPError as exc:
            self.stats.errors += 1
            self.stats.error_examples.append(f"{slot.filename}: {type(exc).__name__}")
            return "error"

        if response.status_code == NOT_AVAILABLE_STATUS:
            self._consecutive_misses += 1
            self.stats.unavailable += 1
            # 403 is ambiguous. A long run of them is the signature of a block,
            # so confirm against a URL that must answer before trusting any of
            # them as evidence that the reports do not exist.
            if self._consecutive_misses >= CANARY_AFTER_CONSECUTIVE_MISSES:
                if not self.canary_reachable():
                    raise ThrottleDetected(
                        f"canary unreachable after {self._consecutive_misses} "
                        "consecutive 403s; run was throttled and its "
                        "'unavailable' verdicts are not evidence"
                    )
                self._consecutive_misses = 0
            self.archive.unavailable_row(
                slot, response.status_code, now or utc_now()
            )
            return "unavailable"

        if response.status_code != 200 or not response.content.startswith(b"%PDF"):
            self.stats.errors += 1
            self.stats.error_examples.append(
                f"{slot.filename}: status={response.status_code}"
            )
            return "error"

        self._consecutive_misses = 0
        before = self.archive.stats.hash_conflicts
        self.archive.store(
            slot,
            response.content,
            http_status=response.status_code,
            headers=dict(response.headers),
            retrieved_at_utc=now or utc_now(),
        )
        self.stats.conflicts += self.archive.stats.hash_conflicts - before
        self.stats.archived += 1
        return "archived"

    def run_range(self, start: date, end: date) -> BackfillStats:
        if not self.canary_reachable():
            raise ThrottleDetected("canary unreachable before the run started")
        current = start
        while current <= end:
            for slot in slots_for_date(current):
                self.fetch_slot(slot)
            self.stats.days_completed += 1
            logger.info(
                "%s done | archived=%d unavailable=%d present=%d",
                current, self.stats.archived, self.stats.unavailable,
                self.stats.already_present,
            )
            current += timedelta(days=1)
        if not self.canary_reachable():
            raise ThrottleDetected("canary unreachable after the run finished")
        return self.stats


def run_pipeline(
    *, start: date, end: date, settings: Settings | None = None
) -> dict[str, Any]:
    settings = settings or load_settings()
    settings.paths.ensure()
    if start < EARLIEST_AVAILABLE_REPORT_DATE:
        raise ConfigError(
            f"{start} precedes the earliest served report "
            f"({EARLIEST_AVAILABLE_REPORT_DATE})."
        )
    if end < start:
        raise ConfigError("end must not precede start")

    archive = ReportArchive(settings.paths.root / "raw" / "availability" / "nba_official")
    started = utc_now()
    with httpx.Client(
        follow_redirects=True, headers={"User-Agent": USER_AGENT}
    ) as client:
        runner = BackfillRunner(archive, client)
        throttled: str | None = None
        try:
            runner.run_range(start, end)
        except ThrottleDetected as exc:
            throttled = str(exc)

    report = {
        "generated_at_utc": utc_now().isoformat(),
        "started_at_utc": started.isoformat(),
        "range": {"start": start.isoformat(), "end": end.isoformat()},
        "stats": runner.stats.to_dict(),
        "throttled": throttled,
        "trustworthy": throttled is None,
    }
    path = settings.paths.reports / "availability_backfill_run.json"
    existing = json.loads(path.read_text()) if path.is_file() else []
    runs = existing if isinstance(existing, list) else [existing]
    runs.append(report)
    path.write_text(json.dumps(runs, indent=2, default=str), encoding="utf-8")
    report["written_files"] = [str(path)]
    return report


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Backfill historical injury reports.")
    parser.add_argument("--start", required=True, help="First report date (YYYY-MM-DD).")
    parser.add_argument("--end", required=True, help="Last report date (YYYY-MM-DD).")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)-7s | %(message)s"
    )
    try:
        report = run_pipeline(
            start=date.fromisoformat(args.start), end=date.fromisoformat(args.end)
        )
    except ConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    print(f"Stats: {report['stats']}")
    if report["throttled"]:
        print(f"\nSTOPPED: {report['throttled']}", file=sys.stderr)
        return 3
    for path in report["written_files"]:
        print(f"Wrote {Path(path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
