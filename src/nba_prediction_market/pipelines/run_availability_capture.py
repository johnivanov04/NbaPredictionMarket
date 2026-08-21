"""Run prospective availability captures for an upcoming slate.

This is the forward-looking half of Phase 3A3B1. The salvage pipeline recovers
what the CDN still holds; this one makes sure nothing else is ever lost, by
archiving the official report on a schedule built backwards from each tipoff.

Two operational lessons are wired in as code rather than left to the operator:

* **Sequential, paced requests.** Fetching this CDN concurrently gets the client
  rate-limited, and it answers a throttled request with 403 -- the same status
  it uses for "this report was never published". A parallel run therefore does
  not fail loudly; it quietly records real reports as missing.
* **A canary.** Because 403 is ambiguous, every run that sees one re-checks a
  URL known to exist. If the canary also 403s, the run was blocked and its
  "unavailable" verdicts are untrustworthy, so the report says so instead of
  letting a throttled run masquerade as a quiet news day.

Nothing here decides availability. It stores raw evidence, timestamped.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import httpx
import pandas as pd

from nba_prediction_market.availability.capture_schedule import plan_captures
from nba_prediction_market.availability.nba_official import (
    NOT_AVAILABLE_STATUS,
    ReportArchive,
)
from nba_prediction_market.availability.runner import AvailabilityRunner, anchor_health
from nba_prediction_market.availability.snapshot_store import SnapshotStore
from nba_prediction_market.config import ConfigError, Settings, load_settings
from nba_prediction_market.ingestion.raw_store import utc_now

logger = logging.getLogger(__name__)

#: Sources this pipeline can capture. Only the official report is implemented;
#: the audit found nothing else with usable as-of semantics.
CAPTURE_SOURCES: list[str] = ["nba_official_injury_report"]

#: Minimum seconds between requests. Deliberately conservative: the archive is
#: built once and then trickles, so there is nothing to gain from going faster
#: and a rate-limit block is expensive and silent.
MIN_REQUEST_INTERVAL_SECONDS: float = 0.35

#: A slot known to exist, used to tell "not published" apart from "blocked".
CANARY_FILENAME: str = "Injury-Report_2026-04-10_04_00PM.pdf"

REQUEST_TIMEOUT_SECONDS: float = 30.0


class PacedFetcher:
    """Sequential HTTP GET with a floor on the interval between requests."""

    def __init__(
        self,
        client: httpx.Client,
        *,
        min_interval: float = MIN_REQUEST_INTERVAL_SECONDS,
        sleep: Any = time.sleep,
        monotonic: Any = time.monotonic,
    ) -> None:
        self._client = client
        self._min_interval = min_interval
        self._sleep = sleep
        self._monotonic = monotonic
        self._last: float | None = None

    def __call__(self, url: str) -> tuple[int, bytes, dict[str, str]]:
        if self._last is not None:
            waited = self._monotonic() - self._last
            if waited < self._min_interval:
                self._sleep(self._min_interval - waited)
        response = self._client.get(url, timeout=REQUEST_TIMEOUT_SECONDS)
        self._last = self._monotonic()
        return response.status_code, response.content, dict(response.headers)


def canary_is_reachable(client: httpx.Client, filename: str = CANARY_FILENAME) -> bool:
    """True when a URL known to exist still answers, i.e. we are not blocked."""
    from nba_prediction_market.availability.nba_official import BASE_URL

    try:
        response = client.head(f"{BASE_URL}/{filename}", timeout=REQUEST_TIMEOUT_SECONDS)
    except httpx.HTTPError:
        return False
    return response.status_code == 200


def upcoming_games(
    settings: Settings, *, now: datetime, horizon_hours: float
) -> list[dict[str, Any]]:
    """Games tipping inside the horizon, from the trusted schedule."""
    path = settings.paths.processed / "nba_regular_season_games_2006_26.parquet"
    if not path.is_file():
        raise ConfigError(f"Missing {path}. Run the earlier phases first.")
    games = pd.read_parquet(path)
    tipoff = pd.to_datetime(games["game_datetime_utc"], utc=True)
    window = (tipoff >= pd.Timestamp(now)) & (
        tipoff <= pd.Timestamp(now + timedelta(hours=horizon_hours))
    )
    selected = games[window]
    return [
        {
            "game_id": row.nba_game_id,
            "scheduled_tipoff_utc": pd.Timestamp(row.game_datetime_utc).to_pydatetime(),
        }
        for row in selected.itertuples()
    ]


def run_pipeline(
    *,
    settings: Settings | None = None,
    now: datetime | None = None,
    horizon_hours: float = 36.0,
    dry_run: bool = False,
) -> dict[str, Any]:
    settings = settings or load_settings()
    settings.paths.ensure()
    now = now or utc_now()

    games = upcoming_games(settings, now=now, horizon_hours=horizon_hours)
    tasks = plan_captures(games, CAPTURE_SOURCES)
    due = [t for t in tasks if t.capture_at_utc <= now]

    report: dict[str, Any] = {
        "generated_at_utc": now.isoformat(),
        "horizon_hours": horizon_hours,
        "games_in_horizon": len(games),
        "tasks_planned": len(tasks),
        "tasks_due": len(due),
        "dry_run": dry_run,
    }
    if dry_run or not due:
        report["results"] = []
        report["note"] = (
            "Dry run: nothing fetched." if dry_run
            else "No capture was due at this time."
        )
        return _write(settings, report)

    archive = ReportArchive(settings.paths.root / "raw" / "availability" / "nba_official")
    store = SnapshotStore(settings.paths.root / "raw" / "availability" / "snapshots")

    with httpx.Client(follow_redirects=True) as client:
        runner = AvailabilityRunner(archive, store, fetch=PacedFetcher(client))
        results = runner.run(due, now=now)

        unavailable = [r for r in results if r.status == "source_unavailable"]
        canary_ok = True
        if unavailable:
            canary_ok = canary_is_reachable(client)

    report.update(
        {
            "stats": runner.stats.to_dict(),
            "anchor_health": anchor_health(results),
            "results": [r.to_dict() for r in results],
            "unavailable_count": len(unavailable),
            "canary_reachable": canary_ok,
            "trustworthy": canary_ok,
        }
    )
    if not canary_ok:
        report["WARNING"] = (
            f"{len(unavailable)} slots returned {NOT_AVAILABLE_STATUS} and the "
            "canary is also unreachable. This run was most likely rate limited, "
            "so its 'unavailable' verdicts must not be treated as evidence that "
            "those reports do not exist. Re-run after a pause."
        )
    return _write(settings, report)


def _write(settings: Settings, report: dict[str, Any]) -> dict[str, Any]:
    path = settings.paths.reports / "availability_capture_run.json"
    path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    report["written_files"] = [str(path)]
    return report


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run prospective availability captures.")
    parser.add_argument(
        "--horizon-hours", type=float, default=36.0,
        help="Look this far ahead for tipoffs (default: 36).",
    )
    parser.add_argument(
        "--now", type=str, default=None,
        help="Override the clock, ISO-8601 UTC. For rehearsal and testing.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Plan captures and report them without fetching anything.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s | %(message)s")
    now = None
    if args.now:
        parsed = datetime.fromisoformat(args.now)
        now = parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)
    try:
        report = run_pipeline(
            now=now, horizon_hours=args.horizon_hours, dry_run=args.dry_run
        )
    except ConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    print(f"Games in horizon : {report['games_in_horizon']}")
    print(f"Tasks planned/due: {report['tasks_planned']} / {report['tasks_due']}")
    if report.get("stats"):
        print(f"Stats            : {report['stats']}")
    if report.get("WARNING"):
        print(f"\nWARNING: {report['WARNING']}")
    for path in report["written_files"]:
        print(f"\nWrote {Path(path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
