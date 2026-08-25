"""Is the system ready to capture tonight's NBA slate?

One command, one answer: PASS, WARN or FAIL, with reasons. Run it before the
slate rather than discovering a problem from a hole in the data weeks later.

Every check is a live probe, not a configuration read. "The schedule file
exists" is not the same claim as "the schedule can be loaded and contains
tonight's games", and only the second one matters at tip-off.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import httpx
import pandas as pd

from nba_prediction_market.availability.nba_official import (
    BASE_URL,
    USER_AGENT,
    latest_slot_at_or_before,
)
from nba_prediction_market.capture.health import CRITICAL, INFO, WARNING
from nba_prediction_market.capture.kalshi_live import (
    PUBLIC_REST_BASE,
    streaming_capability,
)
from nba_prediction_market.config import KALSHI_NBA_SERIES_TICKER, Settings, load_settings
from nba_prediction_market.ingestion.raw_store import utc_now

logger = logging.getLogger(__name__)

#: A slot known to exist, used to tell "not published" from "throttled".
CANARY_FILENAME: str = "Injury-Report_2026-04-10_04_00PM.pdf"
#: How far ahead "tonight's slate" reaches.
SLATE_HORIZON_HOURS: float = 36.0
#: Below this the collector should not be started.
MIN_FREE_GIB: float = 5.0


@dataclass
class Check:
    name: str
    status: str
    detail: str
    context: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "check": self.name, "status": self.status,
            "detail": self.detail, "context": self.context or {},
        }


def _ok(name: str, detail: str, **context: Any) -> Check:
    return Check(name, "PASS", detail, context or None)


def check_clock() -> Check:
    """Local clock sanity, checked against a server's Date header.

    A skewed clock silently corrupts every latency measurement in the phase,
    and it is the one fault that leaves no trace in the data.
    """
    try:
        started = datetime.now(UTC)
        response = httpx.head(PUBLIC_REST_BASE + "/exchange/status", timeout=15.0)
        served = response.headers.get("date")
        if not served:
            return Check("clock", WARNING, "no Date header to compare against")
        remote = pd.Timestamp(served).tz_convert("UTC").to_pydatetime()
        skew = abs((remote - started).total_seconds())
        if skew > 60:
            return Check("clock", CRITICAL,
                         f"local clock differs from server by {skew:.0f}s",
                         {"skew_seconds": skew})
        if skew > 5:
            return Check("clock", WARNING, f"clock skew {skew:.1f}s",
                         {"skew_seconds": skew})
        return _ok("clock", f"skew {skew:.1f}s", skew_seconds=skew)
    except Exception as exc:
        return Check("clock", WARNING, f"could not verify: {type(exc).__name__}")


def check_report_source() -> Check:
    """The official report source answers, and the canary is not blocked."""
    try:
        with httpx.Client(headers={"User-Agent": USER_AGENT}, timeout=25.0) as client:
            canary = client.get(f"{BASE_URL}/{CANARY_FILENAME}")
            if canary.status_code != 200:
                return Check(
                    "official_report_source", CRITICAL,
                    f"canary returned {canary.status_code}; 403 responses "
                    "cannot be trusted as 'not published'",
                )
            slot = latest_slot_at_or_before(utc_now())
            live = client.get(slot.url)
        return _ok(
            "official_report_source",
            f"canary 200; latest slot {slot.filename} -> {live.status_code}",
            latest_slot=slot.filename, latest_status=live.status_code,
        )
    except Exception as exc:
        return Check("official_report_source", CRITICAL,
                     f"unreachable: {type(exc).__name__}: {exc}")


def check_kalshi() -> Check:
    """Public Kalshi REST answers and NBA markets are listed."""
    try:
        with httpx.Client(timeout=25.0) as client:
            response = client.get(
                f"{PUBLIC_REST_BASE}/markets",
                params={"series_ticker": KALSHI_NBA_SERIES_TICKER, "limit": 10},
            )
            if response.status_code != 200:
                return Check("kalshi_api", CRITICAL,
                             f"markets endpoint returned {response.status_code}")
            markets = response.json().get("markets", [])
            if not markets:
                return Check("kalshi_api", WARNING,
                             "reachable but no NBA markets listed yet")
            book = client.get(
                f"{PUBLIC_REST_BASE}/markets/{markets[0]['ticker']}/orderbook",
                params={"depth": 5},
            )
            if book.status_code != 200:
                return Check("kalshi_api", WARNING,
                             f"orderbook returned {book.status_code}; depth "
                             "capture would be unavailable")
        return _ok("kalshi_api", f"{len(markets)} markets listed; orderbook 200",
                   markets_listed=len(markets))
    except Exception as exc:
        return Check("kalshi_api", CRITICAL,
                     f"unreachable: {type(exc).__name__}: {exc}")


def check_schedule(settings: Settings) -> Check:
    """The forward schedule loads and reaches beyond today.

    Checks the *forward* schedule the collector captures against, not the
    frozen historical frame. A schedule that stops before today means the
    collector has nothing to anchor capture windows to.
    """
    path = settings.paths.processed / "nba_forward_schedule_2026_27.parquet"
    if not path.is_file():
        return Check("nba_schedule", CRITICAL,
                     "no forward schedule; run build_forward_schedule")
    try:
        games = pd.read_parquet(
            path, columns=["tipoff_utc", "phase", "counts_toward_research"]
        )
    except Exception as exc:
        return Check("nba_schedule", CRITICAL, f"unreadable: {type(exc).__name__}")

    tipoff = pd.to_datetime(games["tipoff_utc"], utc=True)
    now = pd.Timestamp(utc_now())
    latest = tipoff.max()
    if latest < now:
        return Check("nba_schedule", CRITICAL,
                     f"forward schedule ends {latest.date()}, before today",
                     {"latest_scheduled": str(latest.date())})

    future = int((tipoff >= now).sum())
    soon = int(
        ((tipoff >= now)
         & (tipoff <= now + pd.Timedelta(hours=SLATE_HORIZON_HOURS))).sum()
    )
    preseason = int((games["phase"] == "preseason").sum())
    regular = int(games["counts_toward_research"].sum())
    return _ok(
        "nba_schedule",
        f"{future} future game(s) to {latest.date()}; {soon} in the next "
        f"{SLATE_HORIZON_HOURS:.0f}h ({regular} regular, {preseason} preseason)",
        future_games=future, next_slate=soon,
        regular_season=regular, preseason=preseason,
    )


def check_storage(settings: Settings) -> Check:
    """Raw storage is writable and has room."""
    root = settings.paths.root / "raw" / "availability" / "nba_official"
    try:
        root.mkdir(parents=True, exist_ok=True)
        probe = root / ".readiness_probe"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
    except Exception as exc:
        return Check("raw_storage", CRITICAL,
                     f"not writable: {type(exc).__name__}: {exc}")
    free_gib = shutil.disk_usage(root).free / 2**30
    if free_gib < MIN_FREE_GIB:
        return Check("raw_storage", CRITICAL, f"only {free_gib:.1f} GiB free",
                     {"free_gib": free_gib})
    return _ok("raw_storage", f"writable, {free_gib:.1f} GiB free",
               free_gib=round(free_gib, 1))


def check_model_artifact(settings: Settings) -> Check:
    """The frozen model's inputs exist, so anchor snapshots can be produced."""
    required = [
        settings.paths.processed / "nba_model_features_3a3_2006_26.parquet",
        settings.paths.reports / "model_availability_2025_26.json",
    ]
    missing = [p for p in required if not p.is_file()]
    if missing:
        return Check("frozen_model", WARNING,
                     f"missing {[p.name for p in missing]}")
    frozen = json.loads(required[1].read_text())["frozen_configuration"]
    return _ok("frozen_model",
               f"bundle {frozen['bundle']} C={frozen['C']}",
               bundle=frozen["bundle"], c=frozen["C"])


def check_locks(settings: Settings) -> Check:
    """No stale lock or half-finished process state."""
    lock = settings.paths.root / "raw" / "capture" / "collector.lock"
    if not lock.is_file():
        return _ok("process_state", "no lock held")
    try:
        payload = json.loads(lock.read_text())
        pid = int(payload.get("pid", -1))
        heartbeat = pd.Timestamp(payload.get("heartbeat_at_utc"))
    except Exception:
        return Check("process_state", WARNING, "lock file unreadable")
    alive = False
    try:
        os.kill(pid, 0)
        alive = True
    except (OSError, ProcessLookupError):
        alive = False
    age = (pd.Timestamp(utc_now()) - heartbeat).total_seconds()
    if alive:
        return Check("process_state", INFO,
                     f"collector already running (pid {pid}, heartbeat {age:.0f}s ago)")
    return Check("process_state", WARNING,
                 f"stale lock from dead pid {pid}: the collector died without "
                 f"releasing it. Starting the collector clears it automatically; "
                 f"the warning stands because a crash is worth noticing.")


def run_checks(settings: Settings | None = None) -> dict[str, Any]:
    settings = settings or load_settings()
    settings.paths.ensure()
    checks = [
        check_clock(),
        check_report_source(),
        check_kalshi(),
        check_schedule(settings),
        check_storage(settings),
        check_model_artifact(settings),
        check_locks(settings),
    ]
    severities = {c.status for c in checks}
    status = (
        "FAIL" if CRITICAL in severities
        else "WARN" if WARNING in severities else "PASS"
    )
    return {
        "status": status,
        "checked_at_utc": utc_now().isoformat(),
        "checks": [c.to_dict() for c in checks],
        "streaming": streaming_capability(),
        "horizon_hours": SLATE_HORIZON_HOURS,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Is the system ready to capture tonight's NBA slate?"
    )
    parser.add_argument("--json", action="store_true", help="Emit JSON only.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    logging.basicConfig(level=logging.WARNING)
    report = run_checks()
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(f"\nREADINESS: {report['status']}\n")
        for check in report["checks"]:
            marker = {"PASS": "  ok  ", CRITICAL: " FAIL ", WARNING: " warn ",
                      INFO: " info "}.get(check["status"], "  ?   ")
            print(f"[{marker}] {check['check']:24s} {check['detail']}")
        stream = report["streaming"]
        available = (
            "available" if stream["available_to_this_project"] else "unavailable"
        )
        print(f"\nstreaming: {available} - {stream['reason']}")
        print(f"fallback : {stream['fallback']}")
    return {"PASS": 0, "WARN": 0, "FAIL": 1}[report["status"]]


if __name__ == "__main__":
    raise SystemExit(main())
