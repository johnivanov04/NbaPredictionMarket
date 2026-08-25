"""Read the running collector's health without disturbing it.

Strictly a reader. ``check_capture_readiness`` answers "is it safe to start?";
this answers "is the thing that is already running actually working?" -- which
is a different question, because a collector can hold its lock, beat its
heartbeat, and still be capturing nothing.

Exit code carries the verdict so a supervisor can act on it: 0 PASS, 1 WARN,
2 FAIL, 3 no collector state at all.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from nba_prediction_market.capture.health import HEARTBEAT_TIMEOUT_SECONDS
from nba_prediction_market.config import Settings, load_settings
from nba_prediction_market.ingestion.raw_store import utc_now

EXIT_CODES: dict[str, int] = {"PASS": 0, "WARN": 1, "FAIL": 2}


def read_state(settings: Settings) -> dict[str, Any] | None:
    path = settings.paths.reports / "capture_health.json"
    if not path.is_file():
        return None
    return json.loads(path.read_text())


def read_lock(settings: Settings) -> dict[str, Any] | None:
    path = settings.paths.root / "raw" / "capture" / "collector.lock"
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return None


def run_pipeline(*, settings: Settings | None = None) -> dict[str, Any]:
    settings = settings or load_settings()
    now = utc_now()
    state = read_state(settings)
    lock = read_lock(settings)

    result: dict[str, Any] = {
        "checked_at_utc": now.isoformat(),
        "collector_running": lock is not None,
        "lock": lock,
        "health": state,
    }
    if lock is not None:
        beat = datetime.fromisoformat(lock["heartbeat_at_utc"])
        age = (now - beat).total_seconds()
        result["heartbeat_age_seconds"] = round(age, 1)
        # A held lock with a cold heartbeat is worse than no lock: it means a
        # process is alive but no longer capturing, which nothing else notices.
        result["heartbeat_stale"] = age > HEARTBEAT_TIMEOUT_SECONDS
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Show live collector health.")
    parser.add_argument("--json", action="store_true", help="Emit raw JSON.")
    args = parser.parse_args(argv)

    result = run_pipeline()
    if args.json:
        print(json.dumps(result, indent=2, default=str))

    health = result.get("health")
    if health is None:
        print("No collector health state found. Has the collector ever run?")
        return 3

    lock = result.get("lock")
    if lock is None:
        print("Collector    : not running (no lock held)")
    else:
        stale = " STALE" if result.get("heartbeat_stale") else ""
        print(
            f"Collector    : running, pid {lock['pid']}, "
            f"heartbeat {result['heartbeat_age_seconds']}s ago{stale}"
        )

    status = health.get("status", "unknown")
    print(f"Status       : {status}")
    for key, value in (health.get("counters") or {}).items():
        print(f"  {key:26} {value}")
    for issue in health.get("issues", []):
        print(f"  [{issue['severity']:8}] {issue['code']}: {issue['detail']}")
    print(f"\nRead {Path(load_settings().paths.reports / 'capture_health.json')}")
    return EXIT_CODES.get(status, 2)


if __name__ == "__main__":
    raise SystemExit(main())
