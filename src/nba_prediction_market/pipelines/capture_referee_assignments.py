"""Capture the day's published referee assignments, immutably and as-of.

Runs alongside the main collector but is never load-bearing for it: a failure
here is logged and the slate carries on, because a missing crew costs one
optional feature while a missed report or quote costs coverage that cannot be
recovered.

Every observation is appended, never overwritten. If the league reassigns a
crew during the day both states survive with their own first-observed times,
so a prediction made at T-6h can always be reconstructed from what was
actually known then.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import httpx

from nba_prediction_market.capture.observation import (
    ObservationTiming,
    RawObservation,
    content_hash,
)
from nba_prediction_market.config import Settings, load_settings
from nba_prediction_market.ingestion.raw_store import utc_now
from nba_prediction_market.referees.assignments import (
    ASSIGNMENTS_URL,
    AssignmentLedger,
    AssignmentRow,
    parse_assignments,
)

logger = logging.getLogger(__name__)

EASTERN = ZoneInfo("America/New_York")
REQUEST_TIMEOUT_SECONDS: float = 30.0

#: official.nba.com answers 403 to a request without a browser user agent, so
#: this is required to read a public page rather than a way around any control.
REQUEST_HEADERS: dict[str, str] = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml",
    "Accept-Language": "en-US,en;q=0.9",
}


def capture_root(settings: Settings) -> Path:
    return settings.paths.root / "raw" / "referees" / "assignments"


def _append(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, default=str) + "\n")


def load_ledger(settings: Settings, day: str) -> AssignmentLedger:
    """Rebuild the day's ledger from what was already observed.

    Replaying the log rather than keeping state in memory is what makes a
    restart harmless: the same page re-observed produces no change, and the
    original first-observed times survive.
    """
    ledger = AssignmentLedger()
    path = capture_root(settings) / day / "observations.jsonl"
    if not path.is_file():
        return ledger
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        rows = [
            AssignmentRow(
                matchup=r["matchup"], crew_chief=r.get("crew_chief"),
                referee=r.get("referee"), umpire=r.get("umpire"),
                alternate=r.get("alternate"),
                first_observed_at_utc=datetime.fromisoformat(
                    r["first_observed_at_utc"]
                ),
                source_date_et=r.get("source_date_et"),
            )
            for r in record.get("rows", [])
        ]
        if rows:
            ledger.observe(rows, now=rows[0].first_observed_at_utc)
    return ledger


def capture_once(
    settings: Settings, *, client: httpx.Client | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Fetch, parse, and append one observation of the assignment page."""
    settings.paths.ensure()
    now = now or utc_now()
    day = now.astimezone(EASTERN).date().isoformat()
    owned = client is None
    client = client or httpx.Client(follow_redirects=True, headers=REQUEST_HEADERS)

    started = utc_now()
    try:
        response = client.get(ASSIGNMENTS_URL, timeout=REQUEST_TIMEOUT_SECONDS)
        received = utc_now()
        observation = RawObservation(
            source="nba_referee_assignments", url=ASSIGNMENTS_URL,
            http_status=response.status_code,
            timing=ObservationTiming(started, received, received),
            content_sha256=content_hash(response.content),
            content_bytes=len(response.content),
        )
        html = response.text if response.status_code == 200 else ""
    except httpx.HTTPError as exc:
        received = utc_now()
        observation = RawObservation(
            source="nba_referee_assignments", url=ASSIGNMENTS_URL, http_status=0,
            timing=ObservationTiming(started, received, received),
            content_sha256=content_hash(b""), content_bytes=0, error=str(exc),
        )
        html = ""
    finally:
        if owned:
            client.close()

    root = capture_root(settings) / day
    if not observation.succeeded:
        _append(root / "failures.jsonl", [observation.to_dict()])
        return {
            "captured": False, "date_et": day,
            "error": observation.error or f"HTTP {observation.http_status}",
            "observation": observation.to_dict(),
        }

    # Archive the raw page before parsing it, so a parser change never needs
    # a refetch and never loses the evidence.
    root.mkdir(parents=True, exist_ok=True)
    (root / f"page_{received.strftime('%Y%m%dT%H%M%SZ')}.html").write_text(
        html, encoding="utf-8"
    )

    rows = parse_assignments(html, observed_at_utc=received, source_date_et=day)
    ledger = load_ledger(settings, day)
    changes = ledger.observe(rows, now=received)

    _append(root / "observations.jsonl", [{
        "observed_at_utc": received.isoformat(),
        "timing": observation.to_dict(),
        "rows": [r.to_dict() for r in rows],
    }])
    _append(root / "changes.jsonl", [c.to_dict() for c in changes])

    return {
        "captured": True,
        "date_et": day,
        "observed_at_utc": received.isoformat(),
        "assignments": len(rows),
        "changes": [c.to_dict() for c in changes],
        "ledger": ledger.summary(),
        # An empty table is normal: the offseason, and every morning before
        # the league posts. It is not a failure and must not read as one.
        "empty_is_expected": len(rows) == 0,
    }


def run_pipeline(*, settings: Settings | None = None) -> dict[str, Any]:
    settings = settings or load_settings()
    result = capture_once(settings)
    path = settings.paths.reports / "referee_assignment_capture.json"
    path.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    result["written_files"] = [str(path)]
    return result


def main(argv: list[str] | None = None) -> int:
    argparse.ArgumentParser(description="Capture NBA referee assignments.").parse_args(
        argv
    )
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)-7s | %(message)s"
    )
    result = run_pipeline()
    if not result["captured"]:
        print(f"capture failed: {result['error']}", file=sys.stderr)
        return 1
    print(f"Date (ET)   : {result['date_et']}")
    print(f"Assignments : {result['assignments']}")
    print(f"Changes     : {len(result['changes'])}")
    if result["empty_is_expected"]:
        print("No assignments published yet - expected outside the season "
              "and before ~9:00 AM ET on a game day.")
    for path in result["written_files"]:
        print(f"\nWrote {Path(path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
