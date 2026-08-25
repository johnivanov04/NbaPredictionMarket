"""Post-slate summary of what one night's collector actually captured.

Answers the question the soak-test checklist asks: for every game that was
supposed to be observed, was it? Reads only what the collector wrote, so it is
safe to run while a later slate is in progress.

Coverage is reported against the slate that *was scheduled*, not against what
was captured. Summarising captures alone would report 100% on a night the
collector never started, which is the one night the summary needs to shout.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

from nba_prediction_market.capture.schedule import reports_expected
from nba_prediction_market.config import Settings, load_settings
from nba_prediction_market.ingestion.raw_store import utc_now
from nba_prediction_market.pipelines.build_forward_schedule import load_stored

EASTERN = "America/New_York"


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def games_on(settings: Settings, day: date) -> list[Any]:
    """Every scheduled game whose ET date is ``day``."""
    from zoneinfo import ZoneInfo

    eastern = ZoneInfo(EASTERN)
    return [
        g for g in load_stored(settings)
        if g.tipoff_utc.astimezone(eastern).date() == day
    ]


def run_pipeline(
    *, settings: Settings | None = None, day: date | None = None
) -> dict[str, Any]:
    settings = settings or load_settings()
    day = day or (utc_now() - timedelta(hours=12)).date()
    root = settings.paths.root / "raw" / "capture"

    games = games_on(settings, day)
    books = _read_jsonl(root / "markets" / day.isoformat() / "books.jsonl")
    observations = _read_jsonl(
        root / "markets" / day.isoformat() / "observations.jsonl"
    )
    changes = _read_jsonl(root / "events" / day.isoformat() / "status_changes.jsonl")

    observed_games = {b.get("source_game_id") for b in books}
    scheduled_ids = {g.source_game_id for g in games}
    research = [g for g in games if g.counts_toward_research]

    missed = sorted(
        str(g.source_game_id) for g in games
        if g.source_game_id not in observed_games
    )
    failures = [o for o in observations if not o.get("succeeded")]
    intervals = Counter(b.get("sampling_interval_seconds") for b in books)

    summary = {
        "date_et": day.isoformat(),
        "generated_at_utc": utc_now().isoformat(),
        "games_scheduled": len(games),
        "games_counting_toward_research": len(research),
        "games_with_market_observations": len(observed_games & scheduled_ids),
        "games_never_observed": missed,
        "official_reports_expected": reports_expected(games),
        "book_rows": len(books),
        "fetch_observations": len(observations),
        "fetch_failures": len(failures),
        "status_changes_detected": len(changes),
        "book_rows_at_event_rate": intervals.get(5.0, 0),
        "book_rows_at_baseline_rate": intervals.get(60.0, 0),
    }
    if books:
        stamps = sorted(
            datetime.fromisoformat(b["observed_at_utc"]) for b in books
            if b.get("observed_at_utc")
        )
        if stamps:
            summary["first_observation_utc"] = stamps[0].isoformat()
            summary["last_observation_utc"] = stamps[-1].isoformat()

    verdict = "PASS"
    if games and not books:
        verdict = "FAIL"
    elif missed or failures:
        verdict = "WARN"
    summary["verdict"] = verdict

    path = settings.paths.reports / f"capture_day_{day.isoformat()}.json"
    path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    summary["written_files"] = [str(path)]
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Summarise one captured slate.")
    parser.add_argument(
        "--date", type=str, default=None,
        help="ET date to summarise (YYYY-MM-DD). Defaults to the last slate.",
    )
    args = parser.parse_args(argv)
    day = date.fromisoformat(args.date) if args.date else None

    summary = run_pipeline(day=day)
    print(f"Slate {summary['date_et']}  --  {summary['verdict']}")
    for key in (
        "games_scheduled", "games_counting_toward_research",
        "games_with_market_observations", "official_reports_expected",
        "book_rows", "book_rows_at_event_rate", "book_rows_at_baseline_rate",
        "status_changes_detected", "fetch_failures",
    ):
        print(f"  {key:34} {summary[key]}")
    if summary["games_never_observed"]:
        print(f"  never observed: {', '.join(summary['games_never_observed'])}")
    for path in summary["written_files"]:
        print(f"\nWrote {Path(path)}")
    return 0 if summary["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
