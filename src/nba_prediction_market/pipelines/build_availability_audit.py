"""Phase 3A3B0: availability source audit and prospective capture foundation.

Writes ``data/reports/availability_source_audit.json``. Builds **no** model and
merges nothing into the Phase 3A3 feature set: availability may not become a
feature until a source's as-of properties are proven.

Run with::

    python -m nba_prediction_market.pipelines.build_availability_audit
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import UTC, datetime, timedelta
from typing import Any

from nba_prediction_market.availability.as_of import coverage_at, state_at
from nba_prediction_market.availability.capture_schedule import (
    ANCHOR_MINUTES_BEFORE_TIP,
    DEFAULT_OFFSETS_MINUTES,
    REPORT_GRID_MINUTES,
    anchor_for,
    plan_captures,
    verify_anchor_guarantee,
)
from nba_prediction_market.availability.events import (
    NORMALIZED_STATUSES,
    TemporalPrecision,
    event_from_row,
)
from nba_prediction_market.availability.sources import (
    HISTORICALLY_UNSAFE,
    SOURCE_MATRIX,
)
from nba_prediction_market.config import ConfigError, Settings, load_settings
from nba_prediction_market.ingestion.raw_store import utc_now

logger = logging.getLogger(__name__)

#: Market anchors to preserve for prospective 2026-27 research. Design only.
KALSHI_ANCHORS_MINUTES_BEFORE_TIP: tuple[int, ...] = (
    24 * 60, 6 * 60, 3 * 60, 60, 30, 15, 5,
)


def kalshi_multi_anchor_design() -> dict[str, Any]:
    """Design for preserving market state at several anchors.

    Phase 2 established that Kalshi's 1-minute candlesticks already contain
    every anchor we might want, so the right design is to widen the *stored
    window* rather than to poll at each anchor separately. One request per
    market covering tip-24h to tip yields every anchor below by selection, and
    keeps future anchors reconstructable without refetching.
    """
    return {
        "anchors_minutes_before_tip": list(KALSHI_ANCHORS_MINUTES_BEFORE_TIP),
        "approach": "store_the_candlestick_stream_not_point_samples",
        "rationale": (
            "Phase 2 already selects a T-30 quote from 1-minute candlesticks. "
            "Widening the cached window to tip-24h..tip lets any anchor be "
            "derived later by the same lookahead-safe selector, so a new anchor "
            "costs no additional requests and no refetch."
        ),
        "window": {"start": "scheduled_tipoff - 24h", "end": "scheduled_tipoff"},
        "period_interval_minutes": 1,
        "reuses": "nba_prediction_market.ingestion.candlesticks.select_pregame_quote",
        "selection_rule": "latest candle with end_period_ts <= anchor",
        "storage": "existing CandleCache, keyed by request geometry",
        "estimated_requests": "2 per game (one per team market), unchanged from Phase 2",
        "trading": "explicitly out of scope",
    }


def demonstrate_as_of_engine() -> dict[str, Any]:
    """A worked example proving the anchor boundary, included in the report."""
    tipoff = datetime(2026, 11, 5, 0, 30, tzinfo=UTC)
    anchor = anchor_for(tipoff)
    events = [
        event_from_row(
            "nba_official_injury_report", anchor - timedelta(hours=3),
            player_name="Example, Player", status_raw="Questionable",
            temporal_precision=TemporalPrecision.EXACT, team_id=1,
        ),
        event_from_row(
            "nba_official_injury_report", anchor,
            player_name="Example, Player", status_raw="Out",
            temporal_precision=TemporalPrecision.EXACT, team_id=1,
        ),
        event_from_row(
            "nba_official_injury_report", anchor + timedelta(seconds=1),
            player_name="Example, Player", status_raw="Available",
            temporal_precision=TemporalPrecision.EXACT, team_id=1,
        ),
        event_from_row(
            "sportradar_daily_injuries", anchor - timedelta(hours=2),
            player_name="Example, Player", status_raw="Day To Day",
            temporal_precision=TemporalPrecision.DATE_ONLY, team_id=1,
        ),
    ]
    state = state_at(events, anchor)
    coverage = coverage_at(events, anchor)
    resolved = next(iter(state.values()))
    return {
        "anchor_utc": anchor.isoformat(),
        "observations_supplied": len(events),
        "state_at_anchor": resolved.to_dict(),
        "boundary_behaviour": {
            "observation_exactly_at_anchor": "accepted",
            "observation_one_second_after_anchor": "rejected",
            "date_only_observation": "refused as anchor-unsafe",
        },
        "coverage": coverage.to_dict(),
    }


def build_report(settings: Settings) -> dict[str, Any]:
    """Assemble the availability audit."""
    games = [
        {"game_id": i, "scheduled_tipoff_utc": datetime(2026, 11, 5, 0, 30, tzinfo=UTC)}
        for i in range(1, 4)
    ]
    tasks = plan_captures(
        games, [s.name for s in SOURCE_MATRIX if s.prospective_usefulness != "none at T-30"]
    )
    return {
        "generated_at_utc": utc_now().isoformat(),
        "research_principle": {
            "anchor": f"scheduled tipoff - {ANCHOR_MINUTES_BEFORE_TIP} minutes",
            "usability_rule": "observed_at <= prediction_ts",
            "note": (
                "A status learned one second after the anchor is future "
                "information. Final participation is never a substitute for "
                "pregame status."
            ),
        },
        "source_matrix": [s.to_dict() for s in SOURCE_MATRIX],
        "historically_unsafe_sources": HISTORICALLY_UNSAFE,
        "normalized_status_vocabulary": list(NORMALIZED_STATUSES),
        "temporal_precision_levels": {
            "exact_timestamp": "may satisfy a T-30 anchor",
            "date_only": "may NOT satisfy a T-30 anchor",
            "unknown": "may NOT satisfy a T-30 anchor",
        },
        "as_of_engine_demonstration": demonstrate_as_of_engine(),
        "prospective_capture": {
            "offsets_minutes_before_tip": list(DEFAULT_OFFSETS_MINUTES),
            "report_grid_minutes": REPORT_GRID_MINUTES,
            "planned_tasks_for_example_slate": len(tasks),
            "anchor_guarantee": verify_anchor_guarantee(tasks, games),
            "storage": "append-only immutable snapshots under data/raw/availability/",
            "principle": (
                "Schedule-aware rather than constant polling: work backwards from "
                "each game's tipoff and guarantee one capture immediately before "
                "its anchor."
            ),
        },
        "kalshi_multi_anchor_design": kalshi_multi_anchor_design(),
        "recommendations": {
            "historical_source": (
                "None proven. The NBA official archive has the right timestamp "
                "precision but only ~8 months of retention; Sportradar's "
                "update_date is date-only; BALLDONTLIE has no timestamp at all. "
                "Historical T-30 availability for 2006-2025 appears unrecoverable "
                "from the audited sources."
            ),
            "prospective_sources": [
                "nba_official_injury_report (primary, free, 30-minute grid)",
                "balldontlie_injuries (secondary, already paid for)",
            ],
            "trial_to_run": (
                "SportsDataIO, to answer one question: does a historical "
                "InjuriesByDate/StartingLineupsByDate response preserve "
                "point-in-time status, or overwrite with final state?"
            ),
            "expected_cost": "$0 to begin capturing prospectively",
            "before_availability_enters_a_model": (
                "Accumulate a full season of self-captured snapshots, then verify "
                "on held-out games that state_at(anchor) reproduces what was "
                "knowable, before any feature is derived."
            ),
        },
    }


def run_pipeline(*, settings: Settings | None = None) -> dict[str, Any]:
    settings = settings or load_settings()
    settings.paths.ensure()
    report = build_report(settings)
    path = settings.paths.reports / "availability_source_audit.json"
    path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    report["written_files"] = [str(path)]
    # Create the snapshot hierarchy so prospective capture has somewhere to land.
    for source in (
        "balldontlie", "nba_official", "sportradar", "sportsdataio"
    ):
        (settings.paths.root / "raw" / "availability" / source).mkdir(
            parents=True, exist_ok=True
        )
    return report


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m nba_prediction_market.pipelines.build_availability_audit",
        description="Phase 3A3B0: availability source audit and capture foundation.",
    )
    parser.add_argument("--data-dir", default=None)
    parser.add_argument(
        "--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"]
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level))
    try:
        report = run_pipeline(settings=load_settings(args.data_dir))
    except ConfigError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2

    print()
    print("Availability source matrix")
    print(f"  {'source':<32s} {'coverage':<38s} {'precision':<20s} {'as-of'}")
    for source in report["source_matrix"]:
        print(
            f"  {source['source']:<32s} {source['historical_coverage'][:37]:<38s} "
            f"{source['intraday_timestamp_precision']:<20s} "
            f"{source['historical_as_of_safety']}"
        )
    print()
    demo = report["as_of_engine_demonstration"]
    print(f"As-of engine at {demo['anchor_utc']}:")
    print(f"  resolved status: {demo['state_at_anchor']['status_normalized']} "
          f"(raw {demo['state_at_anchor']['status_raw']!r})")
    print(f"  boundary: {demo['boundary_behaviour']}")
    print()
    print("Files written:")
    for path in report["written_files"]:
        print(f"  {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
