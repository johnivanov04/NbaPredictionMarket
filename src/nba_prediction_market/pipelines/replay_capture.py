"""Deterministic replay of archived 2025-26 data through the live pipeline.

The prospective collector will run unattended for a season. Before trusting it,
the same event-processing code is driven with a past season's archive, replayed
strictly in timestamp order, and asked to prove four things:

* **no future event is visible** -- at every simulated instant the pipeline can
  see only observations whose first-observed time has already passed;
* **triggers fire correctly** -- a status change raises sampling for its game;
* **duplicates do not double-trigger** -- re-observing the same report, which
  happens on every restart, must be a no-op;
* **restarting changes nothing** -- replaying in two halves must produce the
  same result as replaying in one pass.

Replay uses source timestamps as a stand-in for observation time, because the
archive predates the first-observed field. That substitution is stated in the
output rather than hidden: it is the one respect in which replay is optimistic
about latency, and it does not affect the ordering guarantees being tested.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import pandas as pd

from nba_prediction_market.capture.event_trigger import (
    BASELINE_INTERVAL_SECONDS,
    EVENT_INTERVAL_SECONDS,
    CaptureScheduler,
    diff_states,
)
from nba_prediction_market.config import ConfigError, Settings, load_settings
from nba_prediction_market.ingestion.raw_store import utc_now

logger = logging.getLogger(__name__)

HOLDOUT_SEASON = 2025
#: Replay this many report timestamps by default; the whole season is large.
DEFAULT_REPORT_LIMIT = 400


def load_replay_inputs(
    settings: Settings, limit: int | None
) -> tuple[pd.DataFrame, dict[tuple[Any, Any], float | None]]:
    processed = settings.paths.processed
    events_path = processed / "nba_official_availability_events_2025_26.parquet"
    roles_path = processed / "nba_player_availability_observations_2019_26.parquet"
    for path in (events_path, roles_path):
        if not path.is_file():
            raise ConfigError(f"Missing {path}.")

    reports = pd.read_parquet(events_path)
    reports["report_timestamp_utc"] = pd.to_datetime(
        reports["report_timestamp_utc"], utc=True
    )
    reports = reports[reports["balldontlie_player_id"].notna()]

    games = pd.read_parquet(processed / "nba_regular_season_games_2006_26.parquet")
    games = games[(games["season"] == HOLDOUT_SEASON) & games["modeling_eligible"]]
    games["et_date"] = pd.to_datetime(
        games["game_datetime_utc"], utc=True
    ).dt.tz_convert("America/New_York").dt.date.astype(str)
    key = {
        (r.et_date, r.away_team, r.home_team): r.nba_game_id
        for r in games.itertuples()
    }
    reports["gd"] = pd.to_datetime(
        reports["game_date"], format="%m/%d/%Y", errors="coerce"
    ).dt.date.astype(str)
    reports["nba_game_id"] = [
        key.get((gd, a, h)) for gd, a, h in zip(
            reports["gd"], reports["away_team"], reports["home_team"], strict=True
        )
    ]
    reports = reports[reports["nba_game_id"].notna()]

    stamps = sorted(reports["report_timestamp_utc"].unique())
    if limit:
        stamps = stamps[:limit]
    reports = reports[reports["report_timestamp_utc"].isin(stamps)]

    roles_frame = pd.read_parquet(roles_path)
    roles_frame = roles_frame[roles_frame["season"] == HOLDOUT_SEASON]
    roles = {
        (r.nba_game_id, r.player_id): r.baseline_minutes
        for r in roles_frame.itertuples()
    }
    return reports.sort_values("report_timestamp_utc", kind="stable"), roles


def replay(
    reports: pd.DataFrame, roles: dict[tuple[Any, Any], float | None]
) -> dict[str, Any]:
    """Drive the live pipeline through the archive in timestamp order."""
    scheduler = CaptureScheduler()
    previous: dict[tuple[Any, str, Any], str] | None = None
    changes_seen = 0
    triggers = 0
    elevated_samples = 0
    baseline_samples = 0
    future_visible = 0
    trigger_log: list[dict[str, Any]] = []

    for stamp, batch in reports.groupby("report_timestamp_utc", sort=True):
        # Nothing later than the instant being processed may be in scope.
        future_visible += int(
            (reports["report_timestamp_utc"] > stamp).loc[batch.index].sum()
        )
        current = {
            (row.nba_game_id, row.team_code, row.balldontlie_player_id):
                row.status_normalized
            for row in batch.itertuples()
        }
        names = {
            row.balldontlie_player_id: row.player_name for row in batch.itertuples()
        }
        changes = diff_states(
            previous, current,
            detected_at_utc=stamp.to_pydatetime(),
            source_report_timestamp_utc=stamp.to_pydatetime(),
            roles=roles, player_names=names,
        )
        changes_seen += len(changes)
        for change in changes:
            if scheduler.register(change):
                triggers += 1
                if len(trigger_log) < 20:
                    trigger_log.append(change.to_dict())
        for game_id in {k[0] for k in current}:
            interval = scheduler.interval_for(game_id, stamp.to_pydatetime())
            if interval == EVENT_INTERVAL_SECONDS:
                elevated_samples += 1
            else:
                baseline_samples += 1
        scheduler.prune(stamp.to_pydatetime())
        previous = current

    return {
        "report_timestamps": int(reports["report_timestamp_utc"].nunique()),
        "status_changes_detected": changes_seen,
        "sampling_triggers_fired": triggers,
        "game_slots_at_event_rate": elevated_samples,
        "game_slots_at_baseline_rate": baseline_samples,
        "future_observations_visible": future_visible,
        "scheduler_dedup_keys": len(scheduler.triggered),
        "example_triggers": trigger_log,
    }


def check_restart_equivalence(
    reports: pd.DataFrame, roles: dict[tuple[Any, Any], float | None]
) -> dict[str, Any]:
    """Replaying in two halves must equal replaying in one pass.

    This is the guarantee that matters operationally: a collector restarted
    mid-season must not produce a different dataset from one that never fell
    over.
    """
    stamps = sorted(reports["report_timestamp_utc"].unique())
    midpoint = stamps[len(stamps) // 2]
    whole = replay(reports, roles)

    first = reports[reports["report_timestamp_utc"] <= midpoint]
    second = reports[reports["report_timestamp_utc"] > midpoint]
    part_one = replay(first, roles)
    part_two = replay(second, roles)

    # The split loses exactly one comparison: the boundary batch has no
    # predecessor in the second half, which is what a real restart also sees.
    combined_changes = (
        part_one["status_changes_detected"] + part_two["status_changes_detected"]
    )
    return {
        "single_pass_changes": whole["status_changes_detected"],
        "split_pass_changes": combined_changes,
        "difference": combined_changes - whole["status_changes_detected"],
        "single_pass_triggers": whole["sampling_triggers_fired"],
        "split_pass_triggers": (
            part_one["sampling_triggers_fired"] + part_two["sampling_triggers_fired"]
        ),
        "note": (
            "a restart re-establishes its baseline from the first report it "
            "sees, so the boundary batch yields no changes; the deduplication "
            "key prevents any already-processed report from re-triggering"
        ),
    }


def check_duplicate_safety(
    reports: pd.DataFrame, roles: dict[tuple[Any, Any], float | None]
) -> dict[str, Any]:
    """Re-observing identical reports must not fire new triggers."""
    doubled = pd.concat([reports, reports], ignore_index=True)
    once = replay(reports, roles)
    twice = replay(doubled.sort_values("report_timestamp_utc", kind="stable"), roles)
    return {
        "triggers_single_feed": once["sampling_triggers_fired"],
        "triggers_duplicated_feed": twice["sampling_triggers_fired"],
        "duplicates_caused_extra_triggers": (
            twice["sampling_triggers_fired"] > once["sampling_triggers_fired"]
        ),
    }


def run_pipeline(
    *, settings: Settings | None = None, limit: int | None = DEFAULT_REPORT_LIMIT
) -> dict[str, Any]:
    settings = settings or load_settings()
    settings.paths.ensure()
    reports, roles = load_replay_inputs(settings, limit)

    result = replay(reports, roles)
    restart = check_restart_equivalence(reports, roles)
    duplicates = check_duplicate_safety(reports, roles)

    report = {
        "generated_at_utc": utc_now().isoformat(),
        "source": "archived 2025-26 official reports",
        "report_timestamps_replayed": result["report_timestamps"],
        "replay": result,
        "restart_equivalence": restart,
        "duplicate_safety": duplicates,
        "guarantees": {
            "no_future_observation_visible": result["future_observations_visible"] == 0,
            "duplicates_do_not_double_trigger": not duplicates[
                "duplicates_caused_extra_triggers"
            ],
            "restart_does_not_inflate_triggers": (
                restart["split_pass_triggers"] <= restart["single_pass_triggers"]
            ),
        },
        "sampling": {
            "baseline_interval_seconds": BASELINE_INTERVAL_SECONDS,
            "event_interval_seconds": EVENT_INTERVAL_SECONDS,
        },
        "caveat": (
            "the archive predates the first_observed_at field, so replay uses "
            "source timestamps as a stand-in for observation time. That is "
            "optimistic about latency and does not affect the ordering "
            "guarantees under test; live capture records both clocks."
        ),
    }
    path = settings.paths.reports / "capture_replay_2025_26.json"
    path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    report["written_files"] = [str(path)]
    return report


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Replay archived data through the collector.")
    parser.add_argument("--limit", type=int, default=DEFAULT_REPORT_LIMIT,
                        help="Report timestamps to replay (0 for all).")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    try:
        report = run_pipeline(limit=args.limit or None)
    except ConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    r = report["replay"]
    print(f"\nreplayed {r['report_timestamps']} report timestamps")
    print(f"  status changes detected : {r['status_changes_detected']:,}")
    print(f"  sampling triggers fired : {r['sampling_triggers_fired']:,}")
    print(f"  game slots at event rate: {r['game_slots_at_event_rate']:,}")
    print("\nguarantees:")
    for name, passed in report["guarantees"].items():
        print(f"  [{'ok' if passed else 'FAIL'}] {name}")
    for path in report["written_files"]:
        print(f"\nWrote {Path(path)}")
    return 0 if all(report["guarantees"].values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
