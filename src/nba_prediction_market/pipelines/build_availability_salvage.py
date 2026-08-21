"""Phase 3A3B1: salvage the surviving official archive into usable history.

Parses every archived report, resolves players and games against the trusted
Phase 3A0 history, and builds the **true historical T-30 availability state** for
those 2025-26 games whose reports survive.

The defining constraint: a game's state uses only reports where
``report_timestamp <= prediction_ts``. If the first surviving report for a game
is later than its anchor, the state is marked unavailable rather than backfilled
from a later report.

Coverage is partial by construction -- the CDN retains roughly eight months --
and every artefact says so in its name and its report.

Run with::

    python -m nba_prediction_market.pipelines.build_availability_salvage
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

from nba_prediction_market.availability.archive_inventory import build_inventory
from nba_prediction_market.availability.identity import PlayerRegistry, unresolved_report
from nba_prediction_market.availability.nba_official import EASTERN, ReportArchive
from nba_prediction_market.availability.nba_report_parser import (
    ParsedReport,
    ReportParseError,
    parse_report_pdf,
)
from nba_prediction_market.config import ConfigError, Settings, load_settings
from nba_prediction_market.ingestion.raw_store import utc_now
from nba_prediction_market.matching.team_names import resolve_team

logger = logging.getLogger(__name__)

ANCHOR_MINUTES = 30

_MATCHUP_RE = re.compile(r"^([A-Z]{2,3})@([A-Z]{2,3})$")

EVENT_COLUMNS: list[str] = [
    "report_timestamp_utc", "report_timestamp_et", "source_filename",
    "game_date", "game_time_et", "matchup", "away_team", "home_team",
    "team", "team_code", "player_name", "player_name_key",
    "balldontlie_player_id", "status_raw", "status_normalized", "reason_raw",
]

T30_COLUMNS: list[str] = [
    "nba_game_id", "game_datetime_utc", "prediction_ts_utc", "home_team", "away_team",
    "t30_state_available", "selected_report_timestamp_utc", "report_age_minutes",
    "selected_report_filename", "players_reported", "players_resolved",
    "players_unresolved", "n_out", "n_doubtful", "n_questionable", "n_probable",
    "n_available", "n_unknown_status", "teams_not_submitted",
    "both_teams_submitted", "coverage_quality",
]


def parse_archive(archive: ReportArchive) -> tuple[list[ParsedReport], list[dict[str, Any]]]:
    """Parse every archived PDF, keeping failures rather than dropping them."""
    parsed: list[ParsedReport] = []
    failures: list[dict[str, Any]] = []
    for path in sorted(archive.root.rglob("Injury-Report_*.pdf")):
        if ".conflict-" in path.name:
            continue
        try:
            parsed.append(parse_report_pdf(path))
        except ReportParseError as exc:
            failures.append({"filename": path.name, "error": str(exc)})
            logger.warning("parse failure: %s", exc)
    parsed.sort(key=lambda r: r.report_timestamp_utc)
    return parsed, failures


def build_registry(players: pd.DataFrame) -> PlayerRegistry:
    """Register every BALLDONTLIE player, keyed by team and name."""
    registry = PlayerRegistry()
    seen: set[tuple[Any, Any]] = set()
    for row in players.itertuples():
        key = (row.player_id, row.team_id)
        if key in seen or not row.player_name:
            continue
        seen.add(key)
        try:
            registry.register(row.player_id, row.player_name, team_id=row.team_id)
        except ValueError:
            # Two players sharing a normalized name on one team: leave both
            # unresolved rather than picking one.
            logger.debug("ambiguous registration for %s", row.player_name)
    return registry


def events_frame(
    reports: list[ParsedReport], registry: PlayerRegistry, team_ids: dict[str, Any]
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Flatten parsed reports into normalized availability events."""
    from nba_prediction_market.availability.identity import normalize_name

    rows: list[dict[str, Any]] = []
    resolutions = []
    for report in reports:
        for entry in report.entries:
            resolution = resolve_team(entry.team)
            team_code = resolution.abbreviation if resolution.ok else None
            team_id = team_ids.get(team_code) if team_code else None
            resolved = registry.resolve(entry.player_name, team_id=team_id)
            resolutions.append(resolved)
            rows.append(
                {
                    "report_timestamp_utc": report.report_timestamp_utc,
                    "report_timestamp_et": report.report_timestamp_et.isoformat(),
                    "source_filename": report.source_filename,
                    "game_date": entry.game_date,
                    "game_time_et": entry.game_time_et,
                    "matchup": entry.matchup,
                    "away_team": entry.away_team,
                    "home_team": entry.home_team,
                    "team": entry.team,
                    "team_code": team_code,
                    "player_name": entry.player_name,
                    "player_name_key": normalize_name(entry.player_name),
                    "balldontlie_player_id": resolved.balldontlie_player_id,
                    "status_raw": entry.status_raw,
                    "status_normalized": entry.status_normalized,
                    "reason_raw": entry.reason_raw,
                }
            )
    frame = pd.DataFrame(rows, columns=EVENT_COLUMNS)
    frame = frame.sort_values(
        ["report_timestamp_utc", "team_code", "player_name"], kind="stable"
    ).reset_index(drop=True)
    return frame, unresolved_report(resolutions)


def match_reports_to_games(
    events: pd.DataFrame, games: pd.DataFrame
) -> tuple[dict[tuple[str, str, str], Any], dict[str, Any]]:
    """Map (game_date, away, home) from reports onto trusted game ids."""
    lookup: dict[tuple[str, str, str], list[Any]] = defaultdict(list)
    for row in games.itertuples():
        key = (str(row.date), row.away_team, row.home_team)
        lookup[key].append(row.nba_game_id)

    matched: dict[tuple[str, str, str], Any] = {}
    ambiguous, unmatched = [], []
    report_keys = {
        (
            datetime.strptime(r.game_date, "%m/%d/%Y").date().isoformat(),
            r.away_team, r.home_team,
        )
        for r in events.itertuples()
        if r.game_date and r.away_team and r.home_team
    }
    for key in sorted(report_keys):
        candidates = lookup.get(key, [])
        if len(candidates) == 1:
            matched[key] = candidates[0]
        elif len(candidates) > 1:
            ambiguous.append({"key": list(key), "candidates": [int(c) for c in candidates]})
        else:
            unmatched.append({"key": list(key)})

    # An unmatched key is not automatically a defect. The games frame holds the
    # 1,230 regular-season games only, so every playoff and play-in report is
    # expected to miss. Separating those from same-window misses is what makes
    # the count actionable rather than alarming.
    last_regular_season_date = max(str(row.date) for row in games.itertuples())
    after_season = [u for u in unmatched if u["key"][0] > last_regular_season_date]
    in_window = [u for u in unmatched if u["key"][0] <= last_regular_season_date]
    return matched, {
        "report_games": len(report_keys),
        "matched": len(matched),
        "ambiguous": len(ambiguous),
        "unmatched": len(unmatched),
        "ambiguous_examples": ambiguous[:10],
        "unmatched_breakdown": {
            "after_regular_season": len(after_season),
            "within_regular_season_window": len(in_window),
            "note": (
                "after_regular_season covers play-in and playoff games, which "
                "are outside the 1,230-game frame by design. Only the "
                "within-window keys are genuine source discrepancies."
            ),
        },
        "within_window_unmatched": in_window,
        "unmatched_examples": unmatched[:10],
    }


def not_submitted_index(
    reports: list[ParsedReport],
) -> set[tuple[str, str, str, str, str]]:
    """``(filename, game_date, away, home, team)`` for outstanding filings.

    A team missing from a report has *unknown* availability, not a clean bill of
    health, so this has to travel with the T-30 state rather than be inferred
    from the absence of rows. The key carries the game because one report covers
    several dates: a team can still be pending for tomorrow's game while its
    filing for tonight's is already in.
    """
    index: set[tuple[str, str, str, str, str]] = set()
    for report in reports:
        for pending in report.teams_not_submitted:
            if not (pending.game_date and pending.matchup):
                continue
            resolution = resolve_team(pending.team)
            if not resolution.ok:
                continue
            match = _MATCHUP_RE.match(pending.matchup)
            if not match:
                continue
            index.add((
                report.source_filename,
                datetime.strptime(pending.game_date, "%m/%d/%Y").date().isoformat(),
                match.group(1),
                match.group(2),
                resolution.abbreviation,
            ))
    return index


def build_t30_states(
    events: pd.DataFrame,
    games: pd.DataFrame,
    matched: dict[tuple[str, str, str], Any],
    not_submitted: set[tuple[str, str, str, str, str]] | None = None,
) -> pd.DataFrame:
    """The true historical T-30 state for every game whose reports survive."""
    by_game: dict[Any, list[Any]] = defaultdict(list)
    for row in events.itertuples():
        if not (row.game_date and row.away_team and row.home_team):
            continue
        key = (
            datetime.strptime(row.game_date, "%m/%d/%Y").date().isoformat(),
            row.away_team, row.home_team,
        )
        game_id = matched.get(key)
        if game_id is not None:
            by_game[game_id].append(row)

    rows: list[dict[str, Any]] = []
    for game in games.itertuples():
        observations = by_game.get(game.nba_game_id)
        tipoff = pd.Timestamp(game.game_datetime_utc).to_pydatetime()
        anchor = tipoff - timedelta(minutes=ANCHOR_MINUTES)
        base = {
            "nba_game_id": game.nba_game_id,
            "game_datetime_utc": tipoff,
            "prediction_ts_utc": anchor,
            "home_team": game.home_team,
            "away_team": game.away_team,
            "t30_state_available": False,
            "selected_report_timestamp_utc": None,
            "report_age_minutes": None,
            "selected_report_filename": None,
            "players_reported": 0, "players_resolved": 0, "players_unresolved": 0,
            "n_out": 0, "n_doubtful": 0, "n_questionable": 0,
            "n_probable": 0, "n_available": 0, "n_unknown_status": 0,
            # No report survives for this anchor, so nothing is known about
            # who filed. False would assert a pending filing we never observed.
            "teams_not_submitted": None,
            "both_teams_submitted": None,
            "coverage_quality": "no_surviving_report",
        }
        if not observations:
            rows.append(base)
            continue

        # The single chokepoint: only reports published at or before the anchor.
        eligible = [
            o for o in observations
            if pd.Timestamp(o.report_timestamp_utc).to_pydatetime() <= anchor
        ]
        if not eligible:
            base["coverage_quality"] = "first_report_after_anchor"
            rows.append(base)
            continue

        latest = max(pd.Timestamp(o.report_timestamp_utc) for o in eligible)
        selected = [o for o in eligible if pd.Timestamp(o.report_timestamp_utc) == latest]
        counts = Counter(o.status_normalized for o in selected)
        resolved = sum(1 for o in selected if o.balldontlie_player_id is not None)
        age = (anchor - latest.to_pydatetime()).total_seconds() / 60.0
        filename = selected[0].source_filename
        game_key = (
            pd.Timestamp(tipoff).tz_convert(EASTERN).date().isoformat(),
            game.away_team,
            game.home_team,
        )
        pending = [
            code for code in (game.home_team, game.away_team)
            if (filename, *game_key, code) in (not_submitted or set())
        ]
        base.update(
            {
                "t30_state_available": True,
                "selected_report_timestamp_utc": latest.to_pydatetime(),
                "report_age_minutes": age,
                "selected_report_filename": selected[0].source_filename,
                "players_reported": len(selected),
                "players_resolved": resolved,
                "players_unresolved": len(selected) - resolved,
                "n_out": counts.get("out", 0),
                "n_doubtful": counts.get("doubtful", 0),
                "n_questionable": counts.get("questionable", 0),
                "n_probable": counts.get("probable", 0),
                "n_available": counts.get("available", 0),
                "n_unknown_status": counts.get("unknown", 0),
                "teams_not_submitted": ",".join(sorted(pending)),
                "both_teams_submitted": not pending,
                "coverage_quality": (
                    "team_not_yet_submitted" if pending
                    else "ok" if age <= 60
                    else "stale_report"
                ),
            }
        )
        rows.append(base)
    frame = pd.DataFrame(rows, columns=T30_COLUMNS)
    # Nullable so "no report survived" stays distinct from "a team was pending".
    frame["both_teams_submitted"] = frame["both_teams_submitted"].astype("boolean")
    return frame


def run_pipeline(*, settings: Settings | None = None) -> dict[str, Any]:
    settings = settings or load_settings()
    settings.paths.ensure()
    processed = settings.paths.processed

    games_path = processed / "nba_regular_season_games_2006_26.parquet"
    players_path = processed / "nba_player_game_stats_2006_26.parquet"
    for path in (games_path, players_path):
        if not path.is_file():
            raise ConfigError(f"Missing {path}. Run the earlier phases first.")

    games = pd.read_parquet(games_path)
    games = games[(games["modeling_eligible"]) & (games["season"] == 2025)].copy()
    games["game_datetime_utc"] = pd.to_datetime(games["game_datetime_utc"], utc=True)

    players = pd.read_parquet(players_path)
    players = players[players["season"] == 2025][
        ["player_id", "player_name", "team_id"]
    ].drop_duplicates()
    team_ids = {
        row.abbreviation: row.source_team_id
        for row in _franchise_rows()
    }

    archive = ReportArchive(settings.paths.root / "raw" / "availability" / "nba_official")
    inventory = build_inventory(archive.archived_slots())
    reports, failures = parse_archive(archive)
    if not reports:
        raise ConfigError("no parseable reports in the archive; run the archiver first")

    registry = build_registry(players)
    events, identity = events_frame(reports, registry, team_ids)
    matched, game_matching = match_reports_to_games(events, games)
    not_submitted = not_submitted_index(reports)
    t30 = build_t30_states(events, games, matched, not_submitted)

    written: list[Path] = []
    for frame, stem in (
        (events, "nba_official_availability_events_2025_26"),
        (t30, "nba_game_availability_t30_partial_2025_26"),
    ):
        path = processed / f"{stem}.parquet"
        frame.to_parquet(path, index=False)
        written.append(path)

    covered = t30[t30["t30_state_available"]]
    ages = covered["report_age_minutes"].dropna()
    report = {
        "generated_at_utc": utc_now().isoformat(),
        "IMPORTANT": (
            "PARTIAL COVERAGE. The NBA CDN retains roughly eight months of "
            "reports, so this covers only the surviving tail of 2025-26 -- never "
            "the full season. A game without a surviving report is not a game "
            "without injuries."
        ),
        "archive": {
            "reports_parsed": len(reports),
            "parse_failures": len(failures),
            "parse_failure_examples": failures[:10],
            "earliest_report_utc": reports[0].report_timestamp_utc.isoformat(),
            "latest_report_utc": reports[-1].report_timestamp_utc.isoformat(),
            "layout_variants": sorted({r.layout_variant for r in reports}),
            "column_offset_drift": _offset_drift(reports),
            "reports_with_warnings": sum(1 for r in reports if r.warnings),
            "total_entries": len(events),
        },
        "archive_inventory": inventory.to_dict(),
        "outstanding_filings": {
            "reports_with_a_pending_team": sum(
                1 for r in reports if r.teams_not_submitted
            ),
            "distinct_game_team_pairs": len(not_submitted),
            "note": (
                "A team absent from a report has unknown availability, not a "
                "clean bill of health. These are counted at the report level; "
                "the T-30 view below only flags a game when its own team was "
                "still pending in the report selected for that anchor."
            ),
        },
        "player_identity": {
            **identity,
            "unresolved_names": _unresolved_names(events),
            "unresolved_policy": (
                "Every remaining miss is a nickname or legal-name difference "
                "(e.g. the report's 'Bub' Carrington against the roster's "
                "legal name). These are listed rather than fuzzy-matched: "
                "resolving them needs an explicitly verified alias per player, "
                "and a silent name match would be exactly the kind of guess "
                "this project refuses to make."
            ),
        },
        "game_matching": game_matching,
        "t30_coverage": {
            "regular_season_games_2025_26": len(t30),
            "with_valid_pre_anchor_state": len(covered),
            "without_surviving_report": int(
                (t30["coverage_quality"] == "no_surviving_report").sum()
            ),
            "first_report_after_anchor": int(
                (t30["coverage_quality"] == "first_report_after_anchor").sum()
            ),
            "coverage_pct": round(100.0 * len(covered) / len(t30), 3) if len(t30) else None,
            "report_age_minutes": {
                "min": float(ages.min()) if len(ages) else None,
                "p50": float(ages.quantile(0.5)) if len(ages) else None,
                "p95": float(ages.quantile(0.95)) if len(ages) else None,
                "max": float(ages.max()) if len(ages) else None,
            },
            "quality_breakdown": {
                str(k): int(v) for k, v in t30["coverage_quality"].value_counts().items()
            },
        },
        "status_distribution": {
            str(k): int(v) for k, v in events["status_normalized"].value_counts().items()
        },
        "leakage_guarantee": (
            "Every selected report satisfies report_timestamp <= prediction_ts. "
            "Games whose earliest surviving report postdates the anchor are marked "
            "unavailable and never backfilled."
        ),
    }
    report_path = settings.paths.reports / "nba_official_archive_salvage.json"
    report_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    written.append(report_path)
    report["written_files"] = [str(p) for p in written]
    return report


def _unresolved_names(events: pd.DataFrame) -> list[dict[str, Any]]:
    """Every player the registry could not resolve, with row counts."""
    missing = events[events["balldontlie_player_id"].isna()]
    if missing.empty:
        return []
    grouped = missing.groupby(["player_name", "team_code"], dropna=False).size()
    return [
        {"player_name": name, "team_code": team, "rows": int(count)}
        for (name, team), count in grouped.sort_values(ascending=False).items()
    ]


def _offset_drift(reports: list[ParsedReport]) -> dict[str, Any]:
    """Observed range of each header column offset across the archive.

    Offsets move a few characters between reports because the PDF uses a
    proportional font; each report is parsed against its own header, so this is
    reported to show the drift is bounded, not to gate anything.
    """
    columns = [r.column_offsets for r in reports if r.column_offsets]
    if not columns:
        return {}
    widths = {len(c) for c in columns}
    if len(widths) != 1:
        return {"inconsistent_column_counts": sorted(widths)}
    return {
        "column_count": widths.pop(),
        "per_column_min": [min(c[i] for c in columns) for i in range(len(columns[0]))],
        "per_column_max": [max(c[i] for c in columns) for i in range(len(columns[0]))],
        "note": (
            "Offsets are read per report from its own header row, so drift "
            "within this range does not affect parsing."
        ),
    }


def _franchise_rows():
    from nba_prediction_market.matching.franchises import FRANCHISES

    return list(FRANCHISES.values())


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m nba_prediction_market.pipelines.build_availability_salvage",
        description="Phase 3A3B1: parse the surviving archive and build T-30 states.",
    )
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level))
    try:
        report = run_pipeline(settings=load_settings(args.data_dir))
    except ConfigError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2

    archive, coverage = report["archive"], report["t30_coverage"]
    print()
    print(f"Archive: {archive['reports_parsed']} reports parsed, "
          f"{archive['parse_failures']} failures")
    print(f"  {archive['earliest_report_utc']} .. {archive['latest_report_utc']}")
    print(f"  entries: {archive['total_entries']:,}")
    print()
    print(f"Player identity: {report['player_identity']['resolved']:,} resolved / "
          f"{report['player_identity']['total']:,}")
    print(f"Game matching  : {report['game_matching']['matched']} matched, "
          f"{report['game_matching']['ambiguous']} ambiguous, "
          f"{report['game_matching']['unmatched']} unmatched")
    print()
    print(f"T-30 coverage (PARTIAL): {coverage['with_valid_pre_anchor_state']} of "
          f"{coverage['regular_season_games_2025_26']} 2025-26 games "
          f"({coverage['coverage_pct']}%)")
    age = coverage["report_age_minutes"]
    if age["p50"] is not None:
        print(f"  report age at anchor: median {age['p50']:.0f}min, "
              f"p95 {age['p95']:.0f}min, max {age['max']:.0f}min")
    print()
    print("Files written:")
    for path in report["written_files"]:
        print(f"  {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
