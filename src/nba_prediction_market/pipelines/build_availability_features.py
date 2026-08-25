"""Build per-game T-30 availability features across every recovered season.

The output is deliberately *fold-independent*. Expected minutes lost is a
linear function of the per-status role-weighted minute totals::

    lost = sum over statuses of (1 - multiplier[status]) * minutes[status]

so this pipeline emits the per-status totals once and each development fold
applies its own training-derived multipliers later. That keeps a single
expensive pass over the archive from having to know anything about folds, and
makes it structurally impossible for one fold's calibration to leak into
another's features.

Three anchors are built -- T-3h, T-1h and T-30m -- all under the same as-of
rule. T-15m and T-5m are deliberately absent: they fall *after* the prediction
anchor and could not be inputs without breaking the rule the project rests on.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import defaultdict
from datetime import timedelta
from pathlib import Path
from typing import Any

import pandas as pd

from nba_prediction_market.availability.postponements import omission_for
from nba_prediction_market.availability.reason_categories import frequency_table
from nba_prediction_market.config import ConfigError, Settings, load_settings
from nba_prediction_market.features.availability_features import (
    NOT_REPORTED,
    STATUS_ORDER,
    PlayerRoleState,
    ReportedPlayer,
    status_transitions,
)
from nba_prediction_market.ingestion.raw_store import utc_now
from nba_prediction_market.matching.franchises import FRANCHISES

logger = logging.getLogger(__name__)

#: Seasons with recovered official reports. 2018-19 is excluded from modelling
#: by instruction: its coverage is partial and its cadence only three a day.
AVAILABILITY_SEASONS: tuple[int, ...] = (2019, 2020, 2021, 2022, 2023, 2024, 2025)

#: Anchors, in minutes before tipoff. T-30 is the prediction anchor; the two
#: earlier ones exist only to measure movement *into* it.
ANCHORS: dict[str, int] = {"t30": 30, "t1h": 60, "t3h": 180}
PREDICTION_ANCHOR = "t30"

SIDES = ("home", "away")


def _season_tag(season: int) -> str:
    return f"{season}_{(season + 1) % 100:02d}"


def load_events(settings: Settings, season: int) -> pd.DataFrame:
    path = (
        settings.paths.processed
        / f"nba_official_availability_events_{_season_tag(season)}.parquet"
    )
    if not path.is_file():
        raise ConfigError(
            f"Missing {path}. Run build_availability_salvage --season {season}."
        )
    events = pd.read_parquet(path)
    events["report_timestamp_utc"] = pd.to_datetime(
        events["report_timestamp_utc"], utc=True
    )
    return events


#: Minute past the hour the legacy grid publishes on, used to thin the modern
#: era down to one report per hour.
LEGACY_CADENCE_MINUTE: int = 30


def restrict_to_legacy_cadence(events: pd.DataFrame) -> pd.DataFrame:
    """Thin every era down to the legacy hourly cadence.

    From 2025-12-22 the league publishes every 30 minutes; before that, hourly.
    A model trained on hourly-era features and scored on half-hourly-era ones is
    handed fresher information than it ever saw in training, so the holdout is
    scored both ways and the difference reported.

    Era is decided by the **filename convention**, not by the report's internal
    minute. Legacy reports are stamped at :30 almost always but not invariably
    -- on 2025-12-19 the league stamped them at :45, and a handful in 2021-22
    sit at :00. Filtering on the minute would silently discard those genuinely
    legacy reports; filtering on the convention keeps every one of them and
    thins only the modern era, which is the only place extra reports exist.
    """
    legacy_named = events["source_filename"].str.count("_") == 2
    eastern = events["report_timestamp_utc"].dt.tz_convert("America/New_York")
    on_hour_grid = eastern.dt.minute == LEGACY_CADENCE_MINUTE
    return events[legacy_named | on_hour_grid]


def game_states(
    events: pd.DataFrame, games: pd.DataFrame
) -> dict[Any, dict[str, dict[str, dict[Any, str]]]]:
    """``game_id -> anchor -> team_code -> {player_id: status}``.

    For each anchor the latest report at or before it wins, and only whole
    reports are used: mixing players from different reports would invent a
    state the league never published.
    """
    by_key: dict[tuple[str, str, str], list[Any]] = defaultdict(list)
    for row in events.itertuples():
        if not (row.game_date and row.away_team and row.home_team):
            continue
        key = (
            pd.to_datetime(row.game_date, format="%m/%d/%Y").date().isoformat(),
            row.away_team,
            row.home_team,
        )
        by_key[key].append(row)

    out: dict[Any, dict[str, dict[str, dict[Any, str]]]] = {}
    for game in games.itertuples():
        tipoff = pd.Timestamp(game.game_datetime_utc)
        key = (
            tipoff.tz_convert("America/New_York").date().isoformat(),
            game.away_team,
            game.home_team,
        )
        rows = by_key.get(key)
        if not rows:
            continue
        per_anchor: dict[str, dict[str, dict[Any, str]]] = {}
        for anchor, minutes in ANCHORS.items():
            cutoff = tipoff - timedelta(minutes=minutes)
            eligible = [r for r in rows if r.report_timestamp_utc <= cutoff]
            if not eligible:
                continue
            latest = max(r.report_timestamp_utc for r in eligible)
            selected = [r for r in eligible if r.report_timestamp_utc == latest]
            teams: dict[str, dict[Any, str]] = defaultdict(dict)
            for r in selected:
                if r.team_code and r.balldontlie_player_id is not None:
                    teams[r.team_code][r.balldontlie_player_id] = r.status_normalized
            per_anchor[anchor] = {
                "_meta": {
                    "timestamp": latest,
                    "filename": selected[0].source_filename,
                    "age_minutes": (cutoff - latest).total_seconds() / 60.0,
                },
                **teams,
            }
        out[game.nba_game_id] = per_anchor
    return out


def _reported(
    state: dict[Any, str], role: PlayerRoleState
) -> list[ReportedPlayer]:
    return [
        ReportedPlayer(
            player_id=player,
            status=status,
            expected_minutes=role.expected_minutes(player),
            quality=role.player_quality(player),
        )
        for player, status in state.items()
    ]


def _side_block(players: list[ReportedPlayer], prefix: str) -> dict[str, float]:
    """Counts, role-weighted minutes and quality-minutes for one side."""
    block: dict[str, float] = {}
    for status in STATUS_ORDER:
        chosen = [p for p in players if p.status == status]
        block[f"{prefix}_{status}_count"] = float(len(chosen))
        block[f"{prefix}_{status}_expected_minutes"] = float(
            sum(p.expected_minutes or 0.0 for p in chosen)
        )
        block[f"{prefix}_{status}_expected_quality_minutes"] = float(
            sum((p.expected_minutes or 0.0) * (p.quality or 0.0) for p in chosen)
        )
    block[f"{prefix}_reported_players"] = float(len(players))
    block[f"{prefix}_role_unknown"] = float(
        sum(1 for p in players if p.expected_minutes is None)
    )
    return block


def build_rows(
    games: pd.DataFrame,
    states: dict[Any, dict[str, dict[str, dict[Any, str]]]],
    minutes_by_game: dict[Any, dict[Any, dict[Any, float]]],
    plusminus_by_game: dict[Any, dict[Any, dict[Any, float]]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Feature rows plus per-player observations, both leakage-safe.

    The observations pair each designation with what the player actually did.
    They exist solely to *calibrate* what a designation means, using training
    games only; participation never becomes a feature of its own game.
    """
    role: dict[Any, PlayerRoleState] = {}
    current_season: int | None = None
    rows: list[dict[str, Any]] = []
    observations: list[dict[str, Any]] = []

    for game in games.itertuples():
        season = int(game.season)
        if season != current_season:
            # Role weight is a within-season notion, matching Phase 3A3.
            role = {}
            current_season = season

        home_role = role.setdefault(game.home_team, PlayerRoleState())
        away_role = role.setdefault(game.away_team, PlayerRoleState())
        per_anchor = states.get(game.nba_game_id, {})

        row: dict[str, Any] = {
            "nba_game_id": game.nba_game_id,
            "season": season,
            "game_datetime_utc": game.game_datetime_utc,
            "home_team": game.home_team,
            "away_team": game.away_team,
        }

        anchor_players: dict[str, dict[str, list[ReportedPlayer]]] = {}
        for anchor in ANCHORS:
            state = per_anchor.get(anchor)
            covered = state is not None
            row[f"avail_{anchor}_covered"] = covered
            if not covered:
                continue
            meta = state["_meta"]
            if anchor == PREDICTION_ANCHOR:
                row["avail_report_timestamp_utc"] = meta["timestamp"]
                row["avail_report_filename"] = meta["filename"]
                row["avail_report_age_minutes"] = meta["age_minutes"]
            anchor_players[anchor] = {
                "home": _reported(state.get(game.home_team, {}), home_role),
                "away": _reported(state.get(game.away_team, {}), away_role),
            }

        # Per-status blocks. T-30 carries full detail; the earlier anchors only
        # need enough to measure movement into T-30.
        for anchor, sides in anchor_players.items():
            for side, players in sides.items():
                row.update(_side_block(players, f"avail_{side}_{anchor}"))

        # Late news: movement from each earlier anchor into T-30.
        t30 = anchor_players.get(PREDICTION_ANCHOR)
        for anchor in ("t3h", "t1h"):
            earlier = anchor_players.get(anchor)
            for side in SIDES:
                prefix = f"avail_{side}_{anchor}_to_t30"
                if t30 is None or earlier is None:
                    row[f"{prefix}_late_downgrades"] = None
                    row[f"{prefix}_late_upgrades"] = None
                    row[f"{prefix}_newly_out_expected_minutes"] = None
                    continue
                later_map = {p.player_id: p.status for p in t30[side]}
                earlier_map = {p.player_id: p.status for p in earlier[side]}
                weights = {p.player_id: p.expected_minutes for p in t30[side]}
                moved = status_transitions(earlier_map, later_map, weights)
                row[f"{prefix}_late_downgrades"] = moved["late_downgrades"]
                row[f"{prefix}_late_upgrades"] = moved["late_upgrades"]
                row[f"{prefix}_newly_out_expected_minutes"] = (
                    moved["newly_out_expected_minutes"]
                )

        # Calibration observations: the T-30 designation, the role weight that
        # was knowable beforehand, and the minutes that followed.
        if t30 is not None:
            for side, team in (("home", game.home_team), ("away", game.away_team)):
                actual = minutes_by_game.get(game.nba_game_id, {}).get(team, {})
                for player in t30[side]:
                    if player.expected_minutes is None:
                        continue
                    observations.append({
                        "nba_game_id": game.nba_game_id,
                        "season": season,
                        "team_code": team,
                        "side": side,
                        "player_id": player.player_id,
                        "status": player.status,
                        "baseline_minutes": player.expected_minutes,
                        "actual_minutes": float(actual.get(player.player_id, 0.0)),
                    })

        row["availability_coverage"] = bool(
            row.get(f"avail_{PREDICTION_ANCHOR}_covered", False)
        )
        tip_date = pd.Timestamp(game.game_datetime_utc).tz_convert(
            "America/New_York"
        ).date().isoformat()
        row["availability_source_omission"] = (
            omission_for(tip_date, game.away_team, game.home_team) is not None
        )
        rows.append(row)

        # Advance role state only after emitting, so the current game can never
        # inform its own features.
        for team, state_obj in ((game.home_team, home_role), (game.away_team, away_role)):
            minutes = minutes_by_game.get(game.nba_game_id, {}).get(team, {})
            plus = plusminus_by_game.get(game.nba_game_id, {}).get(team, {})
            state_obj.record_game(minutes, plus)
    return rows, observations


def player_minutes_index(
    players: pd.DataFrame,
) -> tuple[dict[Any, dict[Any, dict[Any, float]]], dict[Any, dict[Any, dict[Any, float]]]]:
    """``game_id -> team_code -> {player_id: minutes}`` and the same for +/-."""
    code_by_team_id = {v.source_team_id: v.abbreviation for v in FRANCHISES.values()}
    minutes: dict[Any, dict[Any, dict[Any, float]]] = defaultdict(lambda: defaultdict(dict))
    plus: dict[Any, dict[Any, dict[Any, float]]] = defaultdict(lambda: defaultdict(dict))
    has_pm = "plus_minus" in players.columns
    for row in players.itertuples():
        code = code_by_team_id.get(row.team_id)
        if code is None:
            continue
        value = getattr(row, "minutes", None)
        if value is not None and not pd.isna(value):
            minutes[row.nba_game_id][code][row.player_id] = float(value)
        if has_pm:
            pm = getattr(row, "plus_minus", None)
            if pm is not None and not pd.isna(pm):
                plus[row.nba_game_id][code][row.player_id] = float(pm)
    return minutes, plus


def run_pipeline(
    *, settings: Settings | None = None, cadence: str = "native"
) -> dict[str, Any]:
    settings = settings or load_settings()
    settings.paths.ensure()
    processed = settings.paths.processed

    games = pd.read_parquet(processed / "nba_regular_season_games_2006_26.parquet")
    available_seasons = [
        s for s in AVAILABILITY_SEASONS
        if (
            processed
            / f"nba_official_availability_events_{_season_tag(s)}.parquet"
        ).is_file()
    ]
    games = games[
        games["season"].isin(available_seasons) & games["modeling_eligible"]
    ].copy()
    games["game_datetime_utc"] = pd.to_datetime(games["game_datetime_utc"], utc=True)
    games = games.sort_values("game_datetime_utc", kind="stable")

    players = pd.read_parquet(processed / "nba_player_game_stats_2006_26.parquet")
    players = players[players["season"].isin(AVAILABILITY_SEASONS)]
    minutes_by_game, plus_by_game = player_minutes_index(players)

    states: dict[Any, dict[str, dict[str, dict[Any, str]]]] = {}
    per_season: list[dict[str, Any]] = []
    reason_tables: dict[int, dict[str, Any]] = {}
    skipped: list[int] = []
    for season in AVAILABILITY_SEASONS:
        season_games = games[games["season"] == season]
        try:
            events = load_events(settings, season)
        except ConfigError:
            # A season whose reports have not been salvaged yet is reported as
            # absent rather than silently producing rows with no availability.
            logger.warning("season %s has no salvaged events; skipping", season)
            skipped.append(season)
            continue
        if cadence == "harmonized":
            events = restrict_to_legacy_cadence(events)
        found = game_states(events, season_games)
        states.update(found)
        reason_tables[season] = frequency_table(events["reason_raw"].tolist())
        per_season.append(
            {
                "season": season,
                "games": len(season_games),
                "events": len(events),
                "games_with_any_state": len(found),
            }
        )
        logger.info("season %s: %d games, %d with reports", season,
                    len(season_games), len(found))

    rows, observations = build_rows(games, states, minutes_by_game, plus_by_game)
    frame = pd.DataFrame(rows)
    suffix = "" if cadence == "native" else f"_{cadence}"
    path = processed / f"nba_game_availability_features_2019_26{suffix}.parquet"
    frame.to_parquet(path, index=False)

    obs_frame = pd.DataFrame(observations)
    obs_path = (
        processed / f"nba_player_availability_observations_2019_26{suffix}.parquet"
    )
    obs_frame.to_parquet(obs_path, index=False)

    for entry in per_season:
        season_rows = frame[frame["season"] == entry["season"]]
        entry["t30_covered"] = int(season_rows["availability_coverage"].sum())
        entry["t1h_covered"] = int(season_rows["avail_t1h_covered"].sum())
        entry["t3h_covered"] = int(season_rows["avail_t3h_covered"].sum())
        ages = season_rows["avail_report_age_minutes"].dropna()
        entry["report_age_minutes"] = (
            {
                "median": float(ages.median()),
                "p95": float(ages.quantile(0.95)),
                "max": float(ages.max()),
            }
            if len(ages)
            else None
        )
        entry["uncovered_games"] = [
            int(g) for g in season_rows.loc[
                ~season_rows["availability_coverage"], "nba_game_id"
            ]
        ]

    report = {
        "generated_at_utc": utc_now().isoformat(),
        "cadence": cadence,
        "seasons": per_season,
        "seasons_skipped_no_events": skipped,
        "reason_categories_by_season": {
            str(season): table for season, table in reason_tables.items()
        },
        "rows": len(frame),
        "columns": len(frame.columns),
        "anchors_built": sorted(ANCHORS),
        "anchors_excluded": ["t15m", "t5m"],
        "anchors_excluded_reason": (
            "both fall after the T-30 prediction anchor and could not be model "
            "inputs without breaking the as-of rule"
        ),
        "not_reported_semantics": (
            f"a player absent from a report is {NOT_REPORTED}, never available; "
            "features are built from explicitly reported players only"
        ),
        "observation_rows": len(obs_frame),
        "written_files": [str(path), str(obs_path)],
    }
    report_path = (
        settings.paths.reports / f"availability_features_2019_26{suffix}.json"
    )
    report_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    report["written_files"].append(str(report_path))
    return report


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build T-30 availability features.")
    parser.add_argument(
        "--cadence", choices=["native", "harmonized"], default="native",
        help="harmonized restricts every era to the legacy hourly report grid.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s | %(message)s")
    try:
        report = run_pipeline(cadence=args.cadence)
    except ConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(f"\n{'season':8s} {'games':>6s} {'T-30':>6s} {'T-1h':>6s} {'T-3h':>6s} {'age p95':>8s}")
    for entry in report["seasons"]:
        age = entry["report_age_minutes"]
        age_text = f"{age['p95']:.0f}m" if age else "-"
        print(f"{entry['season']:8d} {entry['games']:6d} {entry['t30_covered']:6d} "
              f"{entry['t1h_covered']:6d} {entry['t3h_covered']:6d} {age_text:>8s}")
    print(f"\nrows={report['rows']} columns={report['columns']}")
    for path in report["written_files"]:
        print(f"Wrote {Path(path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
