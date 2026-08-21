"""Phase 3A3A0: paid-data capability audit and historical ingestion.

Builds a trustworthy cached foundation from the GOAT-tier feeds and reports
exactly what they contain, how far back, and which fields may legally become
model inputs. **No model is built here and nothing is merged into the Phase
3A1/3A2 feature datasets.**

Run with::

    python -m nba_prediction_market.pipelines.build_paid_data
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import date, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from nba_prediction_market.clients.balldontlie import GOAT_MIN_INTERVAL, BallDontLieClient
from nba_prediction_market.config import ConfigError, Settings, load_settings
from nba_prediction_market.ingestion.paid_cache import PaidCache, PaidRequest
from nba_prediction_market.ingestion.paid_endpoints import (
    ENDPOINT_AUDIT,
    PROHIBITED_FOR_HISTORICAL_FEATURES,
    V2_METRIC_ERAS,
)
from nba_prediction_market.ingestion.paid_stats import (
    ADVANCED_COLUMNS,
    BOUNDED_FRACTIONS,
    PLAYER_GAME_COLUMNS,
    build_frame,
)
from nba_prediction_market.ingestion.raw_store import utc_now
from nba_prediction_market.ingestion.season_metadata import HISTORICAL_SEASONS

logger = logging.getLogger(__name__)

FEEDS: dict[str, str] = {
    "player_game_stats": "/v1/stats",
    "advanced_stats_v1": "/v1/stats/advanced",
}

#: Metrics whose per-season coverage is reported in the audit matrix.
ADVANCED_METRICS: tuple[str, ...] = tuple(
    c for c in ADVANCED_COLUMNS
    if c not in {"nba_game_id", "season", "game_date", "player_id", "team_id",
                 "team_abbreviation", "is_home"}
)
PLAYER_METRICS: tuple[str, ...] = tuple(
    c for c in PLAYER_GAME_COLUMNS
    if c not in {"nba_game_id", "season", "game_date", "player_id", "player_name",
                 "team_id", "team_abbreviation", "is_home"}
)


def fetch_feed(
    client: BallDontLieClient, cache: PaidCache, feed: str, endpoint: str, season: int,
    *, refresh: bool = False,
) -> list[dict[str, Any]]:
    """Return every record for one feed-season, using the cache when valid."""
    request = PaidRequest(feed=feed, endpoint=endpoint, season=season, per_page=100)
    if not refresh:
        cached = cache.load(request)
        if cached is not None:
            return cached
    records: list[dict[str, Any]] = []
    for page in client.iter_paid_records(endpoint, season, per_page=100):
        records.extend(page)
    if not records:
        raise ConfigError(
            f"{feed} season {season} returned no records; refusing to cache an "
            "empty season."
        )
    cache.store(request, records)
    return records


# --- coverage and validation ----------------------------------------------


def coverage_matrix(frame: pd.DataFrame, metrics: tuple[str, ...]) -> dict[str, Any]:
    """Per-season non-null share for each metric, plus a stability verdict."""
    matrix: dict[str, dict[str, float]] = {}
    for season, sub in frame.groupby("season", sort=True):
        matrix[str(int(season))] = {
            metric: round(float(sub[metric].notna().mean()), 4) for metric in metrics
        }
    verdicts: dict[str, str] = {}
    for metric in metrics:
        shares = [matrix[s][metric] for s in matrix]
        if all(s >= 0.95 for s in shares):
            verdicts[metric] = "stable_all_seasons"
        elif all(s < 0.01 for s in shares):
            verdicts[metric] = "absent"
        elif shares[0] < 0.01 and shares[-1] >= 0.95:
            first = next(s for s in matrix if matrix[s][metric] >= 0.95)
            verdicts[metric] = f"modern_era_only_from_{first}"
        else:
            verdicts[metric] = "sparse_or_intermittent"
    return {"per_season_non_null_share": matrix, "verdict": verdicts}


def validate_joins(
    frame: pd.DataFrame,
    games: pd.DataFrame,
    label: str,
    all_games: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Join the feed against the trusted 24,038-game history.

    Ids outside that set are not necessarily unknown: the feeds cover playoffs,
    play-in and NBA Cup games too, which Phase 3A0 deliberately holds outside the
    regular-season modelling set. ``all_games`` lets those be attributed rather
    than reported as mysteries.
    """
    trusted = set(games["nba_game_id"])
    present = set(frame["nba_game_id"])
    covered = trusted & present
    per_season = {}
    for season, sub in games.groupby("season", sort=True):
        ids = set(sub["nba_game_id"])
        per_season[str(int(season))] = {
            "expected": len(ids),
            "with_data": len(ids & present),
            "missing": len(ids - present),
        }
    outside = present - trusted
    attributed: dict[str, int] = {}
    unattributed = outside
    if all_games is not None:
        known = all_games[all_games["nba_game_id"].isin(outside)]
        attributed = {
            str(k): int(v) for k, v in known["game_phase"].value_counts().to_dict().items()
        }
        unattributed = outside - set(known["nba_game_id"])
    return {
        "feed": label,
        "trusted_games": len(trusted),
        "games_with_data": len(covered),
        "games_without_data": len(trusted - present),
        "games_without_data_examples": sorted(trusted - present)[:10],
        "ids_outside_the_regular_season_set": len(outside),
        "ids_outside_attributed_to_phase": attributed,
        "genuinely_unknown_game_ids": len(unattributed),
        "genuinely_unknown_examples": sorted(unattributed)[:10],
        "coverage_pct": round(100.0 * len(covered) / len(trusted), 4) if trusted else None,
        "per_season": per_season,
    }


def validate_identity(frame: pd.DataFrame, games: pd.DataFrame) -> dict[str, Any]:
    """Team and player identity checks against the trusted franchise set."""
    from nba_prediction_market.matching.franchises import FRANCHISE_IDS

    team_ids = set(frame["team_id"].dropna().astype(int))
    unknown_teams = sorted(team_ids - set(FRANCHISE_IDS))
    players = frame["player_id"].dropna().astype(int)
    return {
        "distinct_teams": len(team_ids),
        "unknown_team_ids": unknown_teams,
        "distinct_players": int(players.nunique()),
        "rows_missing_player_id": int(frame["player_id"].isna().sum()),
        "rows_missing_team_id": int(frame["team_id"].isna().sum()),
        "home_away_resolved_pct": round(float(frame["is_home"].notna().mean()) * 100, 4),
    }


def consistency_checks(
    players: pd.DataFrame, advanced: pd.DataFrame, games: pd.DataFrame
) -> dict[str, Any]:
    """Sanity checks that would catch a mislabelled or corrupted feed."""
    # Player points must sum to the team's recorded score.
    totals = (
        players.groupby(["nba_game_id", "team_id"], as_index=False)["pts"].sum()
    )
    home = games[["nba_game_id", "home_team_source_id", "home_score"]].rename(
        columns={"home_team_source_id": "team_id", "home_score": "team_score"}
    )
    away = games[["nba_game_id", "away_team_source_id", "away_score"]].rename(
        columns={"away_team_source_id": "team_id", "away_score": "team_score"}
    )
    expected = pd.concat([home, away], ignore_index=True)
    merged = totals.merge(expected, on=["nba_game_id", "team_id"], how="inner")
    merged["delta"] = merged["pts"] - merged["team_score"]
    exact = int((merged["delta"] == 0).sum())

    minutes = players["minutes"].dropna()
    out_of_range = {
        column: int(
            (
                (players[column] < 0) | (players[column] > 1)
                if column in players.columns
                else pd.Series(dtype=bool)
            ).sum()
        )
        for column in BOUNDED_FRACTIONS
        if column in players.columns
    }
    advanced_out_of_range = {
        column: int(((advanced[column] < 0) | (advanced[column] > 1)).sum())
        for column in BOUNDED_FRACTIONS
        if column in advanced.columns
    }

    # The advanced feed reports 0.0, not null, for players who did not appear.
    # Averaging across all rows would therefore be badly biased toward zero, so
    # the plausibility check is made on actual participants.
    minutes_by_row = players[["nba_game_id", "player_id", "minutes"]]
    joined = advanced.merge(minutes_by_row, on=["nba_game_id", "player_id"], how="left")
    zero_pace = joined[joined["pace"] == 0.0]
    played = joined[joined["minutes"].fillna(0.0) > 0.0]
    pace = played["pace"].dropna()
    corrected = {28012, 32587, 34714, 48851}
    corrected_present = sorted(corrected & set(players["nba_game_id"]))
    corrected_sums = merged[merged["nba_game_id"].isin(corrected)]
    return {
        "team_point_totals_checked": len(merged),
        "team_point_totals_exact_match": exact,
        "team_point_totals_exact_pct": round(100.0 * exact / len(merged), 4)
        if len(merged)
        else None,
        "team_point_total_max_abs_delta": float(merged["delta"].abs().max())
        if len(merged)
        else None,
        "minutes_min": float(minutes.min()) if len(minutes) else None,
        "minutes_max": float(minutes.max()) if len(minutes) else None,
        "minutes_over_70": int((minutes > 70).sum()),
        "advanced_rows_with_zero_pace": len(zero_pace),
        "advanced_zero_pace_with_no_minutes": int(
            (zero_pace["minutes"].isna() | (zero_pace["minutes"] == 0.0)).sum()
        ),
        "advanced_zero_pace_despite_minutes": int(
            (zero_pace["minutes"].fillna(0.0) > 0.0).sum()
        ),
        "pace_checked_on_participants": len(pace),
        "pace_p01_participants": float(pace.quantile(0.01)) if len(pace) else None,
        "pace_p50_participants": float(pace.quantile(0.50)) if len(pace) else None,
        "pace_p99_participants": float(pace.quantile(0.99)) if len(pace) else None,
        "pace_above_200_participants": int((pace > 200).sum()),
        "pace_above_200_median_minutes": float(
            played[played["pace"] > 200]["minutes"].median()
        ) if int((pace > 200).sum()) else None,
        "zero_inflation_note": (
            "Advanced rows for non-participants carry 0.0 rather than null. Any "
            "Phase 3A3 aggregate must filter to minutes > 0 and weight by minutes; "
            "per-100-possession rates are unstable below roughly one minute."
        ),
        "fraction_out_of_range_player_feed": out_of_range,
        "fraction_out_of_range_advanced_feed": advanced_out_of_range,
        "corrected_4ot_games_present": corrected_present,
        "corrected_4ot_games_point_totals_match": int(
            (corrected_sums["delta"] == 0).sum()
        ),
        "corrected_4ot_games_checked": len(corrected_sums),
    }


# --- orchestration ---------------------------------------------------------


def run_pipeline(
    *,
    settings: Settings | None = None,
    seasons: list[int] | None = None,
    refresh: bool = False,
    write_csv: bool = False,
) -> dict[str, Any]:
    settings = settings or load_settings()
    settings.paths.ensure()
    started = utc_now().isoformat()
    seasons = seasons or list(HISTORICAL_SEASONS)

    games_path = settings.paths.processed / "nba_regular_season_games_2006_26.parquet"
    if not games_path.is_file():
        raise ConfigError(
            f"Missing {games_path}. Run the Phase 3A0 pipeline first."
        )
    games = pd.read_parquet(games_path)
    games = games[games["modeling_eligible"]]
    all_games_path = settings.paths.processed / "nba_all_games_2006_26.parquet"
    all_games = (
        pd.read_parquet(all_games_path) if all_games_path.is_file() else None
    )

    cache = PaidCache(settings.paths.root / "raw" / "balldontlie")
    frames: dict[str, pd.DataFrame] = {}
    normalisation: dict[str, Any] = {}

    with BallDontLieClient(
        settings.require_balldontlie_key(),
        base_url="https://api.balldontlie.io",
        min_interval=GOAT_MIN_INTERVAL,
        timeout=90,
        max_retries=6,
    ) as client:
        for feed, endpoint in FEEDS.items():
            records: list[dict[str, Any]] = []
            for season in seasons:
                records.extend(
                    fetch_feed(client, cache, feed, endpoint, season, refresh=refresh)
                )
            kind = "player_game" if feed == "player_game_stats" else "advanced"
            frame, stats = build_frame(records, kind)
            frames[feed] = frame
            normalisation[feed] = stats
            logger.info("%s: %d normalized rows", feed, len(frame))

    players, advanced = frames["player_game_stats"], frames["advanced_stats_v1"]

    written: list[Path] = []
    for frame, stem in (
        (advanced, "nba_game_advanced_stats_2006_26"),
        (players, "nba_player_game_stats_2006_26"),
    ):
        path = settings.paths.processed / f"{stem}.parquet"
        frame.to_parquet(path, index=False)
        written.append(path)
        if write_csv:
            csv_path = path.with_suffix(".csv")
            frame.to_csv(csv_path, index=False)
            written.append(csv_path)

    report = {
        "generated_at_utc": utc_now().isoformat(),
        "started_at_utc": started,
        "seasons": seasons,
        "endpoint_capability_matrix": [e.to_dict() for e in ENDPOINT_AUDIT],
        "prohibited_for_historical_features": PROHIBITED_FOR_HISTORICAL_FEATURES,
        "v2_metric_eras": V2_METRIC_ERAS,
        "normalisation": normalisation,
        "dataset_dimensions": {
            "player_game_stats": list(players.shape),
            "advanced_stats_v1": list(advanced.shape),
        },
        "coverage": {
            "player_game_stats": coverage_matrix(players, PLAYER_METRICS),
            "advanced_stats_v1": coverage_matrix(advanced, ADVANCED_METRICS),
        },
        "joins": {
            "player_game_stats": validate_joins(
                players, games, "player_game_stats", all_games
            ),
            "advanced_stats_v1": validate_joins(
                advanced, games, "advanced_stats_v1", all_games
            ),
        },
        "identity": {
            "player_game_stats": validate_identity(players, games),
            "advanced_stats_v1": validate_identity(advanced, games),
        },
        "consistency": consistency_checks(players, advanced, games),
        "cache": cache.stats.to_dict(),
    }
    report_path = settings.paths.reports / "paid_data_audit.json"
    report_path.write_text(json.dumps(report, indent=2, default=_json_default), encoding="utf-8")
    written.append(report_path)
    report["written_files"] = [str(p) for p in written]
    return report


def _json_default(value: Any) -> Any:
    if isinstance(value, datetime | date):
        return value.isoformat()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, np.bool_ | np.integer | np.floating):
        return value.item()
    if hasattr(value, "item"):
        try:
            return value.item()
        except (ValueError, AttributeError):
            pass
    return str(value)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m nba_prediction_market.pipelines.build_paid_data",
        description="Phase 3A3A0: audit and ingest GOAT-tier historical data.",
    )
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--csv", action="store_true", help="Also write CSV copies.")
    parser.add_argument(
        "--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"]
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
    )
    try:
        report = run_pipeline(
            settings=load_settings(args.data_dir), refresh=args.refresh, write_csv=args.csv
        )
    except ConfigError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2

    print()
    print("Dataset dimensions")
    for name, shape in report["dataset_dimensions"].items():
        print(f"  {name:<24s} {shape[0]:>8,} rows x {shape[1]} cols")
    print()
    print("Join coverage against 24,038 trusted games")
    for name, join in report["joins"].items():
        print(
            f"  {name:<24s} {join['games_with_data']:>6,} / {join['trusted_games']:,} "
            f"({join['coverage_pct']}%)  unattributed ids: "
            f"{join['genuinely_unknown_game_ids']}"
        )
    consistency = report["consistency"]
    print()
    print(
        f"  player points sum to team score: "
        f"{consistency['team_point_totals_exact_pct']}% exact "
        f"({consistency['team_point_totals_exact_match']:,}/"
        f"{consistency['team_point_totals_checked']:,})"
    )
    print()
    print("Files written:")
    for path in report["written_files"]:
        print(f"  {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
