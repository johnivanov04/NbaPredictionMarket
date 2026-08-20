"""Phase 3A0: historical NBA ingestion and audit across 20 seasons.

Builds a lookahead-safe regular-season dataset for 2006-07..2025-26 and a set of
audits that make its trustworthiness checkable rather than assumed. No model
features, no Kalshi data, no forecasting.

Run with::

    python -m nba_prediction_market.pipelines.build_history --seasons 2006-2025

Season downloads are cached one file per season and never refetched unless
``--refresh`` is passed, so a run can be interrupted and resumed.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from nba_prediction_market.clients.balldontlie import BallDontLieClient
from nba_prediction_market.config import (
    BALLDONTLIE_PER_PAGE,
    ConfigError,
    Settings,
    load_settings,
)
from nba_prediction_market.ingestion.game_phase import (
    GAME_PHASES,
    PHASE_REGULAR_SEASON,
    PHASE_UNCLASSIFIED,
    verify_regular_season,
)
from nba_prediction_market.ingestion.nba_games import normalize_game
from nba_prediction_market.ingestion.raw_store import utc_now
from nba_prediction_market.ingestion.season_cache import SeasonCache, SeasonRequest
from nba_prediction_market.ingestion.season_metadata import (
    HISTORICAL_SEASONS,
    SEASON_METADATA,
    season_has_nba_cup,
    season_has_play_in,
    season_info,
)
from nba_prediction_market.ingestion.source_corrections import (
    SOURCE_CORRECTIONS,
    apply_corrections,
    eligibility,
)
from nba_prediction_market.matching.franchises import (
    canonical_abbreviation_for_source_id,
    is_nba_franchise,
)

logger = logging.getLogger(__name__)

REPORT_EXAMPLE_LIMIT = 5

HISTORY_COLUMNS: list[str] = [
    "nba_game_id",
    "season",
    "season_label",
    "date",
    "source_game_datetime_utc",
    "game_datetime_utc",
    "datetime_corrected",
    "chronology_precision",
    "tipoff_date_matches_scheduled_date",
    "home_team_source_id",
    "away_team_source_id",
    "home_franchise_id",
    "away_franchise_id",
    "home_team",
    "away_team",
    "home_team_full_name",
    "away_team_full_name",
    "source_home_score",
    "source_away_score",
    "home_score",
    "away_score",
    "score_corrected",
    "home_win",
    "modeling_eligible",
    "exclusion_reason",
    "postseason",
    "ist_stage",
    "game_phase",
    "status",
    "is_final",
]


@dataclass
class HistoryResult:
    """Everything a Phase 3A0 run produced."""

    all_games: pd.DataFrame
    regular_season: pd.DataFrame
    identity: pd.DataFrame
    report: dict[str, Any]
    written_files: list[Path]


# --- ingestion -------------------------------------------------------------


def fetch_season(
    client: BallDontLieClient,
    cache: SeasonCache,
    season: int,
    *,
    per_page: int = BALLDONTLIE_PER_PAGE,
    refresh: bool = False,
) -> list[dict[str, Any]]:
    """Return every raw game for ``season``, using the cache when valid."""
    request = SeasonRequest(season=season, per_page=per_page)
    if not refresh:
        cached = cache.load(request)
        if cached is not None:
            return [g for page in cached for g in (page.get("data") or [])]

    pages: list[dict[str, Any]] = []
    for _ in client.iter_games(season, per_page=per_page, on_page=lambda n, p: pages.append(p)):
        pass
    games = [g for page in pages for g in (page.get("data") or [])]
    if not games:
        raise ConfigError(
            f"Season {season} returned no games; refusing to cache an empty season."
        )
    cache.store(request, pages)
    return games


# --- normalization ---------------------------------------------------------


def build_history_frame(raw_by_season: dict[int, list[dict[str, Any]]]) -> pd.DataFrame:
    """Normalize every raw game across seasons into one all-phases table.

    Franchise id is the source team id -- BALLDONTLIE already presents historical
    franchises under their present-day id, so no relocation mapping exists to
    apply. It is ``None`` for a non-NBA opponent, which is what makes exhibition
    games visible rather than silently counted.
    """
    rows: list[dict[str, Any]] = []
    for season in sorted(raw_by_season):
        for raw in raw_by_season[season]:
            base = normalize_game(raw)
            home_id = base["home_team_id"]
            away_id = base["visitor_team_id"]
            row = {
                    "nba_game_id": base["source_game_id"],
                    "season": base["season"],
                    "season_label": base["season_label"],
                    "date": base["game_date"],
                    # Raw source value, retained verbatim for auditability.
                    "source_game_datetime_utc": base["tipoff_utc"],
                    "game_datetime_utc": base["tipoff_utc"],
                    "datetime_corrected": False,
                    "chronology_precision": None,
                    "tipoff_date_matches_scheduled_date": None,
                    "home_team_source_id": home_id,
                    "away_team_source_id": away_id,
                    "home_franchise_id": home_id if is_nba_franchise(home_id) else None,
                    "away_franchise_id": away_id if is_nba_franchise(away_id) else None,
                    "home_team": base["home_team_code"]
                    or canonical_abbreviation_for_source_id(home_id),
                    "away_team": base["visitor_team_code"]
                    or canonical_abbreviation_for_source_id(away_id),
                    "home_team_full_name": base["home_team_full_name"],
                    "away_team_full_name": base["visitor_team_full_name"],
                    "source_home_score": base["home_score"],
                    "source_away_score": base["visitor_score"],
                    "home_score": base["home_score"],
                    "away_score": base["visitor_score"],
                    "score_corrected": False,
                    "home_win": base["home_win"],
                    "modeling_eligible": False,
                    "exclusion_reason": None,
                    "postseason": base["postseason"],
                    "ist_stage": base["ist_stage"],
                    "game_phase": base["game_phase"],
                    "status": base["status"],
                    "is_final": base["is_final"],
            }

            # Corrections are applied here, never to the raw files. Guards inside
            # apply_corrections fail loudly if the source no longer matches.
            outcome = apply_corrections(row)
            row["datetime_corrected"] = outcome.datetime_corrected
            row["score_corrected"] = outcome.score_corrected
            row["chronology_precision"] = outcome.chronology_precision
            row["tipoff_date_matches_scheduled_date"] = _tipoff_matches_date(
                row["game_datetime_utc"], row["date"]
            )
            eligible, reason = eligibility(row)
            row["modeling_eligible"] = eligible
            row["exclusion_reason"] = reason
            rows.append(row)

    frame = pd.DataFrame(rows, columns=HISTORY_COLUMNS)
    return frame.sort_values(
        ["season", "date", "nba_game_id"], kind="stable", na_position="last"
    ).reset_index(drop=True)


# --- audits ----------------------------------------------------------------


def _tipoff_matches_date(tipoff: Any, game_date: Any) -> bool | None:
    """Whether the tipoff instant is consistent with the scheduled ``date``.

    A US evening game tips the following UTC day, so 0 or 1 day of separation is
    normal. Anything larger means ``date`` still holds the *original* schedule
    for a game that was actually played later -- see ``audit_chronology``.
    """
    if tipoff is None or game_date is None:
        return None
    try:
        stamp = pd.Timestamp(tipoff)
    except (TypeError, ValueError):
        return None
    if pd.isna(stamp):
        return None
    scheduled = _as_date(game_date)
    if scheduled is None:
        return None
    return abs((stamp.date() - scheduled).days) <= 1


def _as_date(value: Any) -> date | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    if isinstance(value, date) and not isinstance(value, datetime):
        return value
    try:
        stamp = pd.Timestamp(value)
    except (TypeError, ValueError):
        return None
    return None if pd.isna(stamp) else stamp.date()


def audit_season(frame: pd.DataFrame, season: int) -> dict[str, Any]:
    """Per-season counts, structural validation, and data-quality findings."""
    season_frame = frame[frame["season"] == season]
    info = season_info(season)
    records = [
        {
            "game_phase": r["game_phase"],
            "home_team_code": r["home_team"],
            "visitor_team_code": r["away_team"],
        }
        for r in season_frame.to_dict("records")
    ]
    structural = verify_regular_season(records, season)

    regular = season_frame[season_frame["game_phase"] == PHASE_REGULAR_SEASON]
    per_team = Counter()
    for r in regular.to_dict("records"):
        for key in ("home_team", "away_team"):
            if r[key]:
                per_team[r[key]] += 1

    phase_counts = season_frame["game_phase"].value_counts().to_dict()
    dupes = season_frame["nba_game_id"][season_frame["nba_game_id"].duplicated()].tolist()

    final_frame = season_frame[season_frame["is_final"].fillna(False)]
    missing_scores = final_frame[
        final_frame["home_score"].isna() | final_frame["away_score"].isna()
    ]
    ties = final_frame[
        final_frame["home_score"].notna()
        & (final_frame["home_score"] == final_frame["away_score"])
    ]
    missing_datetime = season_frame[season_frame["game_datetime_utc"].isna()]

    reg_dates = [d for d in (_as_date(x) for x in regular["date"]) if d is not None]
    outside_window: list[dict[str, Any]] = []
    if info is not None:
        for r in regular.to_dict("records"):
            d = _as_date(r["date"])
            if d is not None and not (
                info.regular_season_start <= d <= info.regular_season_end
            ):
                outside_window.append({"nba_game_id": r["nba_game_id"], "date": str(d)})

    names = {
        (int(r["home_team_source_id"]), str(r["home_team_full_name"]), str(r["home_team"]))
        for r in season_frame.to_dict("records")
        if r["home_team_source_id"] is not None and not pd.isna(r["home_team_source_id"])
    }
    non_franchise = sorted(
        {sid for sid, _, _ in names if not is_nba_franchise(sid)}
    )

    return {
        "season": season,
        "season_label": f"{season}-{(season + 1) % 100:02d}",
        "structure": info.structure if info else None,
        "unusual_reason": info.unusual_reason if info else None,
        "notes": info.notes if info else "",
        "raw_games_returned": len(season_frame),
        "phase_counts": {str(k): int(v) for k, v in phase_counts.items()},
        "regular_season_games": len(regular),
        "expected_regular_season_games": structural["expected_regular_season_games"],
        "playoff_games": int(phase_counts.get("playoffs", 0)),
        "play_in_games": int(phase_counts.get("play_in", 0)),
        "nba_cup_championship_games": int(phase_counts.get("nba_cup_championship", 0)),
        "other_special_games": int(phase_counts.get("other_special", 0)),
        "unclassified_games": int(phase_counts.get("unclassified", 0)),
        "teams": len(per_team),
        "games_per_team": {
            "expected": structural["games_per_team_expected"],
            "min": min(per_team.values()) if per_team else None,
            "max": max(per_team.values()) if per_team else None,
            "distribution": dict(sorted(Counter(per_team.values()).items())),
            "teams_with_unexpected_count": structural["teams_with_unexpected_game_count"],
        },
        "earliest_regular_season_date": str(min(reg_dates)) if reg_dates else None,
        "latest_regular_season_date": str(max(reg_dates)) if reg_dates else None,
        "era": {
            "play_in_expected": season_has_play_in(season),
            "nba_cup_expected": season_has_nba_cup(season),
        },
        "data_quality": {
            "duplicate_game_ids": [int(x) for x in dupes],
            "final_games_missing_scores": len(missing_scores),
            "tied_final_scores": len(ties),
            "missing_datetimes": len(missing_datetime),
            "regular_season_games_outside_declared_window": outside_window,
            "non_franchise_team_ids": non_franchise,
        },
        "validation": structural,
        "validation_status": "pass" if structural["verified"] else "review",
    }


def audit_identity(frame: pd.DataFrame) -> pd.DataFrame:
    """Every distinct (source id, abbreviation, full name) combination observed."""
    seen: dict[tuple[int, str, str], dict[str, Any]] = {}
    for side in ("home", "away"):
        sub = frame[
            [f"{side}_team_source_id", f"{side}_team", f"{side}_team_full_name", "season"]
        ].dropna(subset=[f"{side}_team_source_id"])
        for r in sub.to_dict("records"):
            key = (
                int(r[f"{side}_team_source_id"]),
                str(r[f"{side}_team"]),
                str(r[f"{side}_team_full_name"]),
            )
            entry = seen.setdefault(
                key,
                {
                    "source_team_id": key[0],
                    "historical_abbreviation": key[1],
                    "historical_full_name": key[2],
                    "is_nba_franchise": is_nba_franchise(key[0]),
                    "canonical_franchise_id": key[0] if is_nba_franchise(key[0]) else None,
                    "canonical_abbreviation": canonical_abbreviation_for_source_id(key[0]),
                    "first_season": int(r["season"]),
                    "last_season": int(r["season"]),
                    "seasons_observed": 0,
                    "_seasons": set(),
                },
            )
            entry["first_season"] = min(entry["first_season"], int(r["season"]))
            entry["last_season"] = max(entry["last_season"], int(r["season"]))
            entry["_seasons"].add(int(r["season"]))

    rows = []
    for entry in seen.values():
        entry["seasons_observed"] = len(entry.pop("_seasons"))
        rows.append(entry)
    return pd.DataFrame(rows).sort_values(
        ["source_team_id", "first_season"], kind="stable"
    ).reset_index(drop=True)


def audit_chronology(regular: pd.DataFrame) -> dict[str, Any]:
    """Checks that sequential feature construction can rely on the ordering."""
    stamps = pd.to_datetime(regular["game_datetime_utc"], utc=True, errors="coerce")
    has_stamp = stamps.notna()

    per_season = {}
    for season, sub in regular.groupby("season"):
        s = pd.to_datetime(sub["game_datetime_utc"], utc=True, errors="coerce")
        per_season[str(int(season))] = {
            "games": len(sub),
            "with_datetime": int(s.notna().sum()),
            "missing_datetime": int(s.isna().sum()),
            "distinct_timestamps": int(s.dropna().nunique()),
            "max_games_sharing_one_timestamp": (
                int(s.dropna().value_counts().max()) if s.notna().any() else 0
            ),
        }

    # A team appearing twice at the exact same instant would be impossible.
    impossible: list[dict[str, Any]] = []
    by_team_ts: dict[tuple[str, Any], int] = defaultdict(int)
    for r in regular.to_dict("records"):
        ts = r["game_datetime_utc"]
        if ts is None or pd.isna(ts):
            continue
        for key in ("home_team", "away_team"):
            by_team_ts[(r[key], ts)] += 1
    for (team, ts), n in by_team_ts.items():
        if n > 1:
            impossible.append({"team": team, "timestamp": str(ts), "games": n})

    # The same impossibility check against the `date` column, to show which
    # field can be trusted for ordering.
    by_team_date: dict[tuple[str, Any], int] = defaultdict(int)
    for r in regular.to_dict("records"):
        day = r["date"]
        if day is None or pd.isna(day):
            continue
        for key in ("home_team", "away_team"):
            by_team_date[(r[key], day)] += 1
    impossible_by_date = [
        {"team": team, "date": str(day), "games": n}
        for (team, day), n in by_team_date.items()
        if n > 1
    ]

    diverged = regular[regular["tipoff_date_matches_scheduled_date"] == False]  # noqa: E712

    same_date_counts = regular.groupby(["season", "date"]).size()
    return {
        "timezone_aware": bool(stamps.dropna().dt.tz is not None) if has_stamp.any() else None,
        "total_games": len(regular),
        "with_datetime": int(has_stamp.sum()),
        "missing_datetime": int((~has_stamp).sum()),
        "missing_datetime_pct": round(100.0 * (~has_stamp).sum() / len(regular), 4)
        if len(regular)
        else None,
        "distinct_timestamps": int(stamps.dropna().nunique()),
        "games_sharing_a_timestamp": int(
            len(stamps.dropna()) - stamps.dropna().nunique()
        ),
        "max_games_at_one_timestamp": int(stamps.dropna().value_counts().max())
        if has_stamp.any()
        else 0,
        "max_games_on_one_date": int(same_date_counts.max()) if len(same_date_counts) else 0,
        "same_team_twice_at_one_timestamp": impossible,
        "same_team_twice_on_one_scheduled_date": impossible_by_date,
        "ordering_field": "game_datetime_utc",
        "ordering_note": (
            "`date` is the originally scheduled date and was not updated for "
            "postponed games; ordering by it produces impossible same-day repeats. "
            "`game_datetime_utc` is the played instant and must be the sort key."
        ),
        "tipoff_diverges_from_scheduled_date": len(diverged),
        "tipoff_divergence_by_season": {
            str(int(k)): int(v) for k, v in diverged.groupby("season").size().to_dict().items()
        },
        "datetime_before_date_minus_one_day": int(
            sum(
                1
                for r in regular.to_dict("records")
                if r["game_datetime_utc"] is not None
                and not pd.isna(r["game_datetime_utc"])
                and _as_date(r["date"]) is not None
                and abs((pd.Timestamp(r["game_datetime_utc"]).date() - _as_date(r["date"])).days)
                > 1
            )
        ),
        "per_season": per_season,
    }


def build_report(
    all_games: pd.DataFrame,
    regular: pd.DataFrame,
    identity: pd.DataFrame,
    *,
    seasons: list[int],
    cache_stats: dict[str, int],
    started_at: str,
) -> dict[str, Any]:
    """Assemble the historical audit report."""
    season_reports = [audit_season(all_games, s) for s in seasons]
    by_status: dict[str, list[int]] = defaultdict(list)
    for r in season_reports:
        by_status[r["validation_status"]].append(r["season"])

    structures: dict[str, list[int]] = defaultdict(list)
    for r in season_reports:
        structures[str(r["structure"])].append(r["season"])

    dup_ids = all_games["nba_game_id"][all_games["nba_game_id"].duplicated()].tolist()
    matchup_key = (
        all_games["date"].astype(str)
        + "|"
        + all_games["home_team"].astype(str)
        + "|"
        + all_games["away_team"].astype(str)
    )
    dup_matchups = matchup_key[matchup_key.duplicated()].tolist()

    return {
        "generated_at_utc": utc_now().isoformat(),
        "started_at_utc": started_at,
        "seasons": seasons,
        "season_range": f"{min(seasons)}-{max(seasons)}",
        "totals": {
            "all_games": len(all_games),
            "regular_season_games": len(regular),
            "playoff_games": int((all_games["game_phase"] == "playoffs").sum()),
            "play_in_games": int((all_games["game_phase"] == "play_in").sum()),
            "nba_cup_championship_games": int(
                (all_games["game_phase"] == "nba_cup_championship").sum()
            ),
            "other_special_games": int((all_games["game_phase"] == "other_special").sum()),
            "unclassified_games": int((all_games["game_phase"] == PHASE_UNCLASSIFIED).sum()),
            "distinct_franchises": int(regular["home_franchise_id"].nunique()),
        },
        "season_structures": {k: sorted(v) for k, v in structures.items()},
        "validation_status_counts": {k: sorted(v) for k, v in by_status.items()},
        "seasons_detail": season_reports,
        "chronology": audit_chronology(regular),
        "identity": {
            "distinct_combinations": len(identity),
            "franchises_observed": int(identity["is_nba_franchise"].sum()),
            "non_franchise_ids": sorted(
                {
                    int(r["source_team_id"])
                    for r in identity.to_dict("records")
                    if not r["is_nba_franchise"]
                }
            ),
            "ids_with_multiple_labels": sorted(
                {
                    int(sid)
                    for sid, n in Counter(identity["source_team_id"]).items()
                    if n > 1
                }
            ),
            "note": (
                "BALLDONTLIE returns present-day identity for every era, so an id "
                "with multiple labels would indicate a mid-history rename in the "
                "source. None observed means franchise identity is stable."
            ),
        },
        "data_quality": {
            "duplicate_game_ids": [int(x) for x in dup_ids],
            "duplicate_date_matchups": dup_matchups[:REPORT_EXAMPLE_LIMIT],
            "duplicate_date_matchup_count": len(dup_matchups),
            "regular_season_games_with_missing_scores": int(
                regular["home_score"].isna().sum() + regular["away_score"].isna().sum()
            ),
            "regular_season_non_final_games": int((~regular["is_final"].fillna(False)).sum()),
            "regular_season_ties": int(
                (
                    regular["home_score"].notna()
                    & (regular["home_score"] == regular["away_score"])
                ).sum()
            ),
            "unclassified_in_modelling_dataset": int(
                (regular["game_phase"] != PHASE_REGULAR_SEASON).sum()
            ),
            "phases_present": sorted({str(p) for p in all_games["game_phase"].unique()}),
            "known_phases": list(GAME_PHASES),
        },
        "corrections": {
            "declared": len(SOURCE_CORRECTIONS),
            "games_corrected": len({c.nba_game_id for c in SOURCE_CORRECTIONS}),
            "datetime_corrections_applied": int(
                regular["datetime_corrected"].fillna(False).sum()
            ),
            "score_corrections_applied": int(regular["score_corrected"].fillna(False).sum()),
            "chronology_precision": {
                str(k): int(v)
                for k, v in regular["chronology_precision"].value_counts().to_dict().items()
            },
            "detail": [c.to_dict() for c in SOURCE_CORRECTIONS],
        },
        "modeling_eligibility": {
            "regular_season_rows": len(regular),
            "eligible": int(regular["modeling_eligible"].fillna(False).sum()),
            "ineligible": int((~regular["modeling_eligible"].fillna(False)).sum()),
            "exclusion_reasons": {
                str(k): int(v)
                for k, v in regular["exclusion_reason"].value_counts().to_dict().items()
            },
            "ineligible_rows": [
                {
                    "nba_game_id": int(r["nba_game_id"]),
                    "season": int(r["season"]),
                    "date": str(r["date"]),
                    "matchup": f"{r['away_team']}@{r['home_team']}",
                    "exclusion_reason": r["exclusion_reason"],
                }
                for r in regular[~regular["modeling_eligible"].fillna(False)].to_dict("records")
            ],
        },
        "cache": cache_stats,
    }


# --- orchestration ---------------------------------------------------------


def run_pipeline(
    seasons: list[int],
    *,
    settings: Settings | None = None,
    refresh: bool = False,
    write_csv: bool = True,
    client: BallDontLieClient | None = None,
) -> HistoryResult:
    """Ingest every season, build the datasets, and write the audit report."""
    settings = settings or load_settings()
    settings.paths.ensure()
    started_at = utc_now().isoformat()

    undeclared = [s for s in seasons if s not in SEASON_METADATA]
    if undeclared:
        raise ConfigError(
            f"Seasons {undeclared} have no declared metadata. Add SeasonInfo entries to "
            "SEASON_METADATA in ingestion/season_metadata.py -- a season is never "
            "classified by guessing."
        )

    cache = SeasonCache(settings.paths.raw_nba / "seasons")
    owns_client = client is None
    api = client or BallDontLieClient(
        settings.require_balldontlie_key(),
        base_url=settings.balldontlie_base_url,
        timeout=settings.request_timeout,
        min_interval=settings.balldontlie_min_interval,
        max_retries=settings.max_retries,
    )
    raw_by_season: dict[int, list[dict[str, Any]]] = {}
    try:
        for season in seasons:
            games = fetch_season(api, cache, season, refresh=refresh)
            logger.info("Season %s: %d raw games", season, len(games))
            raw_by_season[season] = games
    finally:
        if owns_client:
            api.close()

    all_games = build_history_frame(raw_by_season)
    regular = all_games[all_games["game_phase"] == PHASE_REGULAR_SEASON].reset_index(drop=True)
    identity = audit_identity(all_games)

    lo, hi = min(seasons), max(seasons)
    slug = f"{lo}_{(hi + 1) % 100:02d}"
    processed = settings.paths.processed
    written: list[Path] = []

    for frame, stem in (
        (regular, f"nba_regular_season_games_{slug}"),
        (all_games, f"nba_all_games_{slug}"),
        (identity, f"nba_team_identity_{slug}"),
    ):
        parquet = processed / f"{stem}.parquet"
        frame.to_parquet(parquet, index=False)
        written.append(parquet)
        if write_csv:
            csv_path = processed / f"{stem}.csv"
            frame.to_csv(csv_path, index=False)
            written.append(csv_path)

    report = build_report(
        all_games, regular, identity,
        seasons=seasons, cache_stats=cache.stats.to_dict(), started_at=started_at,
    )
    report_path = settings.paths.reports / f"historical_nba_{slug}_report.json"
    report_path.write_text(json.dumps(report, indent=2, default=_json_default), encoding="utf-8")
    written.append(report_path)

    return HistoryResult(
        all_games=all_games, regular_season=regular, identity=identity,
        report=report, written_files=written,
    )


def _json_default(value: Any) -> Any:
    if isinstance(value, datetime | date):
        return value.isoformat()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if hasattr(value, "item"):
        try:
            return value.item()
        except (ValueError, AttributeError):
            pass
    return str(value)


def parse_seasons(spec: str) -> list[int]:
    """Parse ``2006-2025`` or ``2006,2011,2019`` into a sorted season list."""
    out: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-", 1)
            out.update(range(int(lo), int(hi) + 1))
        else:
            out.add(int(part))
    if not out:
        raise ValueError(f"no seasons parsed from {spec!r}")
    return sorted(out)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m nba_prediction_market.pipelines.build_history",
        description=(
            "Phase 3A0: ingest and audit NBA history (2006-07..2025-26). "
            "Produces a regular-season dataset plus identity and chronology audits."
        ),
    )
    default = f"{HISTORICAL_SEASONS[0]}-{HISTORICAL_SEASONS[-1]}"
    parser.add_argument("--seasons", default=default,
                        help=f"Range or list, e.g. '2006-2025' (default: {default}).")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--refresh", action="store_true",
                        help="Ignore cached seasons and refetch every one.")
    parser.add_argument("--no-csv", action="store_true")
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
    )
    try:
        settings = load_settings(args.data_dir)
        result = run_pipeline(
            parse_seasons(args.seasons), settings=settings,
            refresh=args.refresh, write_csv=not args.no_csv,
        )
    except ConfigError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2

    totals = result.report["totals"]
    print()
    print(f"Seasons {result.report['season_range']}")
    print(f"  all games            : {totals['all_games']}")
    print(f"  regular season       : {totals['regular_season_games']}")
    print(f"  playoffs             : {totals['playoff_games']}")
    print(f"  play-in              : {totals['play_in_games']}")
    print(f"  NBA Cup final        : {totals['nba_cup_championship_games']}")
    print(f"  other special        : {totals['other_special_games']}")
    print(f"  unclassified         : {totals['unclassified_games']}")
    print()
    for label, seasons in sorted(result.report["validation_status_counts"].items()):
        print(f"  validation {label:8s}: {len(seasons)} seasons {seasons if label!='pass' else ''}")
    print()
    print("Files written:")
    for path in result.written_files:
        print(f"  {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
