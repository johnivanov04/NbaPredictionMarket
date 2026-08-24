"""Season-by-source availability coverage matrix.

Answers one question per season: what is actually known about who was available
30 minutes before tip, and how trustworthy is that knowledge?

Every cell is classified, never averaged into a single "coverage" number,
because the classes are not interchangeable:

``T30_SAFE``
    An exact report timestamp exists and a report at or before the anchor was
    selected. Freshness is reported alongside, because a safe-but-stale state
    is a materially weaker feature than a fresh one.
``EARLY_DAY_ONLY``
    A fixed early-in-the-day snapshot. Never promoted to a T-30 state.
``DATE_ONLY``
    A calendar date and nothing finer.
``UNSAFE_FINAL_STATE``
    Reflects the eventual outcome.
``UNAVAILABLE``
    Nothing usable.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import pandas as pd

from nba_prediction_market.availability.external_sources import (
    DATE_ONLY,
    EXTERNAL_SOURCES,
    T30_SAFE,
    UNAVAILABLE,
)
from nba_prediction_market.config import ConfigError, Settings, load_settings
from nba_prediction_market.ingestion.raw_store import utc_now

logger = logging.getLogger(__name__)

#: Seasons the matrix reports on.
MATRIX_SEASONS: tuple[int, ...] = (2021, 2022, 2023, 2024, 2025)

OFFICIAL_SOURCE = "nba_official_injury_report"


def _season_tag(season: int) -> str:
    return f"{season}_{(season + 1) % 100:02d}"


def official_row(season: int, settings: Settings) -> dict[str, Any]:
    """One matrix row for the official archive in one season."""
    processed = settings.paths.processed
    games_path = processed / "nba_regular_season_games_2006_26.parquet"
    if not games_path.is_file():
        raise ConfigError(f"Missing {games_path}. Run the earlier phases first.")
    games = pd.read_parquet(games_path)
    games = games[(games["season"] == season) & (games["modeling_eligible"])]

    row: dict[str, Any] = {
        "season": f"{season}-{(season + 1) % 100:02d}",
        "source": OFFICIAL_SOURCE,
        "regular_season_games": len(games),
        "games_with_any_state": 0,
        "games_t30_safe": 0,
        "games_early_day_only": 0,
        "asof_class": UNAVAILABLE,
        "timestamp_precision": "exact_timestamp",
        "player_mapping_rate": None,
        "game_mapping_rate": None,
        "report_age_minutes": None,
        "status": "not_yet_recovered",
    }

    t30_path = processed / f"nba_game_availability_t30_{_season_tag(season)}.parquet"
    legacy_path = (
        processed
        / f"nba_game_availability_t30_partial_{_season_tag(season)}.parquet"
    )
    path = t30_path if t30_path.is_file() else legacy_path
    if not path.is_file():
        return row

    t30 = pd.read_parquet(path)
    covered = t30[t30["t30_state_available"]]
    ages = covered["report_age_minutes"].dropna()
    row.update(
        {
            "games_with_any_state": len(covered),
            "games_t30_safe": len(covered),
            "asof_class": T30_SAFE if len(covered) else UNAVAILABLE,
            "game_mapping_rate": (
                round(len(covered) / len(t30), 6) if len(t30) else None
            ),
            "report_age_minutes": (
                {
                    "median": float(ages.median()),
                    "p95": float(ages.quantile(0.95)),
                    "max": float(ages.max()),
                }
                if len(ages)
                else None
            ),
            "status": "recovered",
        }
    )

    events_path = (
        processed
        / f"nba_official_availability_events_{_season_tag(season)}.parquet"
    )
    if events_path.is_file():
        events = pd.read_parquet(events_path)
        if len(events):
            resolved = int(events["balldontlie_player_id"].notna().sum())
            row["player_mapping_rate"] = round(resolved / len(events), 6)
            row["player_rows"] = len(events)
    return row


def external_rows() -> list[dict[str, Any]]:
    """Matrix rows for the audited third-party archives."""
    rows: list[dict[str, Any]] = []
    for source in EXTERNAL_SOURCES:
        rows.append(
            {
                "season": source.seasons,
                "source": source.name,
                "asof_class": source.asof_class,
                "timestamp_precision": source.timestamp_precision,
                "redistributable": source.redistributable,
                "license_status": source.license_status,
                "notes": source.notes,
                "supersededness": (
                    "the official archive covers this span directly, at finer "
                    "cadence and with full provenance, so this source is a "
                    "cross-check rather than a data dependency"
                ),
            }
        )
    return rows


def run_pipeline(*, settings: Settings | None = None) -> dict[str, Any]:
    settings = settings or load_settings()
    settings.paths.ensure()

    official = [official_row(season, settings) for season in MATRIX_SEASONS]
    recovered = [r for r in official if r["status"] == "recovered"]
    total_games = sum(r["regular_season_games"] for r in official)
    total_safe = sum(r["games_t30_safe"] for r in official)

    report = {
        "generated_at_utc": utc_now().isoformat(),
        "official_by_season": official,
        "external_sources": external_rows(),
        "totals": {
            "seasons_in_matrix": len(MATRIX_SEASONS),
            "seasons_recovered": len(recovered),
            "regular_season_games": total_games,
            "games_with_t30_safe_state": total_safe,
            "t30_safe_fraction": (
                round(total_safe / total_games, 6) if total_games else None
            ),
        },
        "class_definitions": {
            T30_SAFE: "exact timestamp, report selected at or before the anchor",
            "EARLY_DAY_ONLY": "fixed early-day snapshot; never promoted to T-30",
            DATE_ONLY: "calendar date only; cannot answer any anchor",
            "UNSAFE_FINAL_STATE": "reflects the eventual outcome",
            UNAVAILABLE: "nothing usable",
        },
    }
    path = settings.paths.reports / "availability_coverage_matrix.json"
    path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    report["written_files"] = [str(path)]
    return report


def build_arg_parser() -> argparse.ArgumentParser:
    return argparse.ArgumentParser(description="Availability coverage matrix.")


def main(argv: list[str] | None = None) -> int:
    build_arg_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    try:
        report = run_pipeline()
    except ConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    header = ("season", "source", "games", "T30-safe", "class", "age p95")
    print(f"{header[0]:9s} {header[1]:28s} {header[2]:>6s} "
          f"{header[3]:>9s} {header[4]:14s} {header[5]:>8s}")
    for row in report["official_by_season"]:
        age = row.get("report_age_minutes")
        age_text = f"{age['p95']:.0f}m" if age else "-"
        print(
            f"{row['season']:9s} {row['source']:28s} {row['regular_season_games']:6d} "
            f"{row['games_t30_safe']:9d} {row['asof_class']:14s} {age_text:>8s}"
        )
    print("\nexternal:")
    for row in report["external_sources"]:
        print(f"  {row['source']:26s} {row['season']:26s} {row['asof_class']}")
    for path in report["written_files"]:
        print(f"\nWrote {Path(path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
