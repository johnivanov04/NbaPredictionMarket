"""Ingest and refresh the forward NBA schedule the collector captures against.

Two sources, because neither covers everything:

* **BALLDONTLIE** for the regular season -- the same structured feed every
  earlier phase used, so team identity and tipoffs stay consistent with the
  frozen historical frame.
* **ESPN** for preseason, which BALLDONTLIE does not carry at all.

Refreshes are idempotent and non-destructive. A newly published game is added,
a moved tipoff is updated *and recorded*, and a game that vanishes from a source
is kept with the disappearance noted. Silently overwriting a tipoff would erase
the only evidence that a game was rescheduled.

This does not touch the Phase 3A0 historical dataset.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
import pandas as pd

from nba_prediction_market.capture.schedule import (
    ScheduledGame,
    apply_refresh,
    assess_completeness,
    diff_schedules,
)
from nba_prediction_market.config import (
    BALLDONTLIE_BASE_URL,
    ConfigError,
    Settings,
    load_settings,
)
from nba_prediction_market.ingestion.game_phase import (
    PHASE_PRESEASON,
    PHASE_REGULAR_SEASON,
)
from nba_prediction_market.ingestion.raw_store import utc_now
from nba_prediction_market.matching.team_names import resolve_team

logger = logging.getLogger(__name__)

TARGET_SEASON: int = 2026
ESPN_SCOREBOARD: str = (
    "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard"
)
#: ESPN's season type for preseason.
ESPN_PRESEASON_TYPE: int = 1
#: Preseason window to sweep. Generous on both sides; empty days cost one call.
PRESEASON_START: str = "2026-10-01"
PRESEASON_END: str = "2026-10-18"

SCHEDULE_FILE: str = "nba_forward_schedule_2026_27.parquet"
CHANGE_LOG_FILE: str = "nba_forward_schedule_changes_2026_27.parquet"


def _parse_utc(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)


def fetch_regular_season(settings: Settings, season: int) -> list[ScheduledGame]:
    """Regular-season games from BALLDONTLIE."""
    key = settings.balldontlie_api_key
    if not key:
        raise ConfigError("BALLDONTLIE_API_KEY is required to ingest the schedule")
    seen_at = utc_now()
    games: list[ScheduledGame] = []
    with httpx.Client(
        base_url=BALLDONTLIE_BASE_URL,
        headers={"Authorization": key, "Accept": "application/json"},
        timeout=60.0,
    ) as client:
        cursor: Any = None
        while True:
            params: dict[str, Any] = {"seasons[]": season, "per_page": 100}
            if cursor:
                params["cursor"] = cursor
            response = client.get("/games", params=params)
            response.raise_for_status()
            payload = response.json()
            for row in payload.get("data", []):
                tipoff = _parse_utc(row.get("datetime"))
                home = resolve_team(row["home_team"]["abbreviation"])
                away = resolve_team(row["visitor_team"]["abbreviation"])
                if tipoff is None or not (home.ok and away.ok):
                    logger.warning("skipping unusable game %s", row.get("id"))
                    continue
                games.append(ScheduledGame(
                    source_game_id=f"bdl:{row['id']}",
                    season=season,
                    phase=PHASE_REGULAR_SEASON,
                    tipoff_utc=tipoff,
                    home_team=home.abbreviation,
                    away_team=away.abbreviation,
                    source="balldontlie",
                    first_seen_at_utc=seen_at,
                ))
            cursor = payload.get("meta", {}).get("next_cursor")
            if not cursor:
                break
            time.sleep(0.12)
    return games


def fetch_preseason(
    season: int, start: str = PRESEASON_START, end: str = PRESEASON_END
) -> list[ScheduledGame]:
    """Preseason games from ESPN, which is the only source that carries them.

    A day with no games is normal, and a day that fails to fetch is logged
    rather than silently treated as empty -- the two are not the same.
    """
    seen_at = utc_now()
    games: list[ScheduledGame] = []
    failures: list[str] = []
    for day in pd.date_range(start, end, freq="D"):
        stamp = day.strftime("%Y%m%d")
        try:
            response = httpx.get(
                ESPN_SCOREBOARD,
                params={"dates": stamp, "seasontype": ESPN_PRESEASON_TYPE},
                timeout=40.0,
            )
            response.raise_for_status()
            payload = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            failures.append(f"{stamp}: {type(exc).__name__}")
            logger.warning("preseason fetch failed for %s: %s", stamp, exc)
            continue
        for event in payload.get("events", []):
            competition = event["competitions"][0]
            sides = {
                c["homeAway"]: c["team"]["abbreviation"]
                for c in competition["competitors"]
            }
            tipoff = _parse_utc(event.get("date"))
            home = resolve_team(sides.get("home", ""))
            away = resolve_team(sides.get("away", ""))
            if tipoff is None or not (home.ok and away.ok):
                continue
            games.append(ScheduledGame(
                source_game_id=f"espn:{event['id']}",
                season=season,
                phase=PHASE_PRESEASON,
                tipoff_utc=tipoff,
                home_team=home.abbreviation,
                away_team=away.abbreviation,
                source="espn",
                first_seen_at_utc=seen_at,
            ))
        time.sleep(0.35)
    if failures:
        logger.warning("%d preseason day(s) failed to fetch", len(failures))
    return games


def load_stored(settings: Settings) -> list[ScheduledGame]:
    path = settings.paths.processed / SCHEDULE_FILE
    if not path.is_file():
        return []
    frame = pd.read_parquet(path)
    return [
        ScheduledGame(
            source_game_id=row.source_game_id,
            season=int(row.season),
            phase=row.phase,
            tipoff_utc=pd.Timestamp(row.tipoff_utc).to_pydatetime(),
            home_team=row.home_team,
            away_team=row.away_team,
            source=row.source,
            first_seen_at_utc=pd.Timestamp(row.first_seen_at_utc).to_pydatetime(),
        )
        for row in frame.itertuples()
    ]


def run_pipeline(
    *,
    settings: Settings | None = None,
    season: int = TARGET_SEASON,
    include_preseason: bool = True,
) -> dict[str, Any]:
    settings = settings or load_settings()
    settings.paths.ensure()
    now = utc_now()

    fetched = fetch_regular_season(settings, season)
    preseason = fetch_preseason(season) if include_preseason else []
    fetched_all = fetched + preseason

    stored = load_stored(settings)
    changes = diff_schedules(stored, fetched_all, detected_at_utc=now)
    merged = apply_refresh(stored, fetched_all)

    frame = pd.DataFrame([g.to_dict() for g in merged])
    path = settings.paths.processed / SCHEDULE_FILE
    frame.to_parquet(path, index=False)

    change_path = settings.paths.processed / CHANGE_LOG_FILE
    change_rows = [c.to_dict() for c in changes]
    if change_rows:
        history = (
            pd.read_parquet(change_path) if change_path.is_file() else pd.DataFrame()
        )
        pd.concat([history, pd.DataFrame(change_rows)], ignore_index=True).to_parquet(
            change_path, index=False
        )

    completeness = assess_completeness(merged, season)
    regular = [g for g in merged if g.phase == PHASE_REGULAR_SEASON]
    pre = [g for g in merged if g.phase == PHASE_PRESEASON]

    tipoffs = pd.to_datetime([g.tipoff_utc for g in regular], utc=True)
    report = {
        "generated_at_utc": now.isoformat(),
        "season": season,
        "total_games": len(merged),
        "regular_season_games": len(regular),
        "preseason_games": len(pre),
        "regular_season_range": (
            [str(tipoffs.min().date()), str(tipoffs.max().date())]
            if len(tipoffs) else None
        ),
        "preseason_range": (
            [
                str(min(g.tipoff_utc for g in pre).date()),
                str(max(g.tipoff_utc for g in pre).date()),
            ] if pre else None
        ),
        "completeness": completeness.to_dict(),
        "changes_this_refresh": {
            "total": len(changes),
            "by_type": pd.Series([c.change for c in changes]).value_counts().to_dict()
            if changes else {},
            "examples": change_rows[:10],
        },
        "preseason_policy": (
            "captured for operational shakedown only: excluded from model "
            "training, regular-season evaluation, and the discovery/validation "
            "protocol"
        ),
        "written_files": [str(path)] + ([str(change_path)] if change_rows else []),
    }
    report_path = settings.paths.reports / "forward_schedule_2026_27.json"
    report_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    report["written_files"].append(str(report_path))
    return report


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Ingest/refresh the forward schedule.")
    parser.add_argument("--season", type=int, default=TARGET_SEASON)
    parser.add_argument("--skip-preseason", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s | %(message)s")
    try:
        report = run_pipeline(
            season=args.season, include_preseason=not args.skip_preseason
        )
    except ConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    print(f"\nseason {report['season']}: {report['total_games']} games")
    print(f"  regular season : {report['regular_season_games']} "
          f"{report['regular_season_range']}")
    print(f"  preseason      : {report['preseason_games']} "
          f"{report['preseason_range']}")
    c = report["completeness"]
    print(f"\ncompleteness: {c['status']}")
    print(f"  assigned {c['games_assigned']} / {c['expected_games']}, "
          f"{c['unassigned_games']} unassigned, per-team {c['games_per_team']}")
    print(f"  {c['explanation']}")
    changes = report["changes_this_refresh"]
    print(f"\nchanges this refresh: {changes['total']} {changes['by_type']}")
    for path in report["written_files"]:
        print(f"\nWrote {Path(path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
