"""What the collector is about to capture, and whether it can.

Run this before a slate. It puts the schedule, the four capture anchors and the
Kalshi mapping side by side so an operational mistake -- an unmapped game, a
tipoff that moved, a slate nobody noticed -- is visible before game day rather
than as a hole in the data afterwards.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import httpx
import pandas as pd

from nba_prediction_market.capture.kalshi_live import PUBLIC_REST_BASE
from nba_prediction_market.capture.market_identity import (
    map_games,
    summarise,
)
from nba_prediction_market.capture.schedule import ANCHOR_LABELS, upcoming
from nba_prediction_market.config import (
    KALSHI_NBA_SERIES_TICKER,
    ConfigError,
    Settings,
    load_settings,
)
from nba_prediction_market.ingestion.raw_store import utc_now
from nba_prediction_market.pipelines.build_forward_schedule import load_stored

logger = logging.getLogger(__name__)

EASTERN = ZoneInfo("America/New_York")
DEFAULT_HORIZON_HOURS: float = 36.0


def fetch_event_tickers(limit_pages: int = 40) -> list[str]:
    """Currently listed KXNBAGAME event tickers.

    Failure returns an empty list and says so upstream: "we could not ask" and
    "nothing is listed" look identical in the output otherwise, and only one of
    them is a problem.
    """
    tickers: list[str] = []
    try:
        with httpx.Client(timeout=40.0) as client:
            cursor: str | None = None
            for _ in range(limit_pages):
                params: dict[str, Any] = {
                    "series_ticker": KALSHI_NBA_SERIES_TICKER, "limit": 200,
                }
                if cursor:
                    params["cursor"] = cursor
                response = client.get(f"{PUBLIC_REST_BASE}/events", params=params)
                response.raise_for_status()
                payload = response.json()
                tickers.extend(
                    str(e.get("event_ticker", "")) for e in payload.get("events", [])
                )
                cursor = payload.get("cursor") or None
                if not cursor:
                    break
    except httpx.HTTPError as exc:
        logger.warning("could not list Kalshi events: %s", exc)
        return []
    return [t for t in tickers if t]


def run_pipeline(
    *,
    settings: Settings | None = None,
    horizon_hours: float = DEFAULT_HORIZON_HOURS,
) -> dict[str, Any]:
    settings = settings or load_settings()
    settings.paths.ensure()
    now = utc_now()

    schedule = load_stored(settings)
    if not schedule:
        raise ConfigError(
            "no forward schedule stored; run build_forward_schedule first"
        )
    games = upcoming(schedule, now, horizon_hours)
    tickers = fetch_event_tickers()
    mappings = map_games(games, tickers, now=now)
    by_game = {m.source_game_id: m for m in mappings}

    rows: list[dict[str, Any]] = []
    for game in games:
        anchors = game.anchors()
        mapping = by_game.get(game.source_game_id)
        rows.append({
            "source_game_id": game.source_game_id,
            "matchup": f"{game.away_team}@{game.home_team}",
            "phase": game.phase,
            "counts_toward_research": game.counts_toward_research,
            "tipoff_utc": game.tipoff_utc,
            "tipoff_et": game.tipoff_utc.astimezone(EASTERN),
            **{label: anchors[label] for label in ANCHOR_LABELS.values()},
            "kalshi_status": mapping.status if mapping else "unknown",
            "kalshi_event_ticker": mapping.event_ticker if mapping else None,
            "identity_severity": mapping.severity(now) if mapping else "WARNING",
        })

    report = {
        "generated_at_utc": now.isoformat(),
        "horizon_hours": horizon_hours,
        "upcoming_games": len(games),
        "kalshi_events_listed": len(tickers),
        "kalshi_listing_reachable": bool(tickers),
        "market_identity": summarise(mappings, now),
        "games": rows,
    }
    path = settings.paths.reports / "upcoming_capture_slate.json"
    path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    report["written_files"] = [str(path)]
    return report


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Show the upcoming capture slate.")
    parser.add_argument("--hours", type=float, default=DEFAULT_HORIZON_HOURS)
    parser.add_argument("--json", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    logging.basicConfig(level=logging.WARNING)
    try:
        report = run_pipeline(horizon_hours=args.hours)
    except ConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if args.json:
        print(json.dumps(report, indent=2, default=str))
        return 0

    print(f"\nupcoming {report['horizon_hours']:.0f}h: {report['upcoming_games']} game(s)"
          f" | kalshi events listed: {report['kalshi_events_listed']}")
    if not report["games"]:
        print("  (no games in the horizon)")
    else:
        print(f"\n{'matchup':10s} {'phase':14s} {'tipoff ET':16s} {'T-6h':6s} "
              f"{'T-3h':6s} {'T-1h':6s} {'T-30m':6s} {'kalshi':14s} {'sev':8s}")
        def clock(row: dict[str, Any], key: str) -> str:
            return pd.Timestamp(row[key]).tz_convert(EASTERN).strftime("%H:%M")

        for row in report["games"]:
            print(f"{row['matchup']:10s} {row['phase']:14s} "
                  f"{pd.Timestamp(row['tipoff_et']).strftime('%Y-%m-%d %H:%M'):16s} "
                  f"{clock(row, 'T-6h'):6s} {clock(row, 'T-3h'):6s} "
                  f"{clock(row, 'T-1h'):6s} {clock(row, 'T-30m'):6s} "
                  f"{row['kalshi_status']:14s} {row['identity_severity']:8s}")
    identity = report["market_identity"]
    print(f"\nmarket identity: {identity['by_status']}")
    if identity["urgent_missing_identity"]:
        print("  URGENT - approaching anchor with no market:")
        for entry in identity["urgent_missing_identity"]:
            print(f"    {entry['matchup']} in {entry['hours_to_tipoff']}h "
                  f"[{entry['severity']}]")
    for path in report["written_files"]:
        print(f"\nWrote {Path(path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
