"""Phase 1 pipeline: ingest -> normalize -> match -> report.

Run with::

    python -m nba_prediction_market.pipelines.build_dataset --season 2025
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from nba_prediction_market.clients.balldontlie import BallDontLieClient
from nba_prediction_market.clients.kalshi import KalshiClient
from nba_prediction_market.config import (
    KALSHI_NBA_SERIES_TICKER,
    ConfigError,
    Settings,
    load_settings,
    season_label,
    season_slug,
    season_window,
)
from nba_prediction_market.ingestion.kalshi_markets import build_markets_frame
from nba_prediction_market.ingestion.nba_games import build_games_frame, verify_season
from nba_prediction_market.ingestion.raw_store import RawSnapshot, RawStore, utc_now, utc_stamp
from nba_prediction_market.matching.game_market_matcher import (
    STATUS_AMBIGUOUS,
    STATUS_MATCHED,
    STATUS_UNMATCHED_KALSHI,
    STATUS_UNMATCHED_NBA,
    build_kalshi_events_frame,
    match_games_to_markets,
)

logger = logging.getLogger(__name__)

#: How many example records per category to embed in the report.
REPORT_EXAMPLE_LIMIT = 5


@dataclass
class PipelineResult:
    """Everything a run produced, for programmatic use and for the CLI summary."""

    games: pd.DataFrame
    markets: pd.DataFrame
    kalshi_events: pd.DataFrame
    matches: pd.DataFrame
    report: dict[str, Any]
    written_files: list[Path]


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


def _clean_examples(frame: pd.DataFrame, columns: list[str], limit: int) -> list[dict[str, Any]]:
    """Take up to ``limit`` rows as plain dicts, dropping null-only fields."""
    subset = [c for c in columns if c in frame.columns]
    records = frame.head(limit)[subset].to_dict("records")
    cleaned: list[dict[str, Any]] = []
    for record in records:
        cleaned.append(
            {k: (None if pd.isna(v) else v) for k, v in record.items() if not _is_blank(v)}
        )
    return cleaned


def _is_blank(value: Any) -> bool:
    if value is None:
        return True
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _write_table(frame: pd.DataFrame, directory: Path, stem: str, *, csv: bool) -> list[Path]:
    """Write a frame to parquet (canonical) and optionally CSV (convenience)."""
    written: list[Path] = []
    parquet_path = directory / f"{stem}.parquet"
    frame.to_parquet(parquet_path, index=False)
    written.append(parquet_path)
    if csv:
        csv_path = directory / f"{stem}.csv"
        frame.to_csv(csv_path, index=False)
        written.append(csv_path)
    return written


def fetch_nba_games(
    settings: Settings, season: int, raw_store: RawStore
) -> tuple[list[dict[str, Any]], RawSnapshot]:
    """Fetch every game for ``season``, preserving raw pages."""
    api_key = settings.require_balldontlie_key()
    pages: list[dict[str, Any]] = []
    games: list[dict[str, Any]] = []

    def capture(page_number: int, payload: dict[str, Any]) -> None:
        pages.append(payload)
        logger.info("BALLDONTLIE page %d: %d games", page_number, len(payload.get("data", [])))

    with BallDontLieClient(
        api_key,
        base_url=settings.balldontlie_base_url,
        timeout=settings.request_timeout,
        min_interval=settings.balldontlie_min_interval,
        max_retries=settings.max_retries,
    ) as client:
        for page in client.iter_games(season, on_page=capture):
            games.extend(page)

    snapshot = raw_store.write(
        f"balldontlie_games_season_{season}",
        pages,
        record_count=len(games),
        params={"seasons[]": season, "per_page": 100, "endpoint": "/games"},
    )
    return games, snapshot


@dataclass
class KalshiFetch:
    """Raw Kalshi pull: both market stores merged, plus events and provenance."""

    markets: list[dict[str, Any]]
    events: list[dict[str, Any]]
    sources_by_ticker: dict[str, list[str]]
    cutoff: dict[str, Any]
    snapshots: list[RawSnapshot]


def fetch_kalshi(settings: Settings, series_ticker: str, raw_store: RawStore) -> KalshiFetch:
    """Fetch Kalshi markets (both stores), events, and the historical cutoff."""
    snapshots: list[RawSnapshot] = []
    market_pages: list[dict[str, Any]] = []
    event_pages: list[dict[str, Any]] = []

    with KalshiClient(
        base_url=settings.kalshi_base_url,
        timeout=settings.request_timeout,
        min_interval=settings.kalshi_min_interval,
        max_retries=settings.max_retries,
    ) as client:
        cutoff = client.get_historical_cutoff()
        logger.info("Kalshi historical cutoff: %s", cutoff)
        snapshots.append(
            raw_store.write(
                "kalshi_historical_cutoff", [cutoff], record_count=1,
                params={"endpoint": "/historical/cutoff"},
            )
        )

        def capture_market(source: str, page_number: int, payload: dict[str, Any]) -> None:
            market_pages.append({"source": source, **payload})
            logger.info(
                "Kalshi %s page %d: %d markets",
                source,
                page_number,
                len(payload.get("markets", [])),
            )

        markets, sources = client.fetch_markets_from_both_stores(
            series_ticker, on_page=capture_market
        )
        snapshots.append(
            raw_store.write(
                f"kalshi_markets_{series_ticker}", market_pages, record_count=len(markets),
                params={
                    "series_ticker": series_ticker,
                    "endpoints": ["/historical/markets", "/markets"],
                },
            )
        )

        events: list[dict[str, Any]] = []
        for page in client.iter_events(
            series_ticker,
            on_page=lambda n, p: event_pages.append(p),
        ):
            events.extend(page)
        snapshots.append(
            raw_store.write(
                f"kalshi_events_{series_ticker}", event_pages, record_count=len(events),
                params={"series_ticker": series_ticker, "endpoint": "/events"},
            )
        )

    return KalshiFetch(
        markets=markets,
        events=events,
        sources_by_ticker=sources,
        cutoff=cutoff,
        snapshots=snapshots,
    )


def _quality_checks(
    games: pd.DataFrame, markets: pd.DataFrame, matches: pd.DataFrame
) -> dict[str, Any]:
    """Cross-source consistency signals worth eyeballing after every run."""
    matched = matches[matches["match_status"] == STATUS_MATCHED]
    orientation = matched["orientation_agrees"]
    settlement = matched["settlement_agrees_with_score"].dropna()
    return {
        "nba_games_with_home_win": int(games["home_win"].notna().sum()),
        "nba_games_final": int(games["is_final"].fillna(False).sum()),
        "nba_games_postseason": int(games["postseason"].fillna(False).sum()),
        "kalshi_markets_flagged_non_nba": int((~markets["is_nba_matchup"]).sum()),
        "kalshi_ticker_date_vs_rules_date_disagreements": int(
            (markets["ticker_date_agrees_with_rules_date"] == False).sum()  # noqa: E712
        ),
        "kalshi_subtitle_vs_ticker_team_disagreements": int(
            (markets["team_codes_agree_with_event_subtitle"] == False).sum()  # noqa: E712
        ),
        "matched_orientation_agrees": int((orientation == True).sum()),  # noqa: E712
        "matched_orientation_disagrees": int((orientation == False).sum()),  # noqa: E712
        "matched_settlement_checked": len(settlement),
        "matched_settlement_agrees": int((settlement == True).sum()),  # noqa: E712
        "matched_settlement_disagrees": int((settlement == False).sum()),  # noqa: E712
    }


def build_report(
    *,
    season: int,
    series_ticker: str,
    games: pd.DataFrame,
    markets: pd.DataFrame,
    kalshi_events: pd.DataFrame,
    matches: pd.DataFrame,
    season_verification: dict[str, Any],
    kalshi_cutoff: dict[str, Any],
    market_sources: dict[str, list[str]],
    snapshots: list[RawSnapshot],
    started_at: str,
) -> dict[str, Any]:
    """Assemble the match report: counts, examples, and provenance."""
    counts = matches["match_status"].value_counts().to_dict()
    nba_side = matches[matches["nba_game_id"].notna()]
    kalshi_side = matches[matches["kalshi_event_ticker"].notna()]

    nba_example_cols = [
        "match_status", "match_tier", "date_offset_days", "nba_game_id", "nba_game_date",
        "nba_home_team_code", "nba_away_team_code", "nba_postseason", "nba_status",
        "kalshi_event_ticker", "kalshi_home_market_ticker", "kalshi_away_market_ticker",
        "ambiguity_reason", "candidate_event_tickers",
    ]
    kalshi_example_cols = [
        "match_status", "kalshi_event_ticker", "kalshi_event_sub_title",
        "kalshi_scheduled_game_date", "kalshi_home_team_code", "kalshi_away_team_code",
        "kalshi_event_title", "ambiguity_reason", "candidate_nba_game_ids",
    ]

    by_status = {}
    for status, cols, frame in (
        (STATUS_MATCHED, nba_example_cols, matches),
        (STATUS_UNMATCHED_NBA, nba_example_cols, matches),
        (STATUS_UNMATCHED_KALSHI, kalshi_example_cols, matches),
        (STATUS_AMBIGUOUS, nba_example_cols, matches),
    ):
        subset = frame[frame["match_status"] == status]
        by_status[status] = {
            "count": len(subset),
            "examples": _clean_examples(subset, cols, REPORT_EXAMPLE_LIMIT),
        }

    matched = matches[matches["match_status"] == STATUS_MATCHED]
    return {
        "generated_at_utc": utc_now().isoformat(),
        "started_at_utc": started_at,
        "season": season,
        "season_label": season_label(season),
        "season_window": [d.isoformat() for d in season_window(season)],
        "kalshi_series_ticker": series_ticker,
        "inputs": {
            "nba_games": len(games),
            "kalshi_markets": len(markets),
            "kalshi_events": len(kalshi_events),
            "kalshi_markets_seen_in_both_stores": sum(
                1 for s in market_sources.values() if len(s) > 1
            ),
            "kalshi_historical_cutoff": kalshi_cutoff,
        },
        "season_verification": season_verification,
        "counts": {
            STATUS_MATCHED: int(counts.get(STATUS_MATCHED, 0)),
            STATUS_UNMATCHED_NBA: int(counts.get(STATUS_UNMATCHED_NBA, 0)),
            STATUS_UNMATCHED_KALSHI: int(counts.get(STATUS_UNMATCHED_KALSHI, 0)),
            STATUS_AMBIGUOUS: int(counts.get(STATUS_AMBIGUOUS, 0)),
            "total_rows": len(matches),
            "nba_games_represented": int(nba_side["nba_game_id"].nunique()),
            "kalshi_events_represented": int(kalshi_side["kalshi_event_ticker"].nunique()),
        },
        "coverage": {
            "nba_games_matched_pct": round(
                100.0 * len(matched) / len(games), 3
            ) if len(games) else None,
            "kalshi_events_matched_pct": round(
                100.0 * len(matched) / len(kalshi_events), 3
            ) if len(kalshi_events) else None,
        },
        "match_tiers": {
            str(k): int(v) for k, v in matched["match_tier"].value_counts().to_dict().items()
        },
        "quality_checks": _quality_checks(games, markets, matches),
        "by_status": by_status,
        "raw_snapshots": [s.to_dict() for s in snapshots],
    }


def run_pipeline(
    season: int,
    *,
    settings: Settings | None = None,
    series_ticker: str = KALSHI_NBA_SERIES_TICKER,
    write_csv: bool = True,
) -> PipelineResult:
    """Execute the full Phase 1 pipeline and write all artefacts."""
    settings = settings or load_settings()
    settings.paths.ensure()
    started_at = utc_now().isoformat()
    run_stamp = utc_stamp()

    nba_store = RawStore(settings.paths.raw_nba, run_stamp=run_stamp)
    kalshi_store = RawStore(settings.paths.raw_kalshi, run_stamp=run_stamp)

    logger.info("Fetching BALLDONTLIE games for season %s (%s)", season, season_label(season))
    raw_games, nba_snapshot = fetch_nba_games(settings, season, nba_store)

    logger.info("Fetching Kalshi %s markets and events", series_ticker)
    kalshi = fetch_kalshi(settings, series_ticker, kalshi_store)

    games = build_games_frame(raw_games, season)
    season_verification = verify_season(games.to_dict("records"), season)
    markets = build_markets_frame(
        kalshi.markets,
        events=kalshi.events,
        sources_by_ticker=kalshi.sources_by_ticker,
        season_window=season_window(season),
    )
    kalshi_events = build_kalshi_events_frame(markets)
    matches, summary = match_games_to_markets(games, kalshi_events)

    logger.info(
        "Match summary: %d matched, %d unmatched NBA, %d unmatched Kalshi, %d ambiguous",
        summary.matched, summary.unmatched_nba, summary.unmatched_kalshi, summary.ambiguous,
    )

    slug = season_slug(season)
    written: list[Path] = []
    written += _write_table(games, settings.paths.processed, f"nba_games_{slug}", csv=write_csv)
    written += _write_table(
        markets, settings.paths.processed, f"kalshi_nba_markets_{slug}", csv=write_csv
    )
    written += _write_table(
        matches, settings.paths.processed, f"nba_kalshi_matches_{slug}", csv=write_csv
    )
    written += _write_table(
        kalshi_events, settings.paths.processed, f"kalshi_nba_events_{slug}", csv=write_csv
    )

    report = build_report(
        season=season,
        series_ticker=series_ticker,
        games=games,
        markets=markets,
        kalshi_events=kalshi_events,
        matches=matches,
        season_verification=season_verification,
        kalshi_cutoff=kalshi.cutoff,
        market_sources=kalshi.sources_by_ticker,
        snapshots=[nba_snapshot, *kalshi.snapshots],
        started_at=started_at,
    )
    report_path = settings.paths.reports / "match_report.json"
    report_path.write_text(
        json.dumps(report, indent=2, default=_json_default), encoding="utf-8"
    )
    written.append(report_path)

    return PipelineResult(
        games=games,
        markets=markets,
        kalshi_events=kalshi_events,
        matches=matches,
        report=report,
        written_files=written,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m nba_prediction_market.pipelines.build_dataset",
        description=(
            "Phase 1: ingest NBA games (BALLDONTLIE) and Kalshi KXNBAGAME market "
            "metadata, normalize both, and match them deterministically."
        ),
    )
    parser.add_argument(
        "--season", type=int, default=2025,
        help="BALLDONTLIE season start year; 2025 means the 2025-26 season (default: 2025).",
    )
    parser.add_argument(
        "--series-ticker", default=KALSHI_NBA_SERIES_TICKER,
        help=f"Kalshi series ticker (default: {KALSHI_NBA_SERIES_TICKER}).",
    )
    parser.add_argument(
        "--data-dir", default=None,
        help="Root directory for raw/processed/report output (default: ./data).",
    )
    parser.add_argument("--no-csv", action="store_true", help="Write only parquet, skip CSV.")
    parser.add_argument(
        "--log-level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="Logging verbosity.",
    )
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
            args.season,
            settings=settings,
            series_ticker=args.series_ticker,
            write_csv=not args.no_csv,
        )
    except ConfigError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2

    counts = result.report["counts"]
    print()
    print(f"Season {result.report['season_label']} | series {args.series_ticker}")
    print(f"  NBA games           : {len(result.games)}")
    print(f"  Kalshi markets      : {len(result.markets)}")
    print(f"  Kalshi events       : {len(result.kalshi_events)}")
    print(f"  matched             : {counts[STATUS_MATCHED]}")
    print(f"  unmatched_nba       : {counts[STATUS_UNMATCHED_NBA]}")
    print(f"  unmatched_kalshi    : {counts[STATUS_UNMATCHED_KALSHI]}")
    print(f"  ambiguous           : {counts[STATUS_AMBIGUOUS]}")
    print()
    print("Files written:")
    for path in result.written_files:
        print(f"  {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
