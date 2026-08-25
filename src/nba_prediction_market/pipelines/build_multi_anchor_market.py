"""Backfill Kalshi market history across several pregame anchors.

Phase 4A0 found no tradable disagreement at T-30. That says nothing about
earlier in the day, when official injury news is still arriving and the market
has had less time to absorb it. Answering that needs the market's state at
several anchors, which needs a wider window than Phase 2 fetched.

One request per market covers all of them. Phase 2 cached a 60-minute window
ending at T-30 under the slug ``t30_lb60_p1``; this fetches **six hours** ending
at the same instant under ``t30_lb360_p1``. Because the slug is part of the
cache path, the Phase 2 cache cannot be touched.

T-15m and T-5m are deliberately absent. They fall *after* the T-30 decision
anchor, so they can never be inputs to a T-30 strategy. The six-hour window ends
at T-30 and does not contain them at all, which makes their exclusion
structural rather than a matter of discipline.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

from nba_prediction_market.clients.kalshi import KalshiClient
from nba_prediction_market.config import (
    DEFAULT_CANDLE_PERIOD_INTERVAL,
    KALSHI_NBA_SERIES_TICKER,
    ConfigError,
    Settings,
    load_settings,
)
from nba_prediction_market.ingestion.candle_cache import (
    CandleCache,
    CandleRequest,
    cache_slug,
)
from nba_prediction_market.ingestion.candlesticks import Candle, select_pregame_quote
from nba_prediction_market.ingestion.raw_store import utc_now
from nba_prediction_market.pipelines.build_pregame_quotes import (
    choose_candle_endpoint,
    fetch_market_candles,
    parse_cutoff_ts,
)

logger = logging.getLogger(__name__)

#: Anchors, in minutes before tipoff. All lie at or before the T-30 decision
#: anchor, so any of them could inform a T-30 forecast.
ANCHOR_MINUTES: tuple[int, ...] = (360, 180, 60, 30)
ANCHOR_LABELS: dict[int, str] = {360: "T-6h", 180: "T-3h", 60: "T-1h", 30: "T-30m"}

#: Window fetched per market, in minutes before tipoff. Ends at T-30.
LOOKBACK_MINUTES: int = 360
ANCHOR_END_MINUTES: int = 30

#: A quote older than this at its anchor is flagged rather than dropped.
MAX_QUOTE_AGE_SECONDS: float = 1800.0

HOLDOUT_SEASON = 2025


@dataclass
class BackfillStats:
    markets_requested: int = 0
    from_cache: int = 0
    fetched: int = 0
    failed: int = 0
    candles: int = 0
    malformed: int = 0
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "markets_requested": self.markets_requested,
            "served_from_cache": self.from_cache,
            "fetched_from_api": self.fetched,
            "failed": self.failed,
            "candles_parsed": self.candles,
            "malformed_candles": self.malformed,
            "error_examples": self.errors[:10],
        }


def window_for(tipoff: datetime) -> tuple[int, int]:
    """Unix-second window covering every anchor, ending at T-30."""
    end = tipoff - timedelta(minutes=ANCHOR_END_MINUTES)
    start = tipoff - timedelta(minutes=LOOKBACK_MINUTES)
    return int(start.timestamp()), int(end.timestamp())


def observation_rows(
    candles: list[Candle], game_id: Any, ticker: str, side: str
) -> list[dict[str, Any]]:
    """Every candle in the window, preserved rather than reduced to anchors.

    Keeping the raw stream is what makes the event study possible later: an
    injury report lands at an arbitrary minute, and only a full series can say
    what the market looked like just before and just after it.
    """
    return [
        {
            "nba_game_id": game_id,
            "market_ticker": ticker,
            "side": side,
            "observed_at_utc": candle.end_ts_utc,
            "yes_bid": candle.yes_bid,
            "yes_ask": candle.yes_ask,
            "midpoint": candle.midpoint,
            "spread": candle.spread,
            "last_trade_price": candle.last_trade_price,
            "previous_trade_price": candle.previous_trade_price,
            "volume": candle.volume,
            "open_interest": candle.open_interest,
        }
        for candle in candles
    ]


def anchor_quotes(
    candles: list[Candle], tipoff: datetime, game_id: Any, side: str
) -> list[dict[str, Any]]:
    """The market state at each anchor, chosen by the Phase 2 selector.

    Reusing that selector is the point: it enforces ``end_period_ts <= anchor``,
    so no anchor can see a candle from after itself.
    """
    rows: list[dict[str, Any]] = []
    for minutes in ANCHOR_MINUTES:
        anchor = tipoff - timedelta(minutes=minutes)
        selection = select_pregame_quote(
            list(candles), anchor, max_age_seconds=MAX_QUOTE_AGE_SECONDS
        )
        candle = selection.candle
        rows.append({
            "nba_game_id": game_id,
            "side": side,
            "anchor_minutes": minutes,
            "anchor": ANCHOR_LABELS[minutes],
            "anchor_ts_utc": anchor,
            "yes_bid": candle.yes_bid if candle else None,
            "yes_ask": candle.yes_ask if candle else None,
            "midpoint": candle.midpoint if candle else None,
            "spread": candle.spread if candle else None,
            "last_trade_price": candle.last_trade_price if candle else None,
            "volume": candle.volume if candle else None,
            "open_interest": candle.open_interest if candle else None,
            "quote_ts_utc": candle.end_ts_utc if candle else None,
            "quote_age_seconds": selection.quote_age_seconds,
            "usable": selection.usable,
            "issue": selection.issue,
        })
    return rows


def run_pipeline(
    *, settings: Settings | None = None, refresh: bool = False, limit: int | None = None
) -> dict[str, Any]:
    settings = settings or load_settings()
    settings.paths.ensure()
    processed = settings.paths.processed

    quotes_path = processed / "nba_kalshi_pregame_t30_2025_26.parquet"
    if not quotes_path.is_file():
        raise ConfigError(f"Missing {quotes_path}. Run Phase 2 first.")
    games = pd.read_parquet(quotes_path)
    games["game_datetime_utc"] = pd.to_datetime(games["game_datetime_utc"], utc=True)
    games = games.sort_values("game_datetime_utc", kind="stable")
    if limit:
        games = games.head(limit)

    slug = cache_slug(
        minutes_before_tip=ANCHOR_END_MINUTES,
        lookback_minutes=LOOKBACK_MINUTES,
        period_interval=DEFAULT_CANDLE_PERIOD_INTERVAL,
    )
    cache = CandleCache(settings.paths.root / "raw" / "kalshi" / "candlesticks", slug)
    logger.info("cache slug %s (Phase 2 used t30_lb60_p1 and is untouched)", slug)

    stats = BackfillStats()
    observations: list[dict[str, Any]] = []
    anchors: list[dict[str, Any]] = []

    with KalshiClient() as client:
        cutoff = parse_cutoff_ts(client.get_historical_cutoff())
        for game in games.itertuples():
            tipoff = pd.Timestamp(game.game_datetime_utc).to_pydatetime()
            start_ts, end_ts = window_for(tipoff)
            game_date = tipoff.date()
            for side, ticker in (
                ("home", game.home_market_ticker),
                ("away", game.away_market_ticker),
            ):
                if not isinstance(ticker, str) or not ticker:
                    continue
                stats.markets_requested += 1
                request = CandleRequest(
                    market_ticker=ticker,
                    start_ts=start_ts,
                    end_ts=end_ts,
                    period_interval=DEFAULT_CANDLE_PERIOD_INTERVAL,
                )
                endpoint = choose_candle_endpoint(tipoff, cutoff)
                outcome = fetch_market_candles(
                    client, cache,
                    market_ticker=ticker, game_date=game_date, request=request,
                    endpoint=endpoint, series_ticker=KALSHI_NBA_SERIES_TICKER,
                    refresh=refresh,
                )
                if outcome.error:
                    stats.failed += 1
                    stats.errors.append(f"{ticker}: {outcome.error}")
                    continue
                if outcome.from_cache:
                    stats.from_cache += 1
                else:
                    stats.fetched += 1
                stats.candles += len(outcome.candles)
                stats.malformed += outcome.malformed
                observations.extend(
                    observation_rows(outcome.candles, game.nba_game_id, ticker, side)
                )
                anchors.extend(
                    anchor_quotes(outcome.candles, tipoff, game.nba_game_id, side)
                )
            if stats.markets_requested % 200 == 0:
                logger.info(
                    "%d markets | cache=%d fetched=%d failed=%d",
                    stats.markets_requested, stats.from_cache, stats.fetched,
                    stats.failed,
                )

    observation_frame = pd.DataFrame(observations)
    anchor_frame = pd.DataFrame(anchors)
    obs_path = processed / "nba_market_observations_6h_2025_26.parquet"
    anchor_path = processed / "nba_market_multi_anchor_2025_26.parquet"
    observation_frame.to_parquet(obs_path, index=False)
    anchor_frame.to_parquet(anchor_path, index=False)

    coverage = []
    for minutes in ANCHOR_MINUTES:
        subset = anchor_frame[anchor_frame["anchor_minutes"] == minutes]
        both = (
            subset.groupby("nba_game_id")["usable"].sum().eq(2).sum()
            if len(subset) else 0
        )
        ages = subset["quote_age_seconds"].dropna()
        coverage.append({
            "anchor": ANCHOR_LABELS[minutes],
            "anchor_minutes": minutes,
            "market_rows": len(subset),
            "usable_quotes": int(subset["usable"].sum()) if len(subset) else 0,
            "games_with_both_sides_usable": int(both),
            "quote_age_seconds": {
                "median": float(ages.median()), "p95": float(ages.quantile(0.95)),
                "max": float(ages.max()),
            } if len(ages) else None,
        })

    report = {
        "generated_at_utc": utc_now().isoformat(),
        "cache_slug": slug,
        "phase_2_cache_untouched": "t30_lb60_p1",
        "window_minutes_before_tip": [LOOKBACK_MINUTES, ANCHOR_END_MINUTES],
        "anchors": [ANCHOR_LABELS[m] for m in ANCHOR_MINUTES],
        "anchors_excluded": ["T-15m", "T-5m"],
        "anchors_excluded_reason": (
            "both fall after the T-30 decision anchor; the fetched window ends "
            "at T-30 so they are structurally absent, not merely unused"
        ),
        "games": len(games),
        "stats": stats.to_dict(),
        "coverage_by_anchor": coverage,
        "observation_rows": len(observation_frame),
        "written_files": [str(obs_path), str(anchor_path)],
    }
    report_path = settings.paths.reports / "market_multi_anchor_backfill_2025_26.json"
    report_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    report["written_files"].append(str(report_path))
    return report


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Backfill multi-anchor Kalshi history.")
    parser.add_argument("--refresh", action="store_true", help="Ignore cached payloads.")
    parser.add_argument("--limit", type=int, default=None, help="First N games only.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s | %(message)s")
    try:
        report = run_pipeline(refresh=args.refresh, limit=args.limit)
    except ConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(f"\nstats: {report['stats']}")
    print(f"\n{'anchor':8s} {'rows':>7s} {'usable':>7s} {'both sides':>11s} {'age p95':>9s}")
    for row in report["coverage_by_anchor"]:
        age = row["quote_age_seconds"]
        age_text = f"{age['p95']:.0f}s" if age else "-"
        print(f"{row['anchor']:8s} {row['market_rows']:7d} {row['usable_quotes']:7d} "
              f"{row['games_with_both_sides_usable']:11d} {age_text:>9s}")
    for path in report["written_files"]:
        print(f"\nWrote {Path(path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
