"""Phase 2 pipeline: pregame Kalshi quotes at T-minus-N for matched NBA games.

Answers one question per game: *what were the executable Kalshi quotes and the
market-implied probability exactly N minutes before scheduled tipoff?*

Run with::

    python -m nba_prediction_market.pipelines.build_pregame_quotes \\
        --season 2025 --minutes-before-tip 30 --max-quote-age-minutes 10

Consumes the Phase 1 processed tables; it never re-derives matching or team
orientation. No modelling, no strategy, no profit calculation.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from nba_prediction_market.clients.base import ApiError
from nba_prediction_market.clients.kalshi import (
    CANDLES_HISTORICAL,
    CANDLES_SERIES,
    KalshiClient,
)
from nba_prediction_market.config import (
    DEFAULT_CANDLE_PERIOD_INTERVAL,
    DEFAULT_MAX_QUOTE_AGE_MINUTES,
    DEFAULT_MINUTES_BEFORE_TIP,
    DEFAULT_QUOTE_LOOKBACK_MINUTES,
    KALSHI_NBA_SERIES_TICKER,
    ConfigError,
    Settings,
    load_settings,
    season_label,
    season_slug,
)
from nba_prediction_market.ingestion.candle_cache import (
    CandleCache,
    CandleRequest,
    cache_slug,
)
from nba_prediction_market.ingestion.candlesticks import (
    DERIVED_PRICE_DECIMALS,
    ISSUE_FETCH_FAILED,
    ISSUE_NO_MARKET_TICKER,
    ISSUE_TEAM_MISMATCH,
    QUOTE_ISSUES,
    Candle,
    QuoteSelection,
    parse_candles,
    prediction_timestamp,
    select_pregame_quote,
)
from nba_prediction_market.ingestion.game_phase import (
    PHASE_REGULAR_SEASON,
    verify_regular_season,
)
from nba_prediction_market.ingestion.raw_store import utc_now
from nba_prediction_market.matching.game_market_matcher import STATUS_MATCHED

logger = logging.getLogger(__name__)

REPORT_EXAMPLE_LIMIT = 5

#: Deviations of (home midpoint + away midpoint) from 1.0 that get counted.
MIDPOINT_DEVIATION_THRESHOLDS: tuple[float, ...] = (0.01, 0.02, 0.05)

_SIDE_FIELDS = [
    "market_ticker",
    "market_team",
    "yes_bid",
    "yes_ask",
    "market_midpoint",
    "last_trade_price",
    "previous_trade_price",
    "spread",
    "candle_volume",
    "open_interest",
    "quote_ts_utc",
    "quote_age_seconds",
    "quote_usable",
    "quote_issue",
    "candle_source",
    "candles_in_window",
]

PREGAME_COLUMNS: list[str] = [
    # NBA
    "nba_game_id",
    "game_date",
    "game_datetime_utc",
    "prediction_ts_utc",
    "season",
    "postseason",
    "home_team",
    "away_team",
    "home_score",
    "away_score",
    "home_win",
    # match provenance
    "kalshi_event_ticker",
    "match_method",
    # per-side
    *[f"home_{name}" for name in _SIDE_FIELDS],
    *[f"away_{name}" for name in _SIDE_FIELDS],
    # quality
    "both_sides_usable",
    "market_midpoint_sum",
    "midpoint_sum_deviation_from_one",
    # run configuration, carried so a row explains itself
    "minutes_before_tip",
    "max_quote_age_seconds",
]


@dataclass
class PregameResult:
    """Everything a Phase 2 run produced."""

    quotes: pd.DataFrame
    report: dict[str, Any]
    written_files: list[Path]


@dataclass
class FetchOutcome:
    """One market's candle pull: payload plus how it was obtained."""

    candles: list[Candle] = field(default_factory=list)
    malformed: int = 0
    source: str | None = None
    error: str | None = None
    from_cache: bool = False


# --- endpoint routing ------------------------------------------------------


def parse_cutoff_ts(cutoff: dict[str, Any] | None) -> datetime | None:
    """Extract the archive boundary from ``GET /historical/cutoff``.

    Uses ``market_settled_ts`` -- the boundary that governs which store holds a
    settled market. Returns ``None`` if the field is absent, which routes
    everything to the live endpoint rather than guessing.
    """
    if not isinstance(cutoff, dict):
        return None
    raw = cutoff.get("market_settled_ts")
    if not raw:
        return None
    text = str(raw).replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    return parsed.replace(tzinfo=UTC) if parsed.tzinfo is None else parsed.astimezone(UTC)


def choose_candle_endpoint(
    market_end_utc: datetime | None, cutoff_utc: datetime | None
) -> str:
    """Route a market to the archived or the live candlestick endpoint.

    A market that finished at or before the historical cutoff lives in the
    archive. Anything later -- or anything whose end time is unknown -- goes to
    the live endpoint, which is the general-purpose one.
    """
    if market_end_utc is None or cutoff_utc is None:
        return CANDLES_SERIES
    return CANDLES_HISTORICAL if market_end_utc <= cutoff_utc else CANDLES_SERIES


def _as_utc(value: Any) -> datetime | None:
    if value is None:
        return None
    try:
        stamp = pd.Timestamp(value)
    except (TypeError, ValueError):
        return None
    if pd.isna(stamp):
        return None
    stamp = stamp.tz_localize(UTC) if stamp.tzinfo is None else stamp.tz_convert(UTC)
    return stamp.to_pydatetime()


# --- fetching --------------------------------------------------------------


def fetch_market_candles(
    client: KalshiClient,
    cache: CandleCache,
    *,
    market_ticker: str,
    game_date: date,
    request: CandleRequest,
    endpoint: str,
    series_ticker: str,
    refresh: bool,
) -> FetchOutcome:
    """Fetch one market's candles, preferring a valid cache entry.

    Falls back from the archived endpoint to the live one on a 404, because tier
    membership is inferred from a cutoff timestamp and can be wrong at the edges.
    """
    if not refresh:
        cached = cache.load(request, game_date)
        if cached is not None:
            candles, malformed = parse_candles(cached)
            return FetchOutcome(
                candles=candles, malformed=malformed, source=endpoint, from_cache=True
            )

    attempts = [endpoint]
    if endpoint == CANDLES_HISTORICAL:
        attempts.append(CANDLES_SERIES)

    last_error: str | None = None
    for attempt in attempts:
        try:
            if attempt == CANDLES_HISTORICAL:
                payload = client.get_historical_candlesticks(
                    market_ticker,
                    start_ts=request.start_ts,
                    end_ts=request.end_ts,
                    period_interval=request.period_interval,
                )
            else:
                payload = client.get_series_candlesticks(
                    market_ticker,
                    series_ticker=series_ticker,
                    start_ts=request.start_ts,
                    end_ts=request.end_ts,
                    period_interval=request.period_interval,
                )
        except ApiError as exc:
            last_error = f"{attempt}: {exc}"
            if getattr(exc, "status_code", None) == 404 and attempt != attempts[-1]:
                logger.info("%s not in the archive; retrying on the live endpoint", market_ticker)
                continue
            logger.warning("Candle fetch failed for %s (%s): %s", market_ticker, attempt, exc)
            return FetchOutcome(error=last_error, source=attempt)

        cache.store(request, game_date, payload, endpoint=attempt)
        candles, malformed = parse_candles(payload)
        return FetchOutcome(candles=candles, malformed=malformed, source=attempt)

    return FetchOutcome(error=last_error or "unknown fetch failure", source=endpoint)


# --- row assembly ----------------------------------------------------------


def _side_row(prefix: str, ticker: Any, team: Any, selection: QuoteSelection | None,
              outcome: FetchOutcome | None, issue_override: str | None) -> dict[str, Any]:
    """Flatten one side's quote into ``home_*`` / ``away_*`` columns."""
    row: dict[str, Any] = {f"{prefix}_{name}": None for name in _SIDE_FIELDS}
    row[f"{prefix}_market_ticker"] = ticker
    row[f"{prefix}_market_team"] = team
    row[f"{prefix}_quote_usable"] = False
    if outcome is not None:
        row[f"{prefix}_candle_source"] = outcome.source
        row[f"{prefix}_candles_in_window"] = len(outcome.candles)

    if issue_override is not None:
        row[f"{prefix}_quote_issue"] = issue_override
        return row
    if selection is None:
        return row

    row[f"{prefix}_quote_issue"] = selection.issue
    row[f"{prefix}_quote_usable"] = selection.usable
    row[f"{prefix}_quote_age_seconds"] = selection.quote_age_seconds
    candle = selection.candle
    if candle is not None:
        row[f"{prefix}_quote_ts_utc"] = candle.end_ts_utc
        row[f"{prefix}_yes_bid"] = candle.yes_bid
        row[f"{prefix}_yes_ask"] = candle.yes_ask
        row[f"{prefix}_market_midpoint"] = candle.midpoint
        row[f"{prefix}_spread"] = candle.spread
        row[f"{prefix}_last_trade_price"] = candle.last_trade_price
        row[f"{prefix}_previous_trade_price"] = candle.previous_trade_price
        row[f"{prefix}_candle_volume"] = candle.volume
        row[f"{prefix}_open_interest"] = candle.open_interest
    return row


def _to_bool(value: Any) -> bool | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return bool(value)


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        return None
    return float(value)


def select_eligible_games(matches: pd.DataFrame, season: int) -> pd.DataFrame:
    """Matched, true regular-season games for ``season`` -- the primary study set.

    Selection is on the explicit ``nba_game_phase`` label, **not** on
    ``postseason == False``: play-in games and the NBA Cup final both carry
    ``postseason = False`` while not being regular-season games, and filtering on
    that flag admitted six play-in games into the 2025-26 set. Phase 1's tables
    still contain every phase; only this study set is narrowed.
    """
    if "nba_game_phase" not in matches.columns:
        raise ConfigError(
            "The Phase 1 match table predates game-phase classification and has no "
            "'nba_game_phase' column. Re-run: python -m "
            f"nba_prediction_market.pipelines.build_dataset --season {season}"
        )
    eligible = matches[
        (matches["match_status"] == STATUS_MATCHED)
        & (matches["nba_season"] == season)
        & (matches["nba_game_phase"] == PHASE_REGULAR_SEASON)
    ].copy()
    return eligible.sort_values(["nba_game_date", "nba_game_id"], kind="stable").reset_index(
        drop=True
    )


def verify_side_assignment(
    game: dict[str, Any], markets_by_ticker: dict[str, dict[str, Any]]
) -> tuple[str | None, str | None]:
    """Check the Phase 1 home/away market assignment. Returns per-side issues.

    Orientation comes from Phase 1's ``market_team_is_home``, never from array
    order. This re-derives the check against the market table so a contradiction
    (missing ticker, wrong team, both sides on one team) marks the row unusable
    instead of quietly producing a mirrored probability.
    """
    issues: list[str | None] = []
    home_ticker = game.get("kalshi_home_market_ticker")
    away_ticker = game.get("kalshi_away_market_ticker")

    for ticker, expected_team in (
        (home_ticker, game.get("nba_home_team_code")),
        (away_ticker, game.get("nba_away_team_code")),
    ):
        if not isinstance(ticker, str) or not ticker:
            issues.append(ISSUE_NO_MARKET_TICKER)
            continue
        market = markets_by_ticker.get(ticker)
        if market is None:
            issues.append(ISSUE_TEAM_MISMATCH)
            continue
        if market.get("market_team_code") != expected_team:
            issues.append(ISSUE_TEAM_MISMATCH)
            continue
        issues.append(None)

    # Both sides resolving to the same market is contradictory by construction.
    if home_ticker and home_ticker == away_ticker:
        return ISSUE_TEAM_MISMATCH, ISSUE_TEAM_MISMATCH
    return issues[0], issues[1]


def build_pregame_frame(
    eligible: pd.DataFrame,
    markets: pd.DataFrame,
    *,
    client: KalshiClient,
    cache: CandleCache,
    cutoff_utc: datetime | None,
    minutes_before_tip: int,
    lookback_minutes: int,
    max_quote_age_seconds: float,
    period_interval: int,
    series_ticker: str,
    refresh: bool,
    progress_every: int = 100,
) -> pd.DataFrame:
    """Fetch quotes for every eligible game and assemble one row per game."""
    markets_by_ticker = {
        m["ticker"]: m for m in markets.to_dict("records") if isinstance(m.get("ticker"), str)
    }

    rows: list[dict[str, Any]] = []
    for index, game in enumerate(eligible.to_dict("records"), start=1):
        tipoff = _as_utc(game.get("nba_tipoff_utc"))
        game_date = game.get("nba_game_date")
        if isinstance(game_date, str):
            game_date = date.fromisoformat(game_date)
        elif isinstance(game_date, pd.Timestamp):
            game_date = game_date.date()

        prediction_ts = prediction_timestamp(tipoff, minutes_before_tip) if tipoff else None
        home_issue, away_issue = verify_side_assignment(game, markets_by_ticker)

        row: dict[str, Any] = {
            "nba_game_id": game.get("nba_game_id"),
            "game_date": game_date,
            "game_datetime_utc": tipoff,
            "prediction_ts_utc": prediction_ts,
            "season": game.get("nba_season"),
            "postseason": _to_bool(game.get("nba_postseason")),
            "home_team": game.get("nba_home_team_code"),
            "away_team": game.get("nba_away_team_code"),
            "home_score": game.get("nba_home_score"),
            "away_score": game.get("nba_away_score"),
            "home_win": _to_bool(game.get("nba_home_win")),
            "kalshi_event_ticker": game.get("kalshi_event_ticker"),
            "match_method": game.get("match_tier"),
            "minutes_before_tip": minutes_before_tip,
            "max_quote_age_seconds": max_quote_age_seconds,
        }

        for prefix, ticker_key, team_key, issue in (
            ("home", "kalshi_home_market_ticker", "nba_home_team_code", home_issue),
            ("away", "kalshi_away_market_ticker", "nba_away_team_code", away_issue),
        ):
            ticker = game.get(ticker_key)
            team = game.get(team_key)
            if issue is not None or prediction_ts is None or game_date is None:
                row.update(
                    _side_row(prefix, ticker, team, None, None, issue or ISSUE_NO_MARKET_TICKER)
                )
                continue

            end_ts = int(prediction_ts.timestamp())
            start_ts = end_ts - lookback_minutes * 60
            request = CandleRequest(
                market_ticker=ticker,
                start_ts=start_ts,
                end_ts=end_ts,
                period_interval=period_interval,
            )
            market = markets_by_ticker.get(ticker, {})
            market_end = _as_utc(
                market.get("settlement_ts_utc") or market.get("close_time_utc")
            )
            endpoint = choose_candle_endpoint(market_end, cutoff_utc)

            outcome = fetch_market_candles(
                client,
                cache,
                market_ticker=ticker,
                game_date=game_date,
                request=request,
                endpoint=endpoint,
                series_ticker=series_ticker,
                refresh=refresh,
            )
            if outcome.error is not None:
                row.update(_side_row(prefix, ticker, team, None, outcome, ISSUE_FETCH_FAILED))
                continue

            selection = select_pregame_quote(
                outcome.candles,
                prediction_ts,
                max_age_seconds=max_quote_age_seconds,
                malformed_candles=outcome.malformed,
            )
            row.update(_side_row(prefix, ticker, team, selection, outcome, None))

        home_mid = _to_float(row.get("home_market_midpoint"))
        away_mid = _to_float(row.get("away_market_midpoint"))
        both_usable = bool(row.get("home_quote_usable")) and bool(row.get("away_quote_usable"))
        # Deliberately NOT normalised to sum to 1 -- the deviation is a signal.
        midpoint_sum = (
            round(home_mid + away_mid, DERIVED_PRICE_DECIMALS)
            if home_mid is not None and away_mid is not None
            else None
        )
        row["both_sides_usable"] = both_usable
        row["market_midpoint_sum"] = midpoint_sum
        row["midpoint_sum_deviation_from_one"] = (
            round(abs(midpoint_sum - 1.0), DERIVED_PRICE_DECIMALS)
            if midpoint_sum is not None
            else None
        )
        rows.append(row)

        if progress_every and index % progress_every == 0:
            logger.info("Processed %d/%d games", index, len(eligible))

    frame = pd.DataFrame(rows, columns=PREGAME_COLUMNS)
    return frame.sort_values(["game_date", "nba_game_id"], kind="stable").reset_index(drop=True)


# --- report ----------------------------------------------------------------


def _percentiles(values: pd.Series, points: tuple[int, ...] = (1, 5, 25, 50, 75, 95, 99)) -> dict:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    if clean.empty:
        return {"count": 0}
    out: dict[str, Any] = {
        "count": len(clean),
        "min": float(clean.min()),
        "mean": float(clean.mean()),
        "max": float(clean.max()),
    }
    for point in points:
        out[f"p{point}"] = float(clean.quantile(point / 100.0))
    return out


def _examples(frame: pd.DataFrame, columns: list[str], limit: int) -> list[dict[str, Any]]:
    subset = [c for c in columns if c in frame.columns]
    records = frame.head(limit)[subset].to_dict("records")
    cleaned = []
    for record in records:
        cleaned.append(
            {
                key: (None if _is_blank(value) else value)
                for key, value in record.items()
                if not _is_blank(value)
            }
        )
    return cleaned


def _is_blank(value: Any) -> bool:
    if value is None:
        return True
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def build_report(
    quotes: pd.DataFrame,
    *,
    season: int,
    phase_selection: dict[str, Any],
    minutes_before_tip: int,
    lookback_minutes: int,
    max_quote_age_seconds: float,
    period_interval: int,
    cutoff: dict[str, Any] | None,
    cache_stats: dict[str, int],
    started_at: str,
) -> dict[str, Any]:
    """Coverage, distributions, and worked examples of every failure category."""
    total = len(quotes)
    home_ok = quotes["home_quote_usable"].fillna(False).astype(bool)
    away_ok = quotes["away_quote_usable"].fillna(False).astype(bool)

    both = quotes[home_ok & away_ok]
    home_only = quotes[home_ok & ~away_ok]
    away_only = quotes[~home_ok & away_ok]
    neither = quotes[~home_ok & ~away_ok]

    issues = pd.concat([quotes["home_quote_issue"], quotes["away_quote_issue"]])
    issue_counts = {
        str(code): int((issues == code).sum()) for code in QUOTE_ISSUES
    }

    example_cols = [
        "nba_game_id", "game_date", "home_team", "away_team", "prediction_ts_utc",
        "home_market_ticker", "home_quote_issue", "home_quote_age_seconds",
        "home_yes_bid", "home_yes_ask", "home_candles_in_window",
        "away_market_ticker", "away_quote_issue", "away_quote_age_seconds",
        "away_yes_bid", "away_yes_ask", "away_candles_in_window",
    ]

    failure_examples: dict[str, Any] = {}
    for code in QUOTE_ISSUES:
        affected = quotes[
            (quotes["home_quote_issue"] == code) | (quotes["away_quote_issue"] == code)
        ]
        if len(affected):
            failure_examples[code] = {
                "games_affected": len(affected),
                "examples": _examples(affected, example_cols, REPORT_EXAMPLE_LIMIT),
            }

    ages = pd.concat([quotes["home_quote_age_seconds"], quotes["away_quote_age_seconds"]])
    spreads = pd.concat([quotes["home_spread"], quotes["away_spread"]])
    midpoint_sum = quotes["market_midpoint_sum"]
    deviation = quotes["midpoint_sum_deviation_from_one"]

    deviation_counts = {
        f"gt_{threshold}": int((deviation > threshold).sum())
        for threshold in MIDPOINT_DEVIATION_THRESHOLDS
    }

    return {
        "generated_at_utc": utc_now().isoformat(),
        "started_at_utc": started_at,
        "season": season,
        "season_label": season_label(season),
        "configuration": {
            "minutes_before_tip": minutes_before_tip,
            "lookback_minutes": lookback_minutes,
            "max_quote_age_seconds": max_quote_age_seconds,
            "period_interval_minutes": period_interval,
            "scope": (
                "matched games with game_phase == 'regular_season' only; play-in, "
                "playoffs, and the NBA Cup final are excluded"
            ),
        },
        "phase_selection": phase_selection,
        "kalshi_historical_cutoff": cutoff,
        "coverage": {
            "eligible_regular_season_matched_games": total,
            "both_sides_usable": len(both),
            "home_only_usable": len(home_only),
            "away_only_usable": len(away_only),
            "neither_usable": len(neither),
            "both_sides_usable_pct": round(100.0 * len(both) / total, 3) if total else None,
        },
        "quote_issues": issue_counts,
        "quote_issue_examples": failure_examples,
        "candle_sources": {
            str(k): int(v)
            for k, v in pd.concat(
                [quotes["home_candle_source"], quotes["away_candle_source"]]
            ).value_counts().to_dict().items()
        },
        "distributions": {
            "quote_age_seconds": _percentiles(ages),
            "spread_dollars": _percentiles(spreads),
            "market_midpoint_sum": _percentiles(midpoint_sum),
            "midpoint_sum_deviation_from_one": _percentiles(deviation),
        },
        "midpoint_sum_deviation_counts": deviation_counts,
        "cache": cache_stats,
        "lookahead_guard": {
            "selected_quotes_after_prediction_ts": int(
                (pd.to_numeric(ages, errors="coerce") < 0).sum()
            ),
            "note": (
                "Must be 0. quote_age_seconds = prediction_ts - candle end_period_ts, "
                "so any negative value would mean a candle from after the anchor was used."
            ),
        },
    }


# --- orchestration ---------------------------------------------------------


def run_pipeline(
    season: int,
    *,
    settings: Settings | None = None,
    minutes_before_tip: int = DEFAULT_MINUTES_BEFORE_TIP,
    lookback_minutes: int = DEFAULT_QUOTE_LOOKBACK_MINUTES,
    max_quote_age_minutes: float = DEFAULT_MAX_QUOTE_AGE_MINUTES,
    period_interval: int = DEFAULT_CANDLE_PERIOD_INTERVAL,
    series_ticker: str = KALSHI_NBA_SERIES_TICKER,
    refresh: bool = False,
    write_csv: bool = True,
    limit: int | None = None,
    client: KalshiClient | None = None,
) -> PregameResult:
    """Execute Phase 2 and write the dataset plus the validation report."""
    settings = settings or load_settings()
    settings.paths.ensure()
    started_at = utc_now().isoformat()
    slug = season_slug(season)

    matches_path = settings.paths.processed / f"nba_kalshi_matches_{slug}.parquet"
    markets_path = settings.paths.processed / f"kalshi_nba_markets_{slug}.parquet"
    for path in (matches_path, markets_path):
        if not path.is_file():
            raise ConfigError(
                f"Missing Phase 1 output {path}. Run the Phase 1 pipeline first: "
                f"python -m nba_prediction_market.pipelines.build_dataset --season {season}"
            )

    matches = pd.read_parquet(matches_path)
    markets = pd.read_parquet(markets_path)
    eligible = select_eligible_games(matches, season)
    if limit is not None:
        eligible = eligible.head(limit)

    # Record what the phase filter did, and audit it against league structure.
    season_matches = matches[matches["nba_season"] == season]
    matched_only = season_matches[season_matches["match_status"] == STATUS_MATCHED]
    phase_selection = {
        "selected_phase": PHASE_REGULAR_SEASON,
        "matched_games_by_phase": {
            str(k): int(v)
            for k, v in matched_only.get(
                "nba_game_phase", pd.Series(dtype=object)
            ).value_counts().to_dict().items()
        },
        "all_games_by_phase": {
            str(k): int(v)
            for k, v in season_matches.get(
                "nba_game_phase", pd.Series(dtype=object)
            ).value_counts().to_dict().items()
        },
        "regular_season_audit": verify_regular_season(
            [
                {
                    "game_phase": r.get("nba_game_phase"),
                    "home_team_code": r.get("nba_home_team_code"),
                    "visitor_team_code": r.get("nba_away_team_code"),
                }
                for r in season_matches.to_dict("records")
            ],
            season,
        ),
    }
    logger.info(
        "Eligible matched regular-season games for %s: %d (phase breakdown: %s)",
        season_label(season), len(eligible), phase_selection["matched_games_by_phase"],
    )

    max_quote_age_seconds = float(max_quote_age_minutes) * 60.0
    cache = CandleCache(
        settings.paths.raw_kalshi / "candlesticks",
        cache_slug(
            minutes_before_tip=minutes_before_tip,
            lookback_minutes=lookback_minutes,
            period_interval=period_interval,
        ),
    )

    owns_client = client is None
    kalshi = client or KalshiClient(
        base_url=settings.kalshi_base_url,
        timeout=settings.request_timeout,
        min_interval=settings.kalshi_min_interval,
        max_retries=settings.max_retries,
    )
    try:
        cutoff = kalshi.get_historical_cutoff()
        cutoff_utc = parse_cutoff_ts(cutoff)
        logger.info("Kalshi historical cutoff (market_settled_ts): %s", cutoff_utc)

        quotes = build_pregame_frame(
            eligible,
            markets,
            client=kalshi,
            cache=cache,
            cutoff_utc=cutoff_utc,
            minutes_before_tip=minutes_before_tip,
            lookback_minutes=lookback_minutes,
            max_quote_age_seconds=max_quote_age_seconds,
            period_interval=period_interval,
            series_ticker=series_ticker,
            refresh=refresh,
        )
    finally:
        if owns_client:
            kalshi.close()

    written: list[Path] = []
    stem = f"nba_kalshi_pregame_t{minutes_before_tip}_{slug}"
    parquet_path = settings.paths.processed / f"{stem}.parquet"
    quotes.to_parquet(parquet_path, index=False)
    written.append(parquet_path)
    if write_csv:
        csv_path = settings.paths.processed / f"{stem}.csv"
        quotes.to_csv(csv_path, index=False)
        written.append(csv_path)

    report = build_report(
        quotes,
        season=season,
        phase_selection=phase_selection,
        minutes_before_tip=minutes_before_tip,
        lookback_minutes=lookback_minutes,
        max_quote_age_seconds=max_quote_age_seconds,
        period_interval=period_interval,
        cutoff=cutoff,
        cache_stats=cache.stats.to_dict(),
        started_at=started_at,
    )
    report_path = settings.paths.reports / f"pregame_t{minutes_before_tip}_report.json"
    report_path.write_text(json.dumps(report, indent=2, default=_json_default), encoding="utf-8")
    written.append(report_path)

    return PregameResult(quotes=quotes, report=report, written_files=written)


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


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m nba_prediction_market.pipelines.build_pregame_quotes",
        description=(
            "Phase 2: extract Kalshi pregame quotes N minutes before tipoff for matched "
            "regular-season NBA games."
        ),
    )
    parser.add_argument("--season", type=int, default=2025,
                        help="Season start year; 2025 means 2025-26 (default: 2025).")
    parser.add_argument(
        "--minutes-before-tip", type=int, default=DEFAULT_MINUTES_BEFORE_TIP,
        help=f"Anchor offset before tipoff (default: {DEFAULT_MINUTES_BEFORE_TIP}).",
    )
    parser.add_argument("--max-quote-age-minutes", type=float,
                        default=DEFAULT_MAX_QUOTE_AGE_MINUTES,
                        help=f"Staleness limit (default: {DEFAULT_MAX_QUOTE_AGE_MINUTES}).")
    parser.add_argument(
        "--lookback-minutes", type=int, default=DEFAULT_QUOTE_LOOKBACK_MINUTES,
        help=f"Candle window length (default: {DEFAULT_QUOTE_LOOKBACK_MINUTES}).",
    )
    parser.add_argument("--period-interval", type=int, default=DEFAULT_CANDLE_PERIOD_INTERVAL,
                        help="Candle granularity in minutes (default: 1).")
    parser.add_argument("--series-ticker", default=KALSHI_NBA_SERIES_TICKER)
    parser.add_argument("--data-dir", default=None,
                        help="Root directory for raw/processed/report output (default: ./data).")
    parser.add_argument("--refresh", action="store_true",
                        help="Ignore cached candle responses and refetch every market.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Process only the first N eligible games (for smoke runs).")
    parser.add_argument("--no-csv", action="store_true", help="Write only parquet, skip CSV.")
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
            args.season,
            settings=settings,
            minutes_before_tip=args.minutes_before_tip,
            lookback_minutes=args.lookback_minutes,
            max_quote_age_minutes=args.max_quote_age_minutes,
            period_interval=args.period_interval,
            series_ticker=args.series_ticker,
            refresh=args.refresh,
            write_csv=not args.no_csv,
            limit=args.limit,
        )
    except ConfigError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2

    coverage = result.report["coverage"]
    ages = result.report["distributions"]["quote_age_seconds"]
    print()
    print(f"Season {result.report['season_label']} | T-{args.minutes_before_tip}min")
    print(f"  eligible games      : {coverage['eligible_regular_season_matched_games']}")
    print(f"  both sides usable   : {coverage['both_sides_usable']} "
          f"({coverage['both_sides_usable_pct']}%)")
    print(f"  home only usable    : {coverage['home_only_usable']}")
    print(f"  away only usable    : {coverage['away_only_usable']}")
    print(f"  neither usable      : {coverage['neither_usable']}")
    if ages.get("count"):
        print(f"  quote age (s)       : median {ages['p50']:.0f}, p95 {ages['p95']:.0f}, "
              f"max {ages['max']:.0f}")
    print(f"  quotes after anchor : "
          f"{result.report['lookahead_guard']['selected_quotes_after_prediction_ts']} (must be 0)")
    print()
    print("Files written:")
    for path in result.written_files:
        print(f"  {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
