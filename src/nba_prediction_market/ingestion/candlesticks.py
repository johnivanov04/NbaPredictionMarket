"""Parse Kalshi candlesticks and select a lookahead-safe pregame quote.

Two endpoints serve the same candles under **different field names** (verified
against live data on 2026-08-19):

======================  ==========================  ============================
value                   ``/historical/markets/...``  ``/series/{s}/markets/...``
======================  ==========================  ============================
traded price (close)    ``price.close``             ``price.close_dollars``
previous traded price   ``price.previous``          ``price.previous_dollars``
yes bid (close)         ``yes_bid.close``           ``yes_bid.close_dollars``
yes ask (close)         ``yes_ask.close``           ``yes_ask.close_dollars``
volume                  ``volume``                  ``volume_fp``
open interest           ``open_interest``           ``open_interest_fp``
======================  ==========================  ============================

Both spellings are accepted so routing between tiers cannot silently produce
null columns. Prices already arrive as decimal dollars in ``[0, 1]`` (as
strings) -- there is no cent conversion to do, and values outside that range are
rejected rather than clamped.

Lookahead safety is enforced in one place: :func:`select_pregame_quote` filters
on ``end_period_ts <= prediction_ts`` before doing anything else, so no value
from after the prediction instant can reach the output.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

logger = logging.getLogger(__name__)

#: Decimals kept on derived prices. Kalshi quotes whole cents, so midpoints land
#: on half-cents and spreads on cents; 6dp is lossless for every real value
#: while removing binary-float noise (0.01 arriving as 0.010000000000000009),
#: which would otherwise make threshold comparisons and grouping unreliable
#: downstream.
DERIVED_PRICE_DECIMALS = 6

# --- issue codes -----------------------------------------------------------

ISSUE_NO_MARKET_TICKER = "no_market_ticker"
ISSUE_TEAM_MISMATCH = "market_team_mismatch"
ISSUE_FETCH_FAILED = "fetch_failed"
ISSUE_NO_CANDLES = "no_candles_returned"
ISSUE_NO_CANDLE_BEFORE = "no_candle_at_or_before_prediction_ts"
ISSUE_NO_QUOTE_DATA = "no_bid_or_ask_in_window"
ISSUE_STALE = "stale_quote"
ISSUE_MISSING_BID = "missing_bid"
ISSUE_MISSING_ASK = "missing_ask"

#: Every issue code, for report scaffolding and tests.
QUOTE_ISSUES: tuple[str, ...] = (
    ISSUE_NO_MARKET_TICKER,
    ISSUE_TEAM_MISMATCH,
    ISSUE_FETCH_FAILED,
    ISSUE_NO_CANDLES,
    ISSUE_NO_CANDLE_BEFORE,
    ISSUE_NO_QUOTE_DATA,
    ISSUE_STALE,
    ISSUE_MISSING_BID,
    ISSUE_MISSING_ASK,
)


def _pick(payload: Any, *names: str) -> Any:
    """First present, non-null value among ``names`` -- tolerating both schemas."""
    if not isinstance(payload, dict):
        return None
    for name in names:
        value = payload.get(name)
        if value is not None:
            return value
    return None


def parse_price(value: Any) -> float | None:
    """Parse a Kalshi price into decimal dollars, or ``None`` if unusable.

    Kalshi sends prices as decimal strings already in ``[0, 1]``. Anything that
    does not parse, or lands outside that range, returns ``None`` -- a bad price
    must never be silently rescaled into a plausible-looking probability.
    """
    if value is None or isinstance(value, bool):
        return None
    try:
        price = float(value)
    except (TypeError, ValueError):
        return None
    if price != price:  # NaN
        return None
    if not 0.0 <= price <= 1.0:
        logger.warning("Discarding out-of-range Kalshi price %r (expected 0..1)", value)
        return None
    return price


def _parse_amount(value: Any) -> float | None:
    """Parse a non-price numeric (volume, open interest); no range constraint."""
    if value is None or isinstance(value, bool):
        return None
    try:
        amount = float(value)
    except (TypeError, ValueError):
        return None
    return None if amount != amount else amount


@dataclass(frozen=True)
class Candle:
    """One normalized 1-minute candle, keyed by the end of its period."""

    end_period_ts: int
    yes_bid: float | None
    yes_ask: float | None
    last_trade_price: float | None
    previous_trade_price: float | None
    volume: float | None
    open_interest: float | None

    @property
    def end_ts_utc(self) -> datetime:
        return datetime.fromtimestamp(self.end_period_ts, tz=UTC)

    @property
    def has_quote(self) -> bool:
        """True when at least one side of the book is quoted."""
        return self.yes_bid is not None or self.yes_ask is not None

    @property
    def midpoint(self) -> float | None:
        """Mid of the two-sided quote, or ``None`` -- never synthesised from trades."""
        if self.yes_bid is None or self.yes_ask is None:
            return None
        return round((self.yes_bid + self.yes_ask) / 2.0, DERIVED_PRICE_DECIMALS)

    @property
    def spread(self) -> float | None:
        if self.yes_bid is None or self.yes_ask is None:
            return None
        return round(self.yes_ask - self.yes_bid, DERIVED_PRICE_DECIMALS)


def parse_candle(raw: Any) -> Candle | None:
    """Normalize one raw candle. Returns ``None`` if it is malformed."""
    if not isinstance(raw, dict):
        return None
    end_ts = raw.get("end_period_ts")
    if isinstance(end_ts, bool) or not isinstance(end_ts, int | float):
        return None
    try:
        end_period_ts = int(end_ts)
    except (TypeError, ValueError):
        return None

    price = raw.get("price") if isinstance(raw.get("price"), dict) else {}
    bid = raw.get("yes_bid") if isinstance(raw.get("yes_bid"), dict) else {}
    ask = raw.get("yes_ask") if isinstance(raw.get("yes_ask"), dict) else {}

    return Candle(
        end_period_ts=end_period_ts,
        yes_bid=parse_price(_pick(bid, "close", "close_dollars")),
        yes_ask=parse_price(_pick(ask, "close", "close_dollars")),
        last_trade_price=parse_price(_pick(price, "close", "close_dollars")),
        previous_trade_price=parse_price(_pick(price, "previous", "previous_dollars")),
        volume=_parse_amount(_pick(raw, "volume", "volume_fp")),
        open_interest=_parse_amount(_pick(raw, "open_interest", "open_interest_fp")),
    )


def parse_candles(payload: Any) -> tuple[list[Candle], int]:
    """Normalize a candlestick payload.

    Returns ``(candles, malformed_count)``. Candles are sorted by
    ``end_period_ts`` and de-duplicated, keeping the **last** occurrence of a
    repeated timestamp (the later record in the response wins).
    """
    if not isinstance(payload, dict):
        return [], 0
    raw_candles = payload.get("candlesticks")
    if not isinstance(raw_candles, list):
        return [], 0

    by_ts: dict[int, Candle] = {}
    malformed = 0
    for raw in raw_candles:
        candle = parse_candle(raw)
        if candle is None:
            malformed += 1
            continue
        by_ts[candle.end_period_ts] = candle
    return [by_ts[ts] for ts in sorted(by_ts)], malformed


@dataclass(frozen=True)
class QuoteSelection:
    """The chosen pregame quote plus why it is (or is not) usable."""

    candle: Candle | None
    quote_age_seconds: float | None
    usable: bool
    issue: str | None
    candles_considered: int = 0
    malformed_candles: int = 0

    @property
    def quote_ts_utc(self) -> datetime | None:
        return self.candle.end_ts_utc if self.candle else None


def select_pregame_quote(
    candles: list[Candle],
    prediction_ts: datetime,
    *,
    max_age_seconds: float,
    malformed_candles: int = 0,
) -> QuoteSelection:
    """Pick the most recent real candle at or before ``prediction_ts``.

    Guarantees, in order:

    0. The input is sorted by ``end_period_ts`` first, so "most recent" never
       depends on the order the caller happened to supply.
    1. Candles after ``prediction_ts`` are discarded outright -- this is the
       single chokepoint that prevents lookahead.
    2. Among what remains, the latest candle that actually carries a quote wins.
       A quote is never forward-filled from a later candle.
    3. ``quote_age_seconds`` is always reported, even when unusable, so staleness
       is measurable rather than invisible.
    4. A quote older than ``max_age_seconds`` is kept for diagnostics but marked
       unusable. Age exactly equal to the limit is still usable.
    5. ``usable`` requires **both** sides quoted, because the midpoint is the
       stated probability benchmark and is only defined two-sided.
    """
    if prediction_ts.tzinfo is None:
        raise ValueError("prediction_ts must be timezone-aware (UTC)")
    cutoff = int(prediction_ts.astimezone(UTC).timestamp())

    def result(candle: Candle | None, issue: str | None, *, usable: bool = False) -> QuoteSelection:
        age = None if candle is None else float(cutoff - candle.end_period_ts)
        return QuoteSelection(
            candle=candle,
            quote_age_seconds=age,
            usable=usable,
            issue=issue,
            candles_considered=len(candles),
            malformed_candles=malformed_candles,
        )

    if not candles:
        return result(None, ISSUE_NO_CANDLES)

    # Sort defensively: "most recent" must not depend on the caller having
    # ordered the list. A stable sort keeps the later of two duplicate
    # timestamps last, matching parse_candles' de-duplication rule.
    ordered = sorted(candles, key=lambda c: c.end_period_ts)
    eligible = [c for c in ordered if c.end_period_ts <= cutoff]
    if not eligible:
        return result(None, ISSUE_NO_CANDLE_BEFORE)

    quoted = [c for c in eligible if c.has_quote]
    if not quoted:
        # Keep the newest eligible candle so the gap is visible in diagnostics.
        return result(eligible[-1], ISSUE_NO_QUOTE_DATA)

    chosen = quoted[-1]
    age = cutoff - chosen.end_period_ts
    if age > max_age_seconds:
        return result(chosen, ISSUE_STALE)
    if chosen.yes_bid is None:
        return result(chosen, ISSUE_MISSING_BID)
    if chosen.yes_ask is None:
        return result(chosen, ISSUE_MISSING_ASK)
    return result(chosen, None, usable=True)


def prediction_timestamp(tipoff_utc: datetime, minutes_before_tip: int) -> datetime:
    """Anchor = scheduled tipoff minus ``minutes_before_tip``, in UTC.

    The NBA tipoff is the source of truth; Kalshi close/settlement times are
    never used for the anchor. Naive input is rejected rather than assumed.
    """
    if tipoff_utc.tzinfo is None:
        raise ValueError("tipoff_utc must be timezone-aware; refusing to guess a zone")
    if minutes_before_tip < 0:
        raise ValueError(f"minutes_before_tip must be >= 0, got {minutes_before_tip}")
    return tipoff_utc.astimezone(UTC) - timedelta(minutes=minutes_before_tip)
