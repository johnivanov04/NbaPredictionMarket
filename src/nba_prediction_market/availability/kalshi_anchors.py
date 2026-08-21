"""Multi-anchor Kalshi market capture for prospective research.

Phase 2 selects a single T-30 quote from 1-minute candlesticks. Rather than
polling at each new anchor, the right move is to widen the *stored window*: one
request per market covering tip-24h..tip already contains every anchor we might
want, so a new anchor costs no extra requests and no refetch.

Anchor selection reuses the Phase 2 selector, which means the same lookahead
guarantee applies at every anchor: the chosen candle satisfies
``end_period_ts <= anchor``, never later.

No trading logic lives here or anywhere in the project, and Kalshi remains a
benchmark only -- never a model feature.

**The Phase 2 cache cannot serve these anchors retrospectively.** It was fetched
under the slug ``t30_lb60_p1``: a 60-minute window *ending* at T-30. That covers
T-1h and T-30m and nothing else -- T-24h, T-6h and T-3h fall before the window
starts, and T-15m and T-5m fall after it ends. Reconstructing the full anchor
set for 2025-26 therefore needs a refetch at ``CAPTURE_WINDOW_HOURS``, one
request per market. Because the cache slug is part of the cache path, that
refetch lands in its own directory and cannot overwrite the Phase 2 quotes.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Final

from nba_prediction_market.ingestion.candlesticks import (
    Candle,
    QuoteSelection,
    select_pregame_quote,
)

#: Research anchors, in minutes before scheduled tipoff.
RESEARCH_ANCHORS_MINUTES: Final[tuple[int, ...]] = (
    24 * 60, 6 * 60, 3 * 60, 60, 30, 15, 5,
)
#: Hours of market history to cache per market, covering every anchor above.
CAPTURE_WINDOW_HOURS: Final = 24
#: Anchors older than this are considered stale for their own anchor.
DEFAULT_MAX_QUOTE_AGE_SECONDS: Final = 600.0


def anchor_label(minutes_before_tip: int) -> str:
    """Human label for an anchor: ``T-24h``, ``T-30m``."""
    if minutes_before_tip % 60 == 0 and minutes_before_tip >= 60:
        return f"T-{minutes_before_tip // 60}h"
    return f"T-{minutes_before_tip}m"


def capture_window(scheduled_tipoff_utc: datetime) -> tuple[int, int]:
    """Unix-second window to request per market, covering every anchor."""
    if scheduled_tipoff_utc.tzinfo is None:
        raise ValueError("scheduled_tipoff_utc must be timezone-aware")
    end = int(scheduled_tipoff_utc.timestamp())
    start = int((scheduled_tipoff_utc - timedelta(hours=CAPTURE_WINDOW_HOURS)).timestamp())
    return start, end


@dataclass(frozen=True)
class AnchorQuote:
    """One market's state at one research anchor."""

    anchor_label: str
    minutes_before_tip: int
    anchor_utc: datetime
    yes_bid: float | None
    yes_ask: float | None
    midpoint: float | None
    spread: float | None
    last_trade_price: float | None
    previous_trade_price: float | None
    volume: float | None
    open_interest: float | None
    quote_ts_utc: datetime | None
    quote_age_seconds: float | None
    usable: bool
    issue: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "anchor": self.anchor_label,
            "minutes_before_tip": self.minutes_before_tip,
            "anchor_utc": self.anchor_utc.isoformat(),
            "yes_bid": self.yes_bid,
            "yes_ask": self.yes_ask,
            "midpoint": self.midpoint,
            "spread": self.spread,
            "last_trade_price": self.last_trade_price,
            "previous_trade_price": self.previous_trade_price,
            "volume": self.volume,
            "open_interest": self.open_interest,
            "quote_ts_utc": self.quote_ts_utc.isoformat() if self.quote_ts_utc else None,
            "quote_age_seconds": self.quote_age_seconds,
            "usable": self.usable,
            "issue": self.issue,
        }


def _from_selection(
    label: str, minutes: int, anchor: datetime, selection: QuoteSelection
) -> AnchorQuote:
    candle = selection.candle
    return AnchorQuote(
        anchor_label=label,
        minutes_before_tip=minutes,
        anchor_utc=anchor,
        yes_bid=candle.yes_bid if candle else None,
        yes_ask=candle.yes_ask if candle else None,
        midpoint=candle.midpoint if candle else None,
        spread=candle.spread if candle else None,
        last_trade_price=candle.last_trade_price if candle else None,
        previous_trade_price=candle.previous_trade_price if candle else None,
        volume=candle.volume if candle else None,
        open_interest=candle.open_interest if candle else None,
        quote_ts_utc=candle.end_ts_utc if candle else None,
        quote_age_seconds=selection.quote_age_seconds,
        usable=selection.usable,
        issue=selection.issue,
    )


def quotes_at_anchors(
    candles: Sequence[Candle],
    scheduled_tipoff_utc: datetime,
    *,
    anchors_minutes: Sequence[int] = RESEARCH_ANCHORS_MINUTES,
    max_quote_age_seconds: float = DEFAULT_MAX_QUOTE_AGE_SECONDS,
) -> list[AnchorQuote]:
    """Reconstruct every research anchor from one cached candle stream.

    Each anchor is resolved by the Phase 2 selector, so no candle from after an
    anchor can reach that anchor's quote.
    """
    if scheduled_tipoff_utc.tzinfo is None:
        raise ValueError("scheduled_tipoff_utc must be timezone-aware")
    out: list[AnchorQuote] = []
    for minutes in sorted(anchors_minutes, reverse=True):
        anchor = scheduled_tipoff_utc - timedelta(minutes=minutes)
        selection = select_pregame_quote(
            list(candles), anchor, max_age_seconds=max_quote_age_seconds
        )
        out.append(_from_selection(anchor_label(minutes), minutes, anchor, selection))
    return out
