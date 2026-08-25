"""Live Kalshi market capture for prospective research.

Phase 4A1 worked from one-minute candles. That was enough to show the market
adjusts to injury news over 5-15 minutes, but not to measure *how* it adjusts,
because a candle is an aggregate and hides the book.

The public REST orderbook gives full depth on both sides without credentials,
which is a genuine upgrade. Two facts about it shape this module:

* **The book returns bids only.** Kalshi lists YES bids and NO bids; there is
  no ask array, because a NO bid at $0.43 *is* a YES ask at $0.57. The ask a
  buyer would pay is therefore derived, and derived exactly rather than
  approximated from a midpoint.
* **Depth is real and must not be invented.** Only levels the exchange returned
  are recorded. An absent level is absent, not zero.

A WebSocket exists (``orderbook_delta``, ``market_ticker``) and would give
push-based updates, but **Kalshi requires API-key authentication even for
public market-data channels**. Without credentials the collector polls, and it
says so rather than pretending to stream.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

#: Public REST base. Reachable without credentials.
PUBLIC_REST_BASE: str = "https://api.elections.kalshi.com/trade-api/v2"
#: Streaming endpoint, for reference. Requires API-key authentication.
WEBSOCKET_URL: str = "wss://api.elections.kalshi.com/trade-api/ws/v2"
WEBSOCKET_CHANNELS: tuple[str, ...] = ("orderbook_delta", "market_ticker")
WEBSOCKET_REQUIRES_AUTH: bool = True

#: Contracts settle at $1, so the two sides of the book are complementary.
CONTRACT_SETTLEMENT_DOLLARS: float = 1.0


def _levels(raw: Any) -> list[tuple[float, float]]:
    """Parse ``[[price, quantity], ...]`` into floats, best level last."""
    if not isinstance(raw, list):
        return []
    out: list[tuple[float, float]] = []
    for item in raw:
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            try:
                out.append((float(item[0]), float(item[1])))
            except (TypeError, ValueError):
                continue
    return sorted(out)


def derive_yes_ask(no_bid_levels: list[tuple[float, float]]) -> tuple[float, float] | None:
    """The best YES ask implied by the NO side of the book.

    A NO bid at price *y* is a standing offer to sell YES at ``1 - y``. The
    best (highest) NO bid therefore gives the cheapest YES ask, which is the
    price a YES buyer would actually pay.
    """
    if not no_bid_levels:
        return None
    price, quantity = no_bid_levels[-1]
    return (round(CONTRACT_SETTLEMENT_DOLLARS - price, 6), quantity)


@dataclass(frozen=True)
class BookSnapshot:
    """One market's order book at one observed instant."""

    market_ticker: str
    observed_at_utc: datetime
    exchange_updated_at_utc: datetime | None
    yes_bids: list[tuple[float, float]] = field(default_factory=list)
    no_bids: list[tuple[float, float]] = field(default_factory=list)
    last_trade_price: float | None = None
    volume: float | None = None
    open_interest: float | None = None
    status: str | None = None

    @property
    def best_yes_bid(self) -> tuple[float, float] | None:
        return self.yes_bids[-1] if self.yes_bids else None

    @property
    def best_yes_ask(self) -> tuple[float, float] | None:
        return derive_yes_ask(self.no_bids)

    @property
    def midpoint(self) -> float | None:
        bid, ask = self.best_yes_bid, self.best_yes_ask
        if bid is None or ask is None:
            return None
        return round((bid[0] + ask[0]) / 2.0, 6)

    @property
    def spread(self) -> float | None:
        bid, ask = self.best_yes_bid, self.best_yes_ask
        if bid is None or ask is None:
            return None
        return round(ask[0] - bid[0], 6)

    def depth_within(self, cents: float) -> dict[str, float]:
        """Resting size within ``cents`` of each side's best price.

        Only observed levels are counted. This is the honest answer to "how
        much could have traded near the touch", and it is the thing candles
        could never answer.
        """
        result = {"yes_bid_size": 0.0, "yes_ask_size": 0.0}
        if self.yes_bids:
            best = self.yes_bids[-1][0]
            result["yes_bid_size"] = sum(
                q for p, q in self.yes_bids if best - p <= cents
            )
        if self.no_bids:
            best_no = self.no_bids[-1][0]
            result["yes_ask_size"] = sum(
                q for p, q in self.no_bids if best_no - p <= cents
            )
        return result

    def to_row(self) -> dict[str, Any]:
        bid, ask = self.best_yes_bid, self.best_yes_ask
        depth = self.depth_within(0.02)
        return {
            "market_ticker": self.market_ticker,
            "observed_at_utc": self.observed_at_utc,
            "exchange_updated_at_utc": self.exchange_updated_at_utc,
            "yes_bid": bid[0] if bid else None,
            "yes_bid_size": bid[1] if bid else None,
            "yes_ask": ask[0] if ask else None,
            "yes_ask_size": ask[1] if ask else None,
            "midpoint": self.midpoint,
            "spread": self.spread,
            "last_trade_price": self.last_trade_price,
            "volume": self.volume,
            "open_interest": self.open_interest,
            "status": self.status,
            "yes_bid_levels": len(self.yes_bids),
            "no_bid_levels": len(self.no_bids),
            "depth_2c_yes_bid": depth["yes_bid_size"],
            "depth_2c_yes_ask": depth["yes_ask_size"],
        }


def _parse_time(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)


def _number(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def parse_book(
    ticker: str,
    orderbook_payload: dict[str, Any],
    observed_at_utc: datetime,
    market_payload: dict[str, Any] | None = None,
) -> BookSnapshot:
    """Build a snapshot from the raw orderbook and market responses."""
    book = (
        orderbook_payload.get("orderbook_fp")
        or orderbook_payload.get("orderbook")
        or {}
    )
    market = (market_payload or {}).get("market", market_payload or {})
    return BookSnapshot(
        market_ticker=ticker,
        observed_at_utc=observed_at_utc,
        exchange_updated_at_utc=_parse_time(market.get("updated_time")),
        yes_bids=_levels(book.get("yes_dollars") or book.get("yes")),
        no_bids=_levels(book.get("no_dollars") or book.get("no")),
        last_trade_price=_number(market.get("last_price_dollars")),
        volume=_number(market.get("volume_fp") or market.get("volume")),
        open_interest=_number(
            market.get("open_interest_fp") or market.get("open_interest")
        ),
        status=market.get("status"),
    )


def streaming_capability() -> dict[str, Any]:
    """What streaming would offer, and why polling is the default."""
    return {
        "websocket_url": WEBSOCKET_URL,
        "channels": list(WEBSOCKET_CHANNELS),
        "requires_authentication": WEBSOCKET_REQUIRES_AUTH,
        "available_to_this_project": False,
        "reason": (
            "Kalshi requires API-key authentication during the WebSocket "
            "handshake even for public market-data channels, and this project "
            "holds no Kalshi credentials"
        ),
        "fallback": "public REST orderbook polling, which needs no credentials",
        "upgrade_path": (
            "supply KALSHI_API_KEY_ID and KALSHI_PRIVATE_KEY and the collector "
            "can subscribe to orderbook_delta instead of polling"
        ),
    }
