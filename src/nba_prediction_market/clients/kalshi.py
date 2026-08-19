"""Kalshi trade-api v2 client for public market/event metadata.

Kalshi splits a series into two stores: settled markets older than the
``/historical/cutoff`` boundary are served by ``/historical/markets``, while the
live store ``/markets`` only retains a recent window. Reading both and
deduplicating by ticker is the only way to be sure a season is complete.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterator
from typing import Any

import httpx

from nba_prediction_market.clients.base import BaseApiClient
from nba_prediction_market.config import (
    DEFAULT_KALSHI_MIN_INTERVAL,
    KALSHI_BASE_URL,
    KALSHI_EVENTS_LIMIT,
    KALSHI_MARKETS_LIMIT,
    KALSHI_NBA_SERIES_TICKER,
)

logger = logging.getLogger(__name__)

SOURCE_HISTORICAL = "historical"
SOURCE_MARKETS = "markets"


def _cursor(payload: dict[str, Any]) -> Any:
    return payload.get("cursor")


class KalshiClient(BaseApiClient):
    """Reader for Kalshi series metadata. Public endpoints need no credentials."""

    def __init__(
        self,
        *,
        base_url: str = KALSHI_BASE_URL,
        timeout: float = 60.0,
        min_interval: float = DEFAULT_KALSHI_MIN_INTERVAL,
        max_retries: int = 5,
        client: httpx.Client | None = None,
    ) -> None:
        super().__init__(
            base_url,
            headers={"Accept": "application/json"},
            timeout=timeout,
            min_interval=min_interval,
            max_retries=max_retries,
            client=client,
        )

    def get_historical_cutoff(self) -> dict[str, Any]:
        """Return ``GET /historical/cutoff``: the archive boundary timestamps."""
        return self.get_json("/historical/cutoff")

    def iter_historical_markets(
        self,
        series_ticker: str = KALSHI_NBA_SERIES_TICKER,
        *,
        limit: int = KALSHI_MARKETS_LIMIT,
        on_page: Callable[[int, dict[str, Any]], None] | None = None,
    ) -> Iterator[list[dict[str, Any]]]:
        yield from self.paginate(
            "/historical/markets",
            params={"series_ticker": series_ticker, "limit": limit},
            items_key="markets",
            next_cursor=_cursor,
            on_page=on_page,
        )

    def iter_markets(
        self,
        series_ticker: str = KALSHI_NBA_SERIES_TICKER,
        *,
        limit: int = KALSHI_MARKETS_LIMIT,
        status: str | None = None,
        on_page: Callable[[int, dict[str, Any]], None] | None = None,
    ) -> Iterator[list[dict[str, Any]]]:
        """Yield pages from the live ``/markets`` store.

        ``status`` is omitted by default on purpose: the documented ``status=all``
        value is rejected with HTTP 400 ``invalid status filter``, whereas sending
        no filter returns every status the endpoint holds.
        """
        params: dict[str, Any] = {"series_ticker": series_ticker, "limit": limit}
        if status is not None:
            params["status"] = status
        yield from self.paginate(
            "/markets",
            params=params,
            items_key="markets",
            next_cursor=_cursor,
            on_page=on_page,
        )

    def iter_events(
        self,
        series_ticker: str = KALSHI_NBA_SERIES_TICKER,
        *,
        limit: int = KALSHI_EVENTS_LIMIT,
        on_page: Callable[[int, dict[str, Any]], None] | None = None,
    ) -> Iterator[list[dict[str, Any]]]:
        """Yield pages of events. ``/events`` caps ``limit`` well below /markets."""
        yield from self.paginate(
            "/events",
            params={"series_ticker": series_ticker, "limit": limit},
            items_key="events",
            next_cursor=_cursor,
            on_page=on_page,
        )

    def fetch_markets_from_both_stores(
        self,
        series_ticker: str = KALSHI_NBA_SERIES_TICKER,
        *,
        on_page: Callable[[str, int, dict[str, Any]], None] | None = None,
    ) -> tuple[list[dict[str, Any]], dict[str, list[str]]]:
        """Fetch and merge the historical and live market stores.

        Returns the deduplicated markets plus a ``ticker -> [source, ...]`` index
        so provenance and cross-store overlap stay auditable. When a ticker is
        present in both stores the historical record wins, because it is the
        stable archival copy.
        """
        by_ticker: dict[str, dict[str, Any]] = {}
        sources: dict[str, list[str]] = {}

        def ingest(source: str, markets: list[dict[str, Any]]) -> None:
            for market in markets:
                ticker = market.get("ticker")
                if not ticker:
                    logger.warning("Skipping %s market without a ticker", source)
                    continue
                sources.setdefault(ticker, [])
                if source not in sources[ticker]:
                    sources[ticker].append(source)
                # Historical is authoritative; never let a live record overwrite it.
                if ticker not in by_ticker or source == SOURCE_HISTORICAL:
                    by_ticker[ticker] = market

        for source, iterator in (
            (SOURCE_HISTORICAL, self.iter_historical_markets(series_ticker)),
            (SOURCE_MARKETS, self.iter_markets(series_ticker)),
        ):
            for page_number, page in enumerate(iterator, start=1):
                if on_page is not None:
                    on_page(source, page_number, {"markets": page})
                ingest(source, page)

        logger.info(
            "Kalshi %s: %d unique markets (%d seen in both stores)",
            series_ticker,
            len(by_ticker),
            sum(1 for s in sources.values() if len(s) > 1),
        )
        return list(by_ticker.values()), sources

    def fetch_events(
        self, series_ticker: str = KALSHI_NBA_SERIES_TICKER
    ) -> list[dict[str, Any]]:
        events: list[dict[str, Any]] = []
        for page in self.iter_events(series_ticker):
            events.extend(page)
        return events
