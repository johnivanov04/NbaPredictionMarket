"""Live smoke checks for the Kalshi candlestick contracts (opt-in).

Deselected by default. Run deliberately with::

    pytest -m integration

These assert the *shape* of the responses -- the assumptions Phase 2 is built
on -- so an upstream change is caught rather than silently corrupting a dataset.
"""

from __future__ import annotations

from itertools import pairwise

import pytest

from nba_prediction_market.clients.base import ApiError
from nba_prediction_market.clients.kalshi import KalshiClient
from nba_prediction_market.ingestion.candlesticks import parse_candles

pytestmark = pytest.mark.integration

#: A settled 2025-26 playoff market, and a one-hour window ending 30 minutes
#: before its scheduled tipoff (2026-06-14T03:30:00Z).
TICKER = "KXNBAGAME-26JUN13NYKSAS-SAS"
START_TS = 1781402400
END_TS = 1781406000


@pytest.fixture(scope="module")
def client() -> KalshiClient:
    with KalshiClient(min_interval=0.0) as live:
        yield live


def test_historical_candlesticks_have_the_fields_we_parse(client: KalshiClient) -> None:
    payload = client.get_historical_candlesticks(
        TICKER, start_ts=START_TS, end_ts=END_TS, period_interval=1
    )
    assert "candlesticks" in payload
    raw = payload["candlesticks"][0]
    for field in ("end_period_ts", "price", "yes_bid", "yes_ask", "volume", "open_interest"):
        assert field in raw, f"historical candles dropped {field!r}"
    assert "close" in raw["yes_bid"], "historical candles switched to suffixed field names"

    candles, malformed = parse_candles(payload)
    assert malformed == 0
    assert candles[0].yes_bid is not None and candles[0].yes_ask is not None


def test_series_candlesticks_use_suffixed_field_names(client: KalshiClient) -> None:
    """The live endpoint names the same values differently; both must parse."""
    payload = client.get_series_candlesticks(TICKER, start_ts=START_TS, end_ts=END_TS)
    raw = payload["candlesticks"][0]
    assert "volume_fp" in raw
    assert "close_dollars" in raw["yes_bid"]

    candles, malformed = parse_candles(payload)
    assert malformed == 0
    assert candles[0].yes_bid is not None


def test_both_endpoints_agree_on_the_same_window(client: KalshiClient) -> None:
    historical, _ = parse_candles(
        client.get_historical_candlesticks(TICKER, start_ts=START_TS, end_ts=END_TS)
    )
    series, _ = parse_candles(
        client.get_series_candlesticks(TICKER, start_ts=START_TS, end_ts=END_TS)
    )
    assert [c.end_period_ts for c in historical] == [c.end_period_ts for c in series]
    assert [c.yes_bid for c in historical] == [c.yes_bid for c in series]


def test_one_minute_candles_are_spaced_sixty_seconds_apart(client: KalshiClient) -> None:
    candles, _ = parse_candles(
        client.get_historical_candlesticks(
            TICKER, start_ts=START_TS, end_ts=END_TS, period_interval=1
        )
    )
    stamps = [c.end_period_ts for c in candles]
    assert stamps == sorted(stamps)
    assert {b - a for a, b in pairwise(stamps)} == {60}


def test_the_window_is_inclusive_of_both_bounds(client: KalshiClient) -> None:
    candles, _ = parse_candles(
        client.get_historical_candlesticks(TICKER, start_ts=START_TS, end_ts=END_TS)
    )
    assert candles[0].end_period_ts == START_TS
    assert candles[-1].end_period_ts == END_TS
    # Nothing past the requested end may come back.
    assert all(c.end_period_ts <= END_TS for c in candles)


def test_prices_arrive_as_decimal_dollars_not_cents(client: KalshiClient) -> None:
    """Parsing assumes [0, 1]; a switch to cents would silently break everything."""
    candles, _ = parse_candles(
        client.get_historical_candlesticks(TICKER, start_ts=START_TS, end_ts=END_TS)
    )
    quoted = [c for c in candles if c.yes_bid is not None and c.yes_ask is not None]
    assert quoted, "expected at least one two-sided quote"
    for candle in quoted:
        assert 0.0 <= candle.yes_bid <= 1.0
        assert 0.0 <= candle.yes_ask <= 1.0
        assert candle.yes_ask >= candle.yes_bid, "ask below bid is a crossed book"


def test_candlesticks_need_no_authentication() -> None:
    """No credentials are configured anywhere in this project."""
    with KalshiClient(min_interval=0.0) as anonymous:
        payload = anonymous.get_historical_candlesticks(
            TICKER, start_ts=START_TS, end_ts=END_TS
        )
    assert payload["candlesticks"]


def test_unknown_ticker_returns_404(client: KalshiClient) -> None:
    """Drives the archive-to-live fallback, so the status code matters."""
    with pytest.raises(ApiError) as excinfo:
        client.get_historical_candlesticks(
            "KXNBAGAME-99XXX99AAABBB-AAA", start_ts=START_TS, end_ts=END_TS
        )
    assert excinfo.value.status_code == 404


def test_invalid_period_interval_is_rejected_by_the_server() -> None:
    """Documents why the client validates period_interval before sending."""
    import httpx

    response = httpx.get(
        f"https://external-api.kalshi.com/trade-api/v2/historical/markets/{TICKER}/candlesticks",
        params={"start_ts": START_TS, "end_ts": END_TS, "period_interval": 7},
        timeout=30,
    )
    assert response.status_code == 400
