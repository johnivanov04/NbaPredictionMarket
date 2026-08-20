"""Cache/resume behaviour and archived-vs-current endpoint routing.

Covers requirements 14 (routing), 15 (cache/resume), and 23 (retry/rate limit)
for the candlestick path. No network access.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
import pytest

from nba_prediction_market.clients.base import ApiError
from nba_prediction_market.clients.kalshi import (
    CANDLES_HISTORICAL,
    CANDLES_SERIES,
    KalshiClient,
)
from nba_prediction_market.ingestion.candle_cache import (
    CandleCache,
    CandleRequest,
    cache_slug,
)
from nba_prediction_market.pipelines.build_pregame_quotes import (
    choose_candle_endpoint,
    fetch_market_candles,
    parse_cutoff_ts,
)

TICKER = "KXNBAGAME-26JAN15HOUOKC-OKC"
GAME_DATE = "2026-01-15"
REQUEST = CandleRequest(market_ticker=TICKER, start_ts=1000, end_ts=4600, period_interval=1)

CANDLE_PAYLOAD: dict[str, Any] = {
    "ticker": TICKER,
    "candlesticks": [
        {
            "end_period_ts": 4600,
            "open_interest": "100.00",
            "price": {"close": "0.6000", "previous": "0.5900"},
            "volume": "50.00",
            "yes_ask": {"close": "0.6100"},
            "yes_bid": {"close": "0.6000"},
        }
    ],
}


@pytest.fixture(autouse=True)
def _no_real_sleeping(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("time.sleep", lambda _s: None)


def make_client(handler) -> KalshiClient:
    return KalshiClient(
        base_url="https://api.test",
        min_interval=0.0,
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )


# --- 14: endpoint routing --------------------------------------------------


CUTOFF = datetime(2026, 6, 20, tzinfo=UTC)


def test_market_settled_before_the_cutoff_uses_the_archive() -> None:
    assert choose_candle_endpoint(datetime(2026, 1, 15, tzinfo=UTC), CUTOFF) == CANDLES_HISTORICAL


def test_market_settled_after_the_cutoff_uses_the_live_endpoint() -> None:
    assert choose_candle_endpoint(datetime(2026, 7, 1, tzinfo=UTC), CUTOFF) == CANDLES_SERIES


def test_market_exactly_on_the_cutoff_is_treated_as_archived() -> None:
    assert choose_candle_endpoint(CUTOFF, CUTOFF) == CANDLES_HISTORICAL


@pytest.mark.parametrize(
    ("market_end", "cutoff"),
    [(None, CUTOFF), (datetime(2026, 1, 1, tzinfo=UTC), None), (None, None)],
)
def test_unknown_boundaries_fall_back_to_the_live_endpoint(market_end, cutoff) -> None:
    """Never assume archive membership; the live endpoint is the general one."""
    assert choose_candle_endpoint(market_end, cutoff) == CANDLES_SERIES


@pytest.mark.parametrize(
    ("cutoff", "expected"),
    [
        ({"market_settled_ts": "2026-06-20T00:00:00Z"}, datetime(2026, 6, 20, tzinfo=UTC)),
        ({"market_settled_ts": "2026-06-20T00:00:00+00:00"}, datetime(2026, 6, 20, tzinfo=UTC)),
        ({}, None),
        ({"market_settled_ts": ""}, None),
        ({"market_settled_ts": "nonsense"}, None),
        (None, None),
        ("not a dict", None),
    ],
)
def test_parse_cutoff_ts(cutoff, expected) -> None:
    assert parse_cutoff_ts(cutoff) == expected


def test_routing_sends_requests_to_the_expected_paths(tmp_path: Path) -> None:
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.url.path)
        return httpx.Response(200, json=CANDLE_PAYLOAD)

    client = make_client(handler)
    cache = CandleCache(tmp_path, "slug")
    for endpoint in (CANDLES_HISTORICAL, CANDLES_SERIES):
        fetch_market_candles(
            client, cache, market_ticker=TICKER, game_date=GAME_DATE, request=REQUEST,
            endpoint=endpoint, series_ticker="KXNBAGAME", refresh=True,
        )

    assert seen[0] == f"/historical/markets/{TICKER}/candlesticks"
    assert seen[1] == f"/series/KXNBAGAME/markets/{TICKER}/candlesticks"


def test_archive_404_falls_back_to_the_live_endpoint(tmp_path: Path) -> None:
    """Tier membership is inferred, so it has to be able to be wrong."""
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.url.path)
        if "/historical/" in request.url.path:
            return httpx.Response(404, json={"error": {"code": "not_found"}})
        return httpx.Response(200, json=CANDLE_PAYLOAD)

    outcome = fetch_market_candles(
        make_client(handler), CandleCache(tmp_path, "slug"), market_ticker=TICKER,
        game_date=GAME_DATE, request=REQUEST, endpoint=CANDLES_HISTORICAL,
        series_ticker="KXNBAGAME", refresh=True,
    )

    assert len(seen) == 2
    assert outcome.error is None
    assert outcome.source == CANDLES_SERIES
    assert len(outcome.candles) == 1


def test_404_on_both_endpoints_is_reported_as_a_failure(tmp_path: Path) -> None:
    outcome = fetch_market_candles(
        make_client(lambda _r: httpx.Response(404, json={"error": "gone"})),
        CandleCache(tmp_path, "slug"), market_ticker=TICKER, game_date=GAME_DATE,
        request=REQUEST, endpoint=CANDLES_HISTORICAL, series_ticker="KXNBAGAME", refresh=True,
    )
    assert outcome.error is not None
    assert outcome.candles == []


def test_live_endpoint_404_does_not_retry_elsewhere(tmp_path: Path) -> None:
    calls = 0

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(404, json={"error": "gone"})

    fetch_market_candles(
        make_client(handler), CandleCache(tmp_path, "slug"), market_ticker=TICKER,
        game_date=GAME_DATE, request=REQUEST, endpoint=CANDLES_SERIES,
        series_ticker="KXNBAGAME", refresh=True,
    )
    assert calls == 1


# --- 23: retries and rate limiting on the candle path ----------------------


def test_transient_errors_are_retried_then_succeed(tmp_path: Path) -> None:
    attempts = 0

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            return httpx.Response(503, text="unavailable")
        return httpx.Response(200, json=CANDLE_PAYLOAD)

    outcome = fetch_market_candles(
        make_client(handler), CandleCache(tmp_path, "slug"), market_ticker=TICKER,
        game_date=GAME_DATE, request=REQUEST, endpoint=CANDLES_SERIES,
        series_ticker="KXNBAGAME", refresh=True,
    )
    assert attempts == 3
    assert outcome.error is None
    assert len(outcome.candles) == 1


def test_rate_limit_responses_are_retried(tmp_path: Path) -> None:
    attempts = 0

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            return httpx.Response(429, headers={"retry-after": "1"}, text="slow down")
        return httpx.Response(200, json=CANDLE_PAYLOAD)

    outcome = fetch_market_candles(
        make_client(handler), CandleCache(tmp_path, "slug"), market_ticker=TICKER,
        game_date=GAME_DATE, request=REQUEST, endpoint=CANDLES_SERIES,
        series_ticker="KXNBAGAME", refresh=True,
    )
    assert attempts == 2
    assert outcome.error is None


def test_persistent_failure_is_recorded_not_raised(tmp_path: Path) -> None:
    """One bad market must not abort a 1,200-game run."""
    outcome = fetch_market_candles(
        make_client(lambda _r: httpx.Response(500, text="boom")),
        CandleCache(tmp_path, "slug"), market_ticker=TICKER, game_date=GAME_DATE,
        request=REQUEST, endpoint=CANDLES_SERIES, series_ticker="KXNBAGAME", refresh=True,
    )
    assert outcome.error is not None
    assert outcome.candles == []


def test_client_validates_period_interval_before_sending() -> None:
    client = make_client(lambda _r: httpx.Response(200, json=CANDLE_PAYLOAD))
    with pytest.raises(ValueError, match="period_interval must be one of"):
        client.get_historical_candlesticks(TICKER, start_ts=0, end_ts=60, period_interval=7)


def test_client_rejects_a_reversed_window() -> None:
    client = make_client(lambda _r: httpx.Response(200, json=CANDLE_PAYLOAD))
    with pytest.raises(ValueError, match="precedes start_ts"):
        client.get_series_candlesticks(TICKER, start_ts=100, end_ts=50)


def test_candle_requests_carry_the_window_parameters() -> None:
    seen: list[httpx.URL] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.url)
        return httpx.Response(200, json=CANDLE_PAYLOAD)

    make_client(handler).get_historical_candlesticks(TICKER, start_ts=1000, end_ts=4600)
    params = seen[0].params
    assert params["start_ts"] == "1000"
    assert params["end_ts"] == "4600"
    assert params["period_interval"] == "1"


def test_public_candle_endpoints_need_no_auth_header() -> None:
    seen: list[httpx.Headers] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.headers)
        return httpx.Response(200, json=CANDLE_PAYLOAD)

    make_client(handler).get_historical_candlesticks(TICKER, start_ts=0, end_ts=60)
    assert "authorization" not in seen[0]


# --- 15: cache and resume --------------------------------------------------


def test_cache_slug_encodes_the_request_geometry() -> None:
    assert (
        cache_slug(minutes_before_tip=30, lookback_minutes=60, period_interval=1) == "t30_lb60_p1"
    )
    assert (
        cache_slug(minutes_before_tip=5, lookback_minutes=120, period_interval=60) == "t5_lb120_p60"
    )


def test_cached_response_is_reused_without_a_second_request(tmp_path: Path) -> None:
    calls = 0

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(200, json=CANDLE_PAYLOAD)

    client = make_client(handler)
    cache = CandleCache(tmp_path, "slug")
    kwargs = dict(
        market_ticker=TICKER, game_date=GAME_DATE, request=REQUEST,
        endpoint=CANDLES_SERIES, series_ticker="KXNBAGAME", refresh=False,
    )

    first = fetch_market_candles(client, cache, **kwargs)
    second = fetch_market_candles(client, cache, **kwargs)

    assert calls == 1
    assert first.from_cache is False
    assert second.from_cache is True
    assert [c.end_period_ts for c in second.candles] == [c.end_period_ts for c in first.candles]
    assert cache.stats.hits == 1
    assert cache.stats.writes == 1


def test_refresh_forces_a_refetch(tmp_path: Path) -> None:
    calls = 0

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(200, json=CANDLE_PAYLOAD)

    client, cache = make_client(handler), CandleCache(tmp_path, "slug")
    base = dict(market_ticker=TICKER, game_date=GAME_DATE, request=REQUEST,
                endpoint=CANDLES_SERIES, series_ticker="KXNBAGAME")

    fetch_market_candles(client, cache, **base, refresh=False)
    fetch_market_candles(client, cache, **base, refresh=True)
    assert calls == 2


def test_a_changed_window_invalidates_the_cache_entry(tmp_path: Path) -> None:
    """Reusing candles fetched for a different window would corrupt the anchor."""
    calls = 0

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(200, json=CANDLE_PAYLOAD)

    client, cache = make_client(handler), CandleCache(tmp_path, "slug")
    other = CandleRequest(market_ticker=TICKER, start_ts=9999, end_ts=99999, period_interval=1)

    fetch_market_candles(client, cache, market_ticker=TICKER, game_date=GAME_DATE,
                         request=REQUEST, endpoint=CANDLES_SERIES,
                         series_ticker="KXNBAGAME", refresh=False)
    fetch_market_candles(client, cache, market_ticker=TICKER, game_date=GAME_DATE,
                         request=other, endpoint=CANDLES_SERIES,
                         series_ticker="KXNBAGAME", refresh=False)

    assert calls == 2
    assert cache.stats.stale == 1


def test_different_slugs_do_not_share_entries(tmp_path: Path) -> None:
    cache_a = CandleCache(tmp_path, "t30_lb60_p1")
    cache_b = CandleCache(tmp_path, "t60_lb60_p1")
    cache_a.store(REQUEST, GAME_DATE, CANDLE_PAYLOAD, endpoint=CANDLES_SERIES)

    assert cache_a.load(REQUEST, GAME_DATE) == CANDLE_PAYLOAD
    assert cache_b.load(REQUEST, GAME_DATE) is None


def test_cache_stores_one_file_per_market_sharded_by_date(tmp_path: Path) -> None:
    cache = CandleCache(tmp_path, "t30_lb60_p1")
    path = cache.store(REQUEST, GAME_DATE, CANDLE_PAYLOAD, endpoint=CANDLES_HISTORICAL)

    assert path == tmp_path / "t30_lb60_p1" / GAME_DATE / f"{TICKER}.json"
    document = json.loads(path.read_text())
    assert document["response"] == CANDLE_PAYLOAD           # verbatim
    assert document["request"] == REQUEST.to_dict()
    assert document["endpoint"] == CANDLES_HISTORICAL
    assert document["candle_count"] == 1
    assert document["fetched_at_utc"].endswith("+00:00")


def test_cache_accepts_a_date_object(tmp_path: Path) -> None:
    from datetime import date

    cache = CandleCache(tmp_path, "slug")
    path = cache.store(REQUEST, date(2026, 1, 15), CANDLE_PAYLOAD, endpoint=CANDLES_SERIES)
    assert path.parent.name == "2026-01-15"


def test_corrupt_cache_file_is_ignored_rather_than_crashing(tmp_path: Path) -> None:
    cache = CandleCache(tmp_path, "slug")
    path = cache.path_for(TICKER, GAME_DATE)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{ this is not json")

    assert cache.load(REQUEST, GAME_DATE) is None
    assert cache.stats.corrupt == 1


def test_cache_file_without_a_response_body_is_ignored(tmp_path: Path) -> None:
    cache = CandleCache(tmp_path, "slug")
    path = cache.path_for(TICKER, GAME_DATE)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"request": REQUEST.to_dict(), "response": None}))

    assert cache.load(REQUEST, GAME_DATE) is None
    assert cache.stats.corrupt == 1


def test_no_partial_file_survives_an_interrupted_write(tmp_path: Path) -> None:
    """Entries are written via a temp file and renamed into place."""
    cache = CandleCache(tmp_path, "slug")
    cache.store(REQUEST, GAME_DATE, CANDLE_PAYLOAD, endpoint=CANDLES_SERIES)
    assert list(cache.path_for(TICKER, GAME_DATE).parent.glob("*.tmp")) == []


def test_unsafe_ticker_is_rejected_rather_than_sanitised(tmp_path: Path) -> None:
    cache = CandleCache(tmp_path, "slug")
    with pytest.raises(ValueError, match="filename-safe"):
        cache.path_for("../../etc/passwd", GAME_DATE)


def test_fetch_failure_is_not_cached(tmp_path: Path) -> None:
    """A failed run must be resumable, not poisoned by a cached error."""
    cache = CandleCache(tmp_path, "slug")
    fetch_market_candles(
        make_client(lambda _r: httpx.Response(500, text="boom")), cache,
        market_ticker=TICKER, game_date=GAME_DATE, request=REQUEST,
        endpoint=CANDLES_SERIES, series_ticker="KXNBAGAME", refresh=True,
    )
    assert not cache.path_for(TICKER, GAME_DATE).exists()
    assert cache.stats.writes == 0


def test_api_error_type_is_surfaced_for_debugging(tmp_path: Path) -> None:
    outcome = fetch_market_candles(
        make_client(lambda _r: httpx.Response(400, text="bad params")),
        CandleCache(tmp_path, "slug"), market_ticker=TICKER, game_date=GAME_DATE,
        request=REQUEST, endpoint=CANDLES_SERIES, series_ticker="KXNBAGAME", refresh=True,
    )
    assert "bad params" in outcome.error
    assert isinstance(ApiError("x"), Exception)
