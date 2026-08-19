"""HTTP plumbing: pagination, retries, rate limiting, and error mapping.

All traffic is served by ``httpx.MockTransport`` -- nothing here touches the
network.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import httpx
import pytest

from nba_prediction_market.clients.balldontlie import BallDontLieClient
from nba_prediction_market.clients.base import (
    ApiError,
    BaseApiClient,
    RateLimitedError,
    RateLimiter,
    TransientApiError,
    _wait_strategy,
)
from nba_prediction_market.clients.kalshi import KalshiClient

BASE = "https://api.test"


@pytest.fixture(autouse=True)
def _no_real_sleeping(monkeypatch: pytest.MonkeyPatch) -> None:
    """Retries back off with time.sleep; make that instantaneous."""
    monkeypatch.setattr("time.sleep", lambda _seconds: None)


def make_client(handler: Callable[[httpx.Request], httpx.Response], **kwargs: Any) -> BaseApiClient:
    transport = httpx.MockTransport(handler)
    return BaseApiClient(BASE, client=httpx.Client(transport=transport), **kwargs)


def json_response(payload: dict[str, Any], status: int = 200) -> httpx.Response:
    return httpx.Response(status, json=payload)


# --- pagination ------------------------------------------------------------


def test_pagination_follows_cursors_to_the_end() -> None:
    """Three pages are stitched together and the cursor is echoed back each time."""
    pages = {
        None: {"data": [1, 2], "meta": {"next_cursor": 100}},
        "100": {"data": [3, 4], "meta": {"next_cursor": 200}},
        "200": {"data": [5], "meta": {"next_cursor": None}},
    }
    seen_cursors: list[str | None] = []

    def handler(request: httpx.Request) -> httpx.Response:
        cursor = request.url.params.get("cursor")
        seen_cursors.append(cursor)
        return json_response(pages[cursor])

    client = make_client(handler)
    collected = list(
        client.paginate(
            "/games",
            params={"per_page": 100},
            items_key="data",
            next_cursor=lambda p: p["meta"]["next_cursor"],
        )
    )

    assert collected == [[1, 2], [3, 4], [5]]
    assert seen_cursors == [None, "100", "200"]


def test_pagination_stops_on_a_repeated_cursor() -> None:
    """A server that keeps handing back the same cursor must not loop forever."""
    calls = 0

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return json_response({"markets": [calls], "cursor": "same"})

    client = make_client(handler)
    collected = list(
        client.paginate(
            "/markets", params={}, items_key="markets", next_cursor=lambda p: p.get("cursor")
        )
    )

    assert collected == [[1], [2]]
    assert calls == 2


def test_pagination_stops_on_an_empty_page() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.params.get("cursor") is None:
            return json_response({"markets": [1], "cursor": "next"})
        return json_response({"markets": [], "cursor": "further"})

    client = make_client(handler)
    assert list(
        client.paginate(
            "/markets", params={}, items_key="markets", next_cursor=lambda p: p.get("cursor")
        )
    ) == [[1]]


def test_pagination_raises_rather_than_silently_truncating() -> None:
    """Hitting the page ceiling must fail loudly -- a partial season is not a season."""
    counter = 0

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal counter
        counter += 1
        return json_response({"data": [counter], "meta": {"next_cursor": counter}})

    client = make_client(handler)
    with pytest.raises(ApiError, match="exceeded max_pages"):
        list(
            client.paginate(
                "/games",
                params={},
                items_key="data",
                next_cursor=lambda p: p["meta"]["next_cursor"],
                max_pages=3,
            )
        )


def test_pagination_rejects_a_missing_items_key() -> None:
    client = make_client(lambda _r: json_response({"meta": {}}))
    with pytest.raises(ApiError, match="has no 'data' key"):
        list(client.paginate("/games", params={}, items_key="data", next_cursor=lambda _p: None))


def test_pagination_rejects_a_non_list_items_value() -> None:
    client = make_client(lambda _r: json_response({"data": {"nope": 1}}))
    with pytest.raises(ApiError, match="to be a list"):
        list(client.paginate("/games", params={}, items_key="data", next_cursor=lambda _p: None))


def test_on_page_callback_sees_every_raw_payload() -> None:
    """Raw payloads must reach the caller so they can be persisted verbatim."""
    pages = {
        None: {"data": [1], "meta": {"next_cursor": 7}},
        "7": {"data": [2], "meta": {"next_cursor": None}},
    }
    captured: list[tuple[int, dict[str, Any]]] = []
    client = make_client(lambda r: json_response(pages[r.url.params.get("cursor")]))

    list(
        client.paginate(
            "/games",
            params={},
            items_key="data",
            next_cursor=lambda p: p["meta"]["next_cursor"],
            on_page=lambda n, payload: captured.append((n, payload)),
        )
    )

    assert [n for n, _ in captured] == [1, 2]
    assert captured[0][1] == pages[None]


# --- retries and error mapping --------------------------------------------


def test_retries_transient_server_errors_then_succeeds() -> None:
    attempts = 0

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            return httpx.Response(503, text="unavailable")
        return json_response({"ok": True})

    client = make_client(handler)
    assert client.get_json("/x") == {"ok": True}
    assert attempts == 3


def test_gives_up_after_max_retries() -> None:
    attempts = 0

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        return httpx.Response(500, text="boom")

    client = make_client(handler, max_retries=3)
    with pytest.raises(TransientApiError, match="Server error 500"):
        client.get_json("/x")
    assert attempts == 3


def test_rate_limit_is_retried_and_carries_retry_after() -> None:
    attempts = 0

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            return httpx.Response(429, headers={"retry-after": "2"}, text="slow down")
        return json_response({"ok": True})

    client = make_client(handler)
    assert client.get_json("/x") == {"ok": True}
    assert attempts == 2


def test_client_errors_are_not_retried() -> None:
    attempts = 0

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        return httpx.Response(404, text="no such series")

    client = make_client(handler)
    with pytest.raises(ApiError) as excinfo:
        client.get_json("/x")
    assert excinfo.value.status_code == 404
    assert "no such series" in str(excinfo.value)
    assert attempts == 1


def test_network_errors_become_transient_errors() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    client = make_client(handler, max_retries=2)
    with pytest.raises(TransientApiError, match="Network error"):
        client.get_json("/x")


def test_timeouts_become_transient_errors() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("too slow")

    client = make_client(handler, max_retries=2)
    with pytest.raises(TransientApiError, match="Timeout"):
        client.get_json("/x")


def test_non_json_body_is_an_error() -> None:
    client = make_client(lambda _r: httpx.Response(200, text="<html>nope</html>"))
    with pytest.raises(ApiError, match="Non-JSON response"):
        client.get_json("/x")


def test_json_array_body_is_an_error() -> None:
    client = make_client(lambda _r: httpx.Response(200, json=[1, 2, 3]))
    with pytest.raises(ApiError, match="Expected a JSON object"):
        client.get_json("/x")


def test_wait_strategy_prefers_the_servers_retry_after() -> None:
    class _Outcome:
        def __init__(self, exc: Exception) -> None:
            self._exc = exc

        def exception(self) -> Exception:
            return self._exc

    class _State:
        def __init__(self, exc: Exception, attempt: int) -> None:
            self.outcome = _Outcome(exc)
            self.attempt_number = attempt

    assert _wait_strategy(_State(RateLimitedError("x", retry_after=5.0), 1)) == 6.0
    # No Retry-After -> exponential backoff.
    assert _wait_strategy(_State(TransientApiError("x"), 3)) == 4.0
    # An absurd Retry-After is capped rather than stalling the run.
    assert _wait_strategy(_State(RateLimitedError("x", retry_after=9999.0), 1)) == 120.0


# --- rate limiter ----------------------------------------------------------


def test_rate_limiter_spaces_calls_by_the_minimum_interval() -> None:
    now = [0.0]
    slept: list[float] = []

    limiter = RateLimiter(
        12.5,
        sleep=lambda s: (slept.append(s), now.__setitem__(0, now[0] + s)),
        monotonic=lambda: now[0],
    )

    limiter.acquire()          # first call is free
    now[0] += 2.0              # only 2s of real work elapsed
    limiter.acquire()          # must wait the remaining 10.5s
    now[0] += 30.0             # plenty of time passed
    limiter.acquire()          # no wait needed

    assert slept == [10.5]


def test_rate_limiter_disabled_at_zero() -> None:
    slept: list[float] = []
    limiter = RateLimiter(0.0, sleep=slept.append, monotonic=lambda: 0.0)
    for _ in range(5):
        limiter.acquire()
    assert slept == []


# --- BALLDONTLIE client ----------------------------------------------------


def test_balldontlie_sends_the_api_key_and_season_params() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return json_response({"data": [{"id": 1}], "meta": {"next_cursor": None}})

    client = BallDontLieClient(
        "secret-key", base_url=BASE, min_interval=0.0, client=httpx.Client(
            transport=httpx.MockTransport(handler)
        )
    )
    games = client.fetch_games(2025)

    assert games == [{"id": 1}]
    request = seen[0]
    # Bare key, no "Bearer " prefix -- that is what the service accepts.
    assert request.headers["authorization"] == "secret-key"
    assert request.url.params["seasons[]"] == "2025"
    assert request.url.params["per_page"] == "100"


def test_balldontlie_requires_an_api_key() -> None:
    with pytest.raises(ValueError, match="api_key is required"):
        BallDontLieClient("")


@pytest.mark.parametrize("per_page", [0, 101, -1])
def test_balldontlie_rejects_out_of_range_page_sizes(per_page: int) -> None:
    client = BallDontLieClient("k", base_url=BASE, min_interval=0.0)
    with pytest.raises(ValueError, match=r"per_page must be within 1\.\.100"):
        list(client.iter_games(2025, per_page=per_page))


def test_balldontlie_paginates_across_pages() -> None:
    pages = {
        None: {"data": [{"id": 1}, {"id": 2}], "meta": {"next_cursor": 2}},
        "2": {"data": [{"id": 3}], "meta": {"next_cursor": None}},
    }
    client = BallDontLieClient(
        "k", base_url=BASE, min_interval=0.0,
        client=httpx.Client(
            transport=httpx.MockTransport(
                lambda r: json_response(pages[r.url.params.get("cursor")])
            )
        ),
    )
    assert [g["id"] for g in client.fetch_games(2025)] == [1, 2, 3]


# --- Kalshi client ---------------------------------------------------------


def _kalshi_handler(
    historical: list[dict[str, Any]], live: list[dict[str, Any]]
) -> Callable[[httpx.Request], httpx.Response]:
    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path.endswith("/historical/markets"):
            return json_response({"markets": historical, "cursor": None})
        if path.endswith("/markets"):
            return json_response({"markets": live, "cursor": None})
        if path.endswith("/historical/cutoff"):
            return json_response({"market_settled_ts": "2026-06-20T00:00:00Z"})
        raise AssertionError(f"unexpected path {path}")

    return handler


def kalshi_client(handler: Callable[[httpx.Request], httpx.Response]) -> KalshiClient:
    return KalshiClient(
        base_url=BASE, min_interval=0.0, client=httpx.Client(transport=httpx.MockTransport(handler))
    )


def test_kalshi_merges_both_stores_and_deduplicates_by_ticker() -> None:
    historical = [{"ticker": "A", "store": "hist"}, {"ticker": "B", "store": "hist"}]
    live = [{"ticker": "B", "store": "live"}, {"ticker": "C", "store": "live"}]

    client = kalshi_client(_kalshi_handler(historical, live))
    markets, sources = client.fetch_markets_from_both_stores()

    by_ticker = {m["ticker"]: m for m in markets}
    assert sorted(by_ticker) == ["A", "B", "C"]
    # The archival copy wins for a ticker present in both stores.
    assert by_ticker["B"]["store"] == "hist"
    assert sources["A"] == ["historical"]
    assert sources["B"] == ["historical", "markets"]
    assert sources["C"] == ["markets"]


def test_kalshi_drops_markets_without_a_ticker_and_records_nothing_for_them() -> None:
    markets, sources = kalshi_client(
        _kalshi_handler([{"ticker": "A"}, {"no_ticker": True}], [])
    ).fetch_markets_from_both_stores()

    assert [m["ticker"] for m in markets] == ["A"]
    assert sorted(sources) == ["A"]


def test_kalshi_dedup_is_idempotent_when_both_stores_are_identical() -> None:
    same = [{"ticker": "A"}, {"ticker": "B"}]
    markets, sources = kalshi_client(_kalshi_handler(same, same)).fetch_markets_from_both_stores()

    assert len(markets) == 2
    assert all(s == ["historical", "markets"] for s in sources.values())


def test_kalshi_historical_cutoff_is_returned_verbatim() -> None:
    cutoff = kalshi_client(_kalshi_handler([], [])).get_historical_cutoff()
    assert cutoff == {"market_settled_ts": "2026-06-20T00:00:00Z"}


def test_kalshi_markets_omits_the_status_filter_by_default() -> None:
    """``status=all`` is rejected by the live API with HTTP 400, so we send none."""
    seen: list[httpx.URL] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.url)
        return json_response({"markets": [], "cursor": None})

    list(kalshi_client(handler).iter_markets("KXNBAGAME"))
    assert "status" not in seen[0].params
    assert seen[0].params["series_ticker"] == "KXNBAGAME"
