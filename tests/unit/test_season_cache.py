"""Season cache: resume behaviour and request-keyed validity. No network."""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from nba_prediction_market.clients.balldontlie import BallDontLieClient
from nba_prediction_market.config import ConfigError
from nba_prediction_market.ingestion.season_cache import SeasonCache, SeasonRequest
from nba_prediction_market.pipelines.build_history import fetch_season

REQUEST = SeasonRequest(season=2011, per_page=100)
PAGES = [
    {"data": [{"id": 1}, {"id": 2}], "meta": {"next_cursor": 2}},
    {"data": [{"id": 3}], "meta": {"next_cursor": None}},
]


def make_client(handler) -> BallDontLieClient:
    return BallDontLieClient(
        "k", base_url="https://api.test", min_interval=0.0,
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )


def paging_handler(counter: list[int]):
    pages = {
        None: {"data": [{"id": 1}, {"id": 2}], "meta": {"next_cursor": 2}},
        "2": {"data": [{"id": 3}], "meta": {"next_cursor": None}},
    }

    def handler(request: httpx.Request) -> httpx.Response:
        counter.append(1)
        return httpx.Response(200, json=pages[request.url.params.get("cursor")])

    return handler


def test_store_and_load_round_trips(tmp_path: Path) -> None:
    cache = SeasonCache(tmp_path)
    path = cache.store(REQUEST, PAGES)

    assert path == tmp_path / "season_2011.json"
    assert cache.load(REQUEST) == PAGES
    document = json.loads(path.read_text())
    assert document["record_count"] == 3
    assert document["page_count"] == 2
    assert document["complete"] is True
    assert document["request"] == REQUEST.to_dict()


def test_a_missing_season_is_a_miss(tmp_path: Path) -> None:
    cache = SeasonCache(tmp_path)
    assert cache.load(REQUEST) is None
    assert cache.stats.misses == 1


def test_a_changed_request_invalidates_the_entry(tmp_path: Path) -> None:
    cache = SeasonCache(tmp_path)
    cache.store(REQUEST, PAGES)
    assert cache.load(SeasonRequest(season=2011, per_page=25)) is None
    assert cache.stats.stale == 1


def test_an_incomplete_entry_is_never_reused(tmp_path: Path) -> None:
    """An interrupted write must not look like a finished season."""
    cache = SeasonCache(tmp_path)
    path = cache.path_for(2011)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"request": REQUEST.to_dict(), "pages": PAGES, "complete": False}))

    assert cache.load(REQUEST) is None
    assert cache.stats.corrupt == 1


def test_a_corrupt_file_is_ignored(tmp_path: Path) -> None:
    cache = SeasonCache(tmp_path)
    path = cache.path_for(2011)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{ not json")

    assert cache.load(REQUEST) is None
    assert cache.stats.corrupt == 1


def test_writes_are_atomic_leaving_no_temp_file(tmp_path: Path) -> None:
    cache = SeasonCache(tmp_path)
    cache.store(REQUEST, PAGES)
    assert list(tmp_path.glob("*.tmp")) == []


def test_cached_seasons_lists_what_is_present(tmp_path: Path) -> None:
    cache = SeasonCache(tmp_path)
    for season in (2011, 2006, 2025):
        cache.store(SeasonRequest(season=season, per_page=100), PAGES)
    assert cache.cached_seasons() == [2006, 2011, 2025]


# --- resume behaviour through fetch_season ---------------------------------


def test_a_cached_season_is_not_refetched(tmp_path: Path) -> None:
    calls: list[int] = []
    client = make_client(paging_handler(calls))
    cache = SeasonCache(tmp_path)

    first = fetch_season(client, cache, 2011)
    assert len(calls) == 2
    second = fetch_season(client, cache, 2011)

    assert len(calls) == 2, "second call must be served from cache"
    assert first == second
    assert [g["id"] for g in second] == [1, 2, 3]
    assert cache.stats.hits == 1


def test_refresh_forces_a_refetch(tmp_path: Path) -> None:
    calls: list[int] = []
    client = make_client(paging_handler(calls))
    cache = SeasonCache(tmp_path)

    fetch_season(client, cache, 2011)
    fetch_season(client, cache, 2011, refresh=True)
    assert len(calls) == 4


def test_an_empty_season_is_never_cached(tmp_path: Path) -> None:
    """Caching an empty season would permanently poison the dataset."""
    cache = SeasonCache(tmp_path)
    client = make_client(
        lambda _r: httpx.Response(200, json={"data": [], "meta": {"next_cursor": None}})
    )
    with pytest.raises(ConfigError, match="refusing to cache an empty season"):
        fetch_season(client, cache, 2011)
    assert not cache.path_for(2011).exists()


def test_transient_errors_are_retried_during_a_season_fetch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("time.sleep", lambda _s: None)
    attempts: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        attempts.append(1)
        if len(attempts) < 3:
            return httpx.Response(503, text="unavailable")
        return httpx.Response(200, json={"data": [{"id": 1}], "meta": {"next_cursor": None}})

    games = fetch_season(make_client(handler), SeasonCache(tmp_path), 2011)
    assert len(attempts) == 3
    assert [g["id"] for g in games] == [1]


def test_rate_limit_is_retried_during_a_season_fetch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("time.sleep", lambda _s: None)
    attempts: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        attempts.append(1)
        if len(attempts) == 1:
            return httpx.Response(429, headers={"retry-after": "1"}, text="slow down")
        return httpx.Response(200, json={"data": [{"id": 1}], "meta": {"next_cursor": None}})

    fetch_season(make_client(handler), SeasonCache(tmp_path), 2011)
    assert len(attempts) == 2
