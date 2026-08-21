"""Paid-feed normalization, caching, and pagination. No network."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx
import pandas as pd
import pytest

from nba_prediction_market.clients.balldontlie import MAX_PER_PAGE, BallDontLieClient
from nba_prediction_market.ingestion.paid_cache import PaidCache, PaidRequest
from nba_prediction_market.ingestion.paid_stats import (
    ADVANCED_COLUMNS,
    PLAYER_GAME_COLUMNS,
    build_frame,
    normalize_advanced,
    normalize_player_game,
    parse_minutes,
)


def game_blob(gid: int = 19051, season: int = 2006, home_id: int = 14) -> dict[str, Any]:
    return {"id": gid, "season": season, "date": "2006-11-01", "home_team_id": home_id,
            "visitor_team_id": 11}


def player_record(**over: Any) -> dict[str, Any]:
    base = {
        "id": 1, "game": game_blob(), "min": "34:12",
        "player": {"id": 1472, "first_name": "Luke", "last_name": "Walton"},
        "team": {"id": 14, "abbreviation": "LAL"},
        "pts": 20, "reb": 5, "oreb": 1, "dreb": 4, "ast": 7, "stl": 2, "blk": 1,
        "turnover": 3, "pf": 2, "fgm": 8, "fga": 15, "fg_pct": 0.533,
        "fg3m": 2, "fg3a": 5, "fg3_pct": 0.4, "ftm": 2, "fta": 2, "ft_pct": 1.0,
        "plus_minus": 7,
    }
    base.update(over)
    return base


def advanced_record(**over: Any) -> dict[str, Any]:
    base = {
        "id": 1896, "game": game_blob(),
        "player": {"id": 1472, "first_name": "Luke", "last_name": "Walton"},
        "team": {"id": 14, "abbreviation": "LAL"},
        "pace": 99.12, "offensive_rating": 111.4, "defensive_rating": 81.4,
        "net_rating": 30, "pie": 0.251, "true_shooting_percentage": 0.795,
        "effective_field_goal_percentage": 0.75, "usage_percentage": 0.162,
        "assist_percentage": 0.292, "assist_ratio": 36.8, "assist_to_turnover": 7,
        "turnover_ratio": 5.3, "offensive_rebound_percentage": 0.038,
        "defensive_rebound_percentage": 0.125, "rebound_percentage": 0.086,
    }
    base.update(over)
    return base


# --- minutes ---------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("34:12", 34.2), ("34", 34.0), ("0:00", 0.0), ("00:00", 0.0), ("5:30", 5.5),
     ("48:00", 48.0)],
)
def test_parse_minutes(raw: str, expected: float) -> None:
    assert parse_minutes(raw) == pytest.approx(expected, abs=1e-9)


@pytest.mark.parametrize("raw", [None, "", "  ", "not-minutes", "DNP"])
def test_unparseable_minutes_are_missing_not_zero(raw) -> None:
    """A player who did not appear has unknown minutes, not zero."""
    assert parse_minutes(raw) is None


# --- normalization ---------------------------------------------------------


def test_player_record_normalizes_to_the_documented_schema() -> None:
    row = normalize_player_game(player_record())
    assert set(row) == set(PLAYER_GAME_COLUMNS)
    assert row["nba_game_id"] == 19051
    assert row["season"] == 2006
    assert row["player_id"] == 1472
    assert row["player_name"] == "Luke Walton"
    assert row["team_id"] == 14
    assert row["minutes"] == pytest.approx(34.2)
    assert row["pts"] == 20
    assert row["plus_minus"] == 7


def test_home_away_is_derived_from_the_embedded_game() -> None:
    home = normalize_player_game(player_record())
    away = normalize_player_game(
        player_record(team={"id": 11, "abbreviation": "HOU"})
    )
    assert home["is_home"] is True
    assert away["is_home"] is False


def test_home_away_is_missing_when_it_cannot_be_determined() -> None:
    row = normalize_player_game(
        player_record(game={"id": 1, "season": 2006, "date": "2006-11-01"})
    )
    assert row["is_home"] is None


def test_advanced_record_normalizes_to_the_documented_schema() -> None:
    row = normalize_advanced(advanced_record())
    assert set(row) == set(ADVANCED_COLUMNS)
    assert row["pace"] == pytest.approx(99.12)
    assert row["offensive_rating"] == pytest.approx(111.4)
    assert row["true_shooting_percentage"] == pytest.approx(0.795)


@pytest.mark.parametrize(
    "field", ["fg_pct", "fg3_pct", "ft_pct"]
)
def test_out_of_range_percentages_are_rejected(field: str) -> None:
    row = normalize_player_game(player_record(**{field: 1.5}))
    assert row[field] is None
    row = normalize_player_game(player_record(**{field: -0.2}))
    assert row[field] is None


def test_out_of_range_advanced_fractions_are_rejected() -> None:
    row = normalize_advanced(advanced_record(true_shooting_percentage=2.0))
    assert row["true_shooting_percentage"] is None
    # Unbounded metrics are untouched.
    assert normalize_advanced(advanced_record(pace=300.0))["pace"] == 300.0


def test_missing_fields_become_null_not_zero() -> None:
    row = normalize_player_game(player_record(pts=None, plus_minus=None))
    assert row["pts"] is None
    assert row["plus_minus"] is None


def test_a_record_without_a_game_or_player_is_dropped_and_counted() -> None:
    records = [player_record(), player_record(game={}), player_record(player={})]
    frame, stats = build_frame(records, "player_game")
    assert len(frame) == 1
    assert stats["malformed_dropped"] == 2
    assert stats["records_in"] == 3


def test_duplicate_game_player_rows_are_collapsed_and_counted() -> None:
    frame, stats = build_frame([player_record(), player_record()], "player_game")
    assert len(frame) == 1
    assert stats["duplicate_game_player_rows"] == 1


def test_two_players_in_one_game_are_both_kept() -> None:
    other = player_record(player={"id": 99, "first_name": "A", "last_name": "B"})
    frame, stats = build_frame([player_record(), other], "player_game")
    assert len(frame) == 2
    assert stats["duplicate_game_player_rows"] == 0


def test_frames_are_deterministically_ordered() -> None:
    a = player_record(player={"id": 5, "first_name": "A", "last_name": "B"})
    b = player_record(player={"id": 2, "first_name": "C", "last_name": "D"})
    frame, _ = build_frame([a, b], "player_game")
    assert list(frame["player_id"]) == [2, 5]


def test_an_unknown_feed_kind_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown feed kind"):
        build_frame([], "something_else")


def test_an_empty_feed_yields_an_empty_but_typed_frame() -> None:
    frame, stats = build_frame([], "advanced")
    assert list(frame.columns) == list(ADVANCED_COLUMNS)
    assert len(frame) == 0
    assert stats["rows_out"] == 0


def test_historical_schema_variant_with_nested_home_team_is_handled() -> None:
    """Older payload shapes nest the team object instead of a flat id."""
    record = player_record(
        game={"id": 1, "season": 2006, "date": "2006-11-01",
              "home_team": {"id": 14}, "visitor_team": {"id": 11}}
    )
    assert normalize_player_game(record)["is_home"] is True


# --- cache -----------------------------------------------------------------


REQUEST = PaidRequest(feed="player_game_stats", endpoint="/v1/stats", season=2006, per_page=100)
RECORDS = [player_record(), player_record(player={"id": 2})]


def test_cache_round_trips(tmp_path: Path) -> None:
    cache = PaidCache(tmp_path)
    path = cache.store(REQUEST, RECORDS)
    assert path == tmp_path / "player_game_stats" / "season_2006.json"
    assert cache.load(REQUEST) == RECORDS
    document = json.loads(path.read_text())
    assert document["complete"] is True
    assert document["record_count"] == 2


def test_a_missing_season_is_a_miss(tmp_path: Path) -> None:
    cache = PaidCache(tmp_path)
    assert cache.load(REQUEST) is None
    assert cache.stats.misses == 1


def test_a_changed_request_invalidates_the_entry(tmp_path: Path) -> None:
    cache = PaidCache(tmp_path)
    cache.store(REQUEST, RECORDS)
    other = PaidRequest(feed="player_game_stats", endpoint="/v1/stats", season=2006, per_page=25)
    assert cache.load(other) is None
    assert cache.stats.stale == 1


def test_an_incomplete_entry_is_never_reused(tmp_path: Path) -> None:
    cache = PaidCache(tmp_path)
    path = cache.path_for("player_game_stats", 2006)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"request": REQUEST.to_dict(), "records": RECORDS,
                                "complete": False}))
    assert cache.load(REQUEST) is None
    assert cache.stats.corrupt == 1


def test_a_corrupt_entry_is_ignored(tmp_path: Path) -> None:
    cache = PaidCache(tmp_path)
    path = cache.path_for("player_game_stats", 2006)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{ not json")
    assert cache.load(REQUEST) is None
    assert cache.stats.corrupt == 1


def test_writes_are_atomic(tmp_path: Path) -> None:
    cache = PaidCache(tmp_path)
    cache.store(REQUEST, RECORDS)
    assert list((tmp_path / "player_game_stats").glob("*.tmp")) == []


def test_feeds_are_namespaced_separately(tmp_path: Path) -> None:
    cache = PaidCache(tmp_path)
    advanced = PaidRequest(
        feed="advanced_stats_v1", endpoint="/v1/stats/advanced", season=2006, per_page=100
    )
    cache.store(REQUEST, RECORDS)
    assert cache.load(advanced) is None
    assert cache.cached_seasons("player_game_stats") == [2006]
    assert cache.cached_seasons("advanced_stats_v1") == []


def test_an_unsafe_feed_name_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="filename-safe"):
        PaidCache(tmp_path).path_for("../escape", 2006)


# --- pagination ------------------------------------------------------------


def test_paid_pagination_follows_cursors() -> None:
    pages = {
        None: {"data": [{"id": 1}, {"id": 2}], "meta": {"next_cursor": 2}},
        "2": {"data": [{"id": 3}], "meta": {"next_cursor": None}},
    }
    seen: list[httpx.URL] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.url)
        return httpx.Response(200, json=pages[request.url.params.get("cursor")])

    client = BallDontLieClient(
        "k", base_url="https://api.test", min_interval=0.0,
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    collected = [r for page in client.iter_paid_records("/v1/stats", 2006) for r in page]

    assert [r["id"] for r in collected] == [1, 2, 3]
    assert seen[0].params["seasons[]"] == "2006"
    assert seen[0].params["per_page"] == str(MAX_PER_PAGE)


@pytest.mark.parametrize("per_page", [0, 101, -1])
def test_paid_pagination_rejects_out_of_range_page_sizes(per_page: int) -> None:
    client = BallDontLieClient("k", base_url="https://api.test", min_interval=0.0)
    with pytest.raises(ValueError, match="per_page must be within"):
        list(client.iter_paid_records("/v1/stats", 2006, per_page=per_page))


def test_season_filter_is_sent_for_every_page() -> None:
    seen: list[str] = []
    pages = {
        None: {"data": [{"id": 1}], "meta": {"next_cursor": 7}},
        "7": {"data": [{"id": 2}], "meta": {"next_cursor": None}},
    }

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.url.params["seasons[]"])
        return httpx.Response(200, json=pages[request.url.params.get("cursor")])

    client = BallDontLieClient(
        "k", base_url="https://api.test", min_interval=0.0,
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    list(client.iter_paid_records("/v1/stats", 2011))
    assert seen == ["2011", "2011"]


def test_frames_join_on_the_trusted_game_id() -> None:
    """The feed's game id must be the same identifier Phase 3A0 trusts."""
    frame, _ = build_frame([player_record(game=game_blob(gid=19368))], "player_game")
    trusted = pd.DataFrame({"nba_game_id": [19368, 19369]})
    merged = frame.merge(trusted, on="nba_game_id", how="inner")
    assert len(merged) == 1
