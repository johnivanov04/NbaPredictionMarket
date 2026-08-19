"""Live-API smoke checks.

These are the only tests that touch the network, and they are deselected by
default (``addopts = -m 'not integration'``). Run them deliberately when you
want to confirm the upstream contracts still hold::

    pytest -m integration

They assert the *shape* of the responses -- the assumptions the ingestion code
is built on -- rather than any particular data, so they stay stable while
verifying that a schema change upstream would be caught.
"""

from __future__ import annotations

import os

import pytest

from nba_prediction_market.clients.balldontlie import BallDontLieClient
from nba_prediction_market.clients.kalshi import KalshiClient
from nba_prediction_market.config import KALSHI_NBA_SERIES_TICKER, load_settings
from nba_prediction_market.ingestion.kalshi_markets import normalize_market, parse_event_ticker
from nba_prediction_market.ingestion.nba_games import normalize_game
from nba_prediction_market.matching.team_names import resolve_team

pytestmark = pytest.mark.integration

SEASON = 2025


@pytest.fixture(scope="module")
def api_key() -> str:
    load_settings(load_env=True)
    key = os.environ.get("BALLDONTLIE_API_KEY", "").strip()
    if not key:
        pytest.skip("BALLDONTLIE_API_KEY is not set")
    return key


def test_balldontlie_games_have_the_fields_we_normalize(api_key: str) -> None:
    with BallDontLieClient(api_key, min_interval=0.0) as client:
        page = next(iter(client.iter_games(SEASON, per_page=5)))

    assert page, "expected at least one game for the season"
    game = page[0]
    for field in ("id", "date", "season", "status", "postseason", "home_team", "visitor_team"):
        assert field in game, f"BALLDONTLIE dropped the {field!r} field"
    for side in ("home_team", "visitor_team"):
        assert {"id", "abbreviation", "full_name"} <= set(game[side])
        assert resolve_team(game[side]["abbreviation"]).ok

    row = normalize_game(game)
    assert row["season"] == SEASON
    assert row["home_team_code"] and row["visitor_team_code"]


def test_balldontlie_season_2025_really_is_2025_26(api_key: str) -> None:
    """Guard the season-numbering assumption against an upstream change."""
    with BallDontLieClient(api_key, min_interval=0.0) as client:
        page = next(iter(client.iter_games(SEASON, per_page=5)))

    dates = sorted(normalize_game(g)["game_date"] for g in page)
    assert dates[0].year == SEASON
    assert dates[0].month >= 9, "the season should open in the autumn of its first year"


def test_balldontlie_pagination_advances(api_key: str) -> None:
    with BallDontLieClient(api_key, min_interval=0.0) as client:
        pages = []
        for page in client.iter_games(SEASON, per_page=5):
            pages.append(page)
            if len(pages) == 2:
                break

    assert len(pages) == 2
    first_ids = {g["id"] for g in pages[0]}
    assert first_ids and not (first_ids & {g["id"] for g in pages[1]})


def test_kalshi_historical_cutoff_shape() -> None:
    with KalshiClient(min_interval=0.0) as client:
        cutoff = client.get_historical_cutoff()

    assert isinstance(cutoff, dict) and cutoff
    assert any(key.endswith("_ts") for key in cutoff)


def test_kalshi_market_metadata_matches_our_schema() -> None:
    with KalshiClient(min_interval=0.0) as client:
        page = next(iter(client.iter_historical_markets(KALSHI_NBA_SERIES_TICKER, limit=5)))

    assert page
    market = page[0]
    for field in ("ticker", "event_ticker", "title", "yes_sub_title", "open_time", "close_time"):
        assert field in market, f"Kalshi dropped the {field!r} field"

    parts = parse_event_ticker(market["event_ticker"])
    assert parts, f"event ticker {market['event_ticker']!r} no longer parses"

    row = normalize_market(market)
    assert row["home_team_code"] and row["away_team_code"]
    assert row["scheduled_game_date"] is not None


def test_kalshi_no_sub_title_still_mirrors_yes_sub_title() -> None:
    """The ingestion code relies on no_sub_title NOT naming the opponent."""
    with KalshiClient(min_interval=0.0) as client:
        page = next(iter(client.iter_historical_markets(KALSHI_NBA_SERIES_TICKER, limit=20)))

    mirrored = [m for m in page if m.get("no_sub_title") == m.get("yes_sub_title")]
    assert len(mirrored) == len(page), "no_sub_title now differs -- revisit kalshi_markets.py"


def test_kalshi_events_carry_the_abbreviated_sub_title_we_key_on() -> None:
    with KalshiClient(min_interval=0.0) as client:
        page = next(iter(client.iter_events(KALSHI_NBA_SERIES_TICKER, limit=10)))

    assert page
    for event in page:
        assert "sub_title" in event
        assert " at " in event["sub_title"], f"unexpected sub_title {event['sub_title']!r}"


def test_kalshi_status_all_is_still_rejected() -> None:
    """Documents why the client omits the status filter entirely."""
    from nba_prediction_market.clients.base import ApiError

    with KalshiClient(min_interval=0.0) as client, pytest.raises(ApiError) as excinfo:
        list(client.iter_markets(KALSHI_NBA_SERIES_TICKER, limit=1, status="all"))

    assert excinfo.value.status_code == 400
