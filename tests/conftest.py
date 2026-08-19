"""Shared fixtures.

Every payload here is a trimmed copy of a real response captured from the live
APIs on 2026-08-19, so the tests exercise the field names and value types the
code will actually meet. No test in ``tests/unit`` performs network I/O.
"""

from __future__ import annotations

from typing import Any

import pytest

# --- BALLDONTLIE -----------------------------------------------------------

BDL_GAME_OKC_HOU: dict[str, Any] = {
    "id": 18446819,
    "date": "2025-10-21",
    "season": 2025,
    "status": "Final",
    "status_state": "final",
    "period": 6,
    "time": "Final",
    "postseason": False,
    "postponed": False,
    "home_team_score": 125,
    "visitor_team_score": 124,
    "datetime": "2025-10-21T23:30:00.000Z",
    "home_team": {
        "id": 21,
        "conference": "West",
        "division": "Northwest",
        "city": "Oklahoma City",
        "name": "Thunder",
        "full_name": "Oklahoma City Thunder",
        "abbreviation": "OKC",
    },
    "visitor_team": {
        "id": 11,
        "conference": "West",
        "division": "Southwest",
        "city": "Houston",
        "name": "Rockets",
        "full_name": "Houston Rockets",
        "abbreviation": "HOU",
    },
}

BDL_GAME_LAL_GSW: dict[str, Any] = {
    "id": 18446820,
    "date": "2025-10-21",
    "season": 2025,
    "status": "Final",
    "status_state": "final",
    "period": 4,
    "postseason": False,
    "postponed": False,
    "home_team_score": 109,
    "visitor_team_score": 119,
    "datetime": "2025-10-22T02:00:00.000Z",
    "home_team": {
        "id": 14,
        "city": "Los Angeles",
        "name": "Lakers",
        "full_name": "Los Angeles Lakers",
        "abbreviation": "LAL",
    },
    "visitor_team": {
        "id": 10,
        "city": "Golden State",
        "name": "Warriors",
        "full_name": "Golden State Warriors",
        "abbreviation": "GSW",
    },
}


def bdl_game(**overrides: Any) -> dict[str, Any]:
    """A BALLDONTLIE game with fields overridden."""
    game = {k: (v.copy() if isinstance(v, dict) else v) for k, v in BDL_GAME_OKC_HOU.items()}
    game.update(overrides)
    return game


# --- Kalshi ----------------------------------------------------------------

#: Note ``no_sub_title == yes_sub_title`` -- verified on every live market, so
#: the "no" side does NOT name the opposing team.
KALSHI_MARKET_SAS: dict[str, Any] = {
    "ticker": "KXNBAGAME-26JUN13NYKSAS-SAS",
    "event_ticker": "KXNBAGAME-26JUN13NYKSAS",
    "title": "Game 5: New York at San Antonio Winner?",
    "yes_sub_title": "San Antonio",
    "no_sub_title": "San Antonio",
    "open_time": "2026-06-09T03:55:00Z",
    "close_time": "2026-06-14T03:31:47Z",
    "expected_expiration_time": "2026-06-14T03:30:00Z",
    "expiration_time": "2026-06-28T00:30:00Z",
    "settlement_ts": "2026-06-14T03:32:25.229203Z",
    "occurrence_datetime": "2026-06-14T03:30:00Z",
    "status": "finalized",
    "result": "no",
    "expiration_value": "New York",
    "settlement_value_dollars": "0.0000",
    "last_price_dollars": "0.0100",
    "previous_price_dollars": "0.0100",
    "yes_bid_dollars": "0.0000",
    "yes_ask_dollars": "1.0000",
    "no_bid_dollars": "0.0000",
    "no_ask_dollars": "1.0000",
    "volume_fp": "48467702.82",
    "volume_24h_fp": "0.00",
    "open_interest_fp": "0.00",
    "liquidity_dollars": "0.0000",
    "market_type": "binary",
    "strike_type": "structured",
    "custom_strike": {"basketball_team": "ad36c3e8-4194-4e63-920f-7c50f46191a6"},
    "rules_primary": (
        "If San Antonio wins the Game 5: New York at San Antonio professional "
        "basketball game originally scheduled for Jun 13, 2026, then the market "
        "resolves to Yes."
    ),
}

KALSHI_MARKET_NYK: dict[str, Any] = {
    **KALSHI_MARKET_SAS,
    "ticker": "KXNBAGAME-26JUN13NYKSAS-NYK",
    "yes_sub_title": "New York",
    "no_sub_title": "New York",
    "result": "yes",
    "custom_strike": {"basketball_team": "67468ecc-b868-43a4-b9dc-751a52894bb0"},
    "rules_primary": (
        "If New York wins the Game 5: New York at San Antonio professional "
        "basketball game originally scheduled for Jun 13, 2026, then the market "
        "resolves to Yes."
    ),
}

KALSHI_EVENT_NYKSAS: dict[str, Any] = {
    "event_ticker": "KXNBAGAME-26JUN13NYKSAS",
    "series_ticker": "KXNBAGAME",
    "sub_title": "NYK at SAS (Jun 13)",
    "title": "Game 5: New York at San Antonio",
    "category": "Sports",
    "mutually_exclusive": True,
}


def kalshi_pair(
    *,
    event_ticker: str,
    sub_title: str,
    away: str,
    home: str,
    rules_date: str,
    title: str | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Build the two per-team markets and the event for one matchup."""
    title = title or f"{away} at {home} Winner?"
    markets = []
    for team, result in ((away, "no"), (home, "yes")):
        markets.append(
            {
                **KALSHI_MARKET_SAS,
                "ticker": f"{event_ticker}-{team}",
                "event_ticker": event_ticker,
                "title": title,
                "yes_sub_title": team,
                "no_sub_title": team,
                "result": result,
                "rules_primary": (
                    f"If {team} wins the professional basketball game scheduled "
                    f"for {rules_date}, then the market resolves to Yes."
                ),
            }
        )
    event = {
        "event_ticker": event_ticker,
        "series_ticker": "KXNBAGAME",
        "sub_title": sub_title,
        "title": title,
    }
    return markets, event


@pytest.fixture
def raw_games() -> list[dict[str, Any]]:
    return [BDL_GAME_OKC_HOU, BDL_GAME_LAL_GSW]


@pytest.fixture
def raw_markets() -> list[dict[str, Any]]:
    return [KALSHI_MARKET_SAS, KALSHI_MARKET_NYK]


@pytest.fixture
def raw_events() -> list[dict[str, Any]]:
    return [KALSHI_EVENT_NYKSAS]
