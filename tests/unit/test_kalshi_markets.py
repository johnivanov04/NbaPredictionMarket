"""Normalization of Kalshi KXNBAGAME market metadata.

The parsing rules encoded here mirror behaviour verified against live data; each
test names the assumption it is pinning down.
"""

from __future__ import annotations

from datetime import UTC, date, datetime

import pandas as pd
import pytest
from tests.conftest import (
    KALSHI_EVENT_NYKSAS,
    KALSHI_MARKET_NYK,
    KALSHI_MARKET_SAS,
    kalshi_pair,
)

from nba_prediction_market.ingestion.kalshi_markets import (
    KALSHI_MARKET_COLUMNS,
    build_markets_frame,
    normalize_market,
    parse_event_subtitle,
    parse_event_ticker,
    parse_market_ticker,
    parse_rules_date,
    parse_title_orientation,
)

# --- ticker parsing --------------------------------------------------------


def test_parse_event_ticker_splits_date_and_both_teams() -> None:
    parts = parse_event_ticker("KXNBAGAME-26JUN13NYKSAS")
    assert parts == {
        "series_ticker": "KXNBAGAME",
        "ticker_date": date(2026, 6, 13),
        "away_raw": "NYK",
        "home_raw": "SAS",
    }


def test_event_ticker_team_segment_is_split_evenly_not_greedily() -> None:
    """``NYKSAS`` must split 3/3; a flexible width would produce ``NYKS``+``AS``."""
    parts = parse_event_ticker("KXNBAGAME-26JUN13NYKSAS")
    assert (parts["away_raw"], parts["home_raw"]) == ("NYK", "SAS")


@pytest.mark.parametrize(
    "ticker",
    [
        None,
        "",
        "NOT-A-TICKER",
        "KXNBAGAME-26XXX13NYKSAS",   # bad month
        "KXNBAGAME-26JUN99NYKSAS",   # impossible day
        "KXNBAGAME-26JUN13NYK",      # only one team
        "KXNBAGAME-26JUN13NYKSASX",  # trailing junk
    ],
)
def test_parse_event_ticker_returns_empty_rather_than_guessing(ticker: str | None) -> None:
    assert parse_event_ticker(ticker) == {}


def test_parse_market_ticker_splits_event_and_team() -> None:
    assert parse_market_ticker("KXNBAGAME-26JUN13NYKSAS-SAS") == {
        "event_ticker": "KXNBAGAME-26JUN13NYKSAS",
        "team_raw": "SAS",
    }


@pytest.mark.parametrize("ticker", [None, "", "KXNBAGAME-26JUN13NYKSAS", "garbage"])
def test_parse_market_ticker_returns_empty_on_mismatch(ticker: str | None) -> None:
    assert parse_market_ticker(ticker) == {}


# --- other metadata fields -------------------------------------------------


@pytest.mark.parametrize(
    ("rules", "expected"),
    [
        ("...game scheduled for Oct 21, 2025, then...", date(2025, 10, 21)),
        ("...game originally scheduled for Jun 13, 2026, then...", date(2026, 6, 13)),
        ("...scheduled for Jan 5, 2026, then...", date(2026, 1, 5)),
        ("no date at all", None),
        (None, None),
        ("", None),
    ],
)
def test_parse_rules_date(rules: str | None, expected: date | None) -> None:
    assert parse_rules_date(rules) == expected


@pytest.mark.parametrize(
    ("sub_title", "expected"),
    [
        ("NYK at SAS (Jun 13)", ("NYK", "SAS")),
        ("HOU at OKC (Oct 21)", ("HOU", "OKC")),
        ("GSW at LAL (Oct 21)", ("GSW", "LAL")),
        ("something else entirely", (None, None)),
        (None, (None, None)),
    ],
)
def test_parse_event_subtitle(sub_title: str | None, expected: tuple) -> None:
    assert parse_event_subtitle(sub_title) == expected


@pytest.mark.parametrize(
    ("title", "expected"),
    [
        ("New York at San Antonio Winner?", ("New York", "San Antonio")),
        ("Game 5: New York at San Antonio Winner?", ("New York", "San Antonio")),
        # "vs" carries no home/away information, so none is invented.
        ("Boston vs Miami Winner?", (None, None)),
        ("Game 1: Boston vs Miami Winner?", (None, None)),
        (None, (None, None)),
    ],
)
def test_parse_title_orientation(title: str | None, expected: tuple) -> None:
    assert parse_title_orientation(title) == expected


# --- full market normalization --------------------------------------------


def normalized(raw: dict, event: dict | None = KALSHI_EVENT_NYKSAS, **kwargs) -> dict:
    events = {event["event_ticker"]: event} if event else {}
    return normalize_market(raw, events_by_ticker=events, **kwargs)


def test_normalizes_a_settled_playoff_market() -> None:
    row = normalized(KALSHI_MARKET_SAS)

    assert row["source"] == "kalshi"
    assert row["ticker"] == "KXNBAGAME-26JUN13NYKSAS-SAS"
    assert row["event_ticker"] == "KXNBAGAME-26JUN13NYKSAS"
    assert row["market_team_code"] == "SAS"
    assert row["away_team_code"] == "NYK"
    assert row["home_team_code"] == "SAS"
    assert row["market_team_is_home"] is True
    assert row["orientation_source"] == "event_sub_title"
    assert row["scheduled_game_date"] == date(2026, 6, 13)
    assert row["scheduled_date_source"] == "rules_primary"
    assert row["ticker_date_agrees_with_rules_date"] is True
    assert row["team_codes_agree_with_event_subtitle"] is True
    assert row["is_nba_matchup"] is True
    assert row["result"] == "no"
    assert row["status"] == "finalized"
    assert row["matchup_key"] == "2026-06-13|NYK|SAS"
    assert row["volume_fp"] == pytest.approx(48467702.82)
    assert row["open_time_utc"] == datetime(2026, 6, 9, 3, 55, tzinfo=UTC)
    assert row["occurrence_datetime_utc"] == datetime(2026, 6, 14, 3, 30, tzinfo=UTC)
    assert row["custom_strike_team_uuid"] == "ad36c3e8-4194-4e63-920f-7c50f46191a6"


def test_the_two_markets_of_an_event_share_a_matchup_key_and_split_orientation() -> None:
    home = normalized(KALSHI_MARKET_SAS)
    away = normalized(KALSHI_MARKET_NYK)

    assert home["matchup_key"] == away["matchup_key"]
    assert home["market_team_is_home"] is True
    assert away["market_team_is_home"] is False
    assert away["market_team_code"] == "NYK"


def test_no_sub_title_is_not_treated_as_the_opponent() -> None:
    """Live data has ``no_sub_title == yes_sub_title``; it names the same team."""
    assert KALSHI_MARKET_SAS["no_sub_title"] == KALSHI_MARKET_SAS["yes_sub_title"]
    row = normalized(KALSHI_MARKET_SAS)
    assert row["no_sub_title"] == "San Antonio"
    # The opponent still comes out right, because it comes from the event, not
    # from no_sub_title.
    assert row["away_team_code"] == "NYK"


def test_orientation_falls_back_to_the_event_ticker_when_no_event_is_supplied() -> None:
    row = normalized(KALSHI_MARKET_SAS, event=None)
    assert row["orientation_source"] == "event_ticker"
    assert (row["away_team_code"], row["home_team_code"]) == ("NYK", "SAS")
    assert row["event_sub_title"] is None
    # With only one source of orientation there is nothing to cross-check.
    assert row["team_codes_agree_with_event_subtitle"] is None


def test_los_angeles_subtitle_stays_ambiguous_without_breaking_the_row() -> None:
    """``yes_sub_title`` is a city label; for LA it cannot identify the club."""
    markets, event = kalshi_pair(
        event_ticker="KXNBAGAME-25OCT21GSWLAL",
        sub_title="GSW at LAL (Oct 21)",
        away="GSW",
        home="LAL",
        rules_date="Oct 21, 2025",
    )
    lal_market = {**markets[1], "yes_sub_title": "Los Angeles", "no_sub_title": "Los Angeles"}
    row = normalize_market(lal_market, events_by_ticker={event["event_ticker"]: event})

    # The team still resolves -- from the ticker suffix, not the ambiguous label.
    assert row["market_team_code"] == "LAL"
    assert row["market_team_is_home"] is True
    # ...and the ambiguous cross-check is recorded as "unknown", never guessed.
    assert row["market_team_agrees_with_yes_sub_title"] is None


def test_a_disagreement_between_event_and_ticker_is_flagged_not_hidden() -> None:
    mismatched_event = {**KALSHI_EVENT_NYKSAS, "sub_title": "BOS at MIA (Jun 13)"}
    row = normalized(KALSHI_MARKET_SAS, event=mismatched_event)

    assert row["team_codes_agree_with_event_subtitle"] is False
    # The event sub_title still wins, but the disagreement is on the record.
    assert (row["away_team_code"], row["home_team_code"]) == ("BOS", "MIA")


def test_a_date_disagreement_is_flagged() -> None:
    shifted = {**KALSHI_MARKET_SAS, "rules_primary": "...scheduled for Jun 14, 2026, then..."}
    row = normalized(shifted)
    assert row["scheduled_game_date"] == date(2026, 6, 14)
    assert row["ticker_date_agrees_with_rules_date"] is False


def test_unparseable_rules_fall_back_to_the_ticker_date() -> None:
    row = normalized({**KALSHI_MARKET_SAS, "rules_primary": "no date here"})
    assert row["scheduled_game_date"] == date(2026, 6, 13)
    assert row["scheduled_date_source"] == "event_ticker"
    assert row["ticker_date_agrees_with_rules_date"] is None


def test_a_non_nba_opponent_is_flagged_and_gets_no_matchup_key() -> None:
    """The archive contains an exhibition vs a non-NBA club (``GUA``)."""
    row = normalize_market(
        {
            **KALSHI_MARKET_SAS,
            "ticker": "KXNBAGAME-25OCT10GUABKN-BKN",
            "event_ticker": "KXNBAGAME-25OCT10GUABKN",
            "rules_primary": "...scheduled for Oct 10, 2025, then...",
        },
        events_by_ticker={},
    )
    assert row["away_team_code"] is None
    assert row["is_nba_matchup"] is False
    assert row["matchup_key"] is None


def test_source_endpoints_are_recorded_for_provenance() -> None:
    row = normalized(KALSHI_MARKET_SAS, source_endpoints=["markets", "historical"])
    assert row["source_endpoints"] == "historical,markets"


# --- frame building --------------------------------------------------------


def test_build_frame_has_the_documented_schema(raw_markets, raw_events) -> None:
    frame = build_markets_frame(raw_markets, events=raw_events)
    assert list(frame.columns) == KALSHI_MARKET_COLUMNS
    assert len(frame) == 2


def test_build_frame_deduplicates_by_ticker(raw_markets, raw_events) -> None:
    """The historical and live stores overlap; a ticker must appear once."""
    frame = build_markets_frame([*raw_markets, *raw_markets], events=raw_events)
    assert len(frame) == 2
    assert frame["ticker"].is_unique


def test_build_frame_rejects_a_market_without_a_ticker() -> None:
    with pytest.raises(ValueError, match="without a ticker"):
        build_markets_frame([{**KALSHI_MARKET_SAS, "ticker": None}])


def test_season_window_filters_other_seasons_out(raw_events) -> None:
    in_season, event = kalshi_pair(
        event_ticker="KXNBAGAME-25OCT21HOUOKC",
        sub_title="HOU at OKC (Oct 21)",
        away="HOU",
        home="OKC",
        rules_date="Oct 21, 2025",
    )
    frame = build_markets_frame(
        [*in_season, KALSHI_MARKET_SAS],
        events=[event, *raw_events],
        season_window=(date(2025, 7, 1), date(2026, 6, 30)),
    )
    # Both the October 2025 and June 2026 markets belong to 2025-26.
    assert len(frame) == 3

    narrow = build_markets_frame(
        [*in_season, KALSHI_MARKET_SAS],
        events=[event, *raw_events],
        season_window=(date(2024, 7, 1), date(2025, 6, 30)),
    )
    assert len(narrow) == 0


def test_markets_with_an_underivable_date_are_kept_not_dropped() -> None:
    """Dropping them silently would hide them from the report."""
    undated = {
        **KALSHI_MARKET_SAS,
        "ticker": "WEIRD-TICKER",
        "event_ticker": "WEIRD",
        "rules_primary": "no date here",
    }
    frame = build_markets_frame([undated], season_window=(date(2025, 7, 1), date(2026, 6, 30)))
    assert len(frame) == 1
    assert pd.isna(frame.iloc[0]["scheduled_game_date"])
