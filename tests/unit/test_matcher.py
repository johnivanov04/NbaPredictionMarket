"""Matching NBA games to Kalshi events: deterministic, and never arbitrary."""

from __future__ import annotations

from datetime import UTC, date, datetime
from typing import Any

import pandas as pd
import pytest

from nba_prediction_market.matching.game_market_matcher import (
    MATCH_COLUMNS,
    STATUS_AMBIGUOUS,
    STATUS_MATCHED,
    STATUS_UNMATCHED_KALSHI,
    STATUS_UNMATCHED_NBA,
    TIER_DATE_OFFSET,
    TIER_EXACT,
    build_kalshi_events_frame,
    match_games_to_markets,
)


def game(
    game_id: int,
    game_date: str,
    home: str,
    away: str,
    *,
    home_score: int | None = 110,
    away_score: int | None = 100,
    home_win: bool | None = True,
    postseason: bool = False,
) -> dict[str, Any]:
    return {
        "source_game_id": game_id,
        "game_date": date.fromisoformat(game_date),
        "season": 2025,
        "postseason": postseason,
        "status": "Final",
        "is_final": True,
        "tipoff_utc": datetime(2025, 10, 21, 23, 30, tzinfo=UTC),
        "home_team_code": home,
        "visitor_team_code": away,
        "home_score": home_score,
        "visitor_score": away_score,
        "home_win": home_win,
        "matchup_key": f"{game_date}|{'|'.join(sorted((home, away)))}",
    }


def event(
    event_ticker: str,
    game_date: str,
    home: str,
    away: str,
    *,
    home_result: str | None = "yes",
    away_result: str | None = "no",
) -> dict[str, Any]:
    return {
        "event_ticker": event_ticker,
        "matchup_key": f"{game_date}|{'|'.join(sorted((home, away)))}",
        "scheduled_game_date": date.fromisoformat(game_date),
        "home_team_code": home,
        "away_team_code": away,
        "title": f"{away} at {home} Winner?",
        "event_sub_title": f"{away} at {home}",
        "home_market_ticker": f"{event_ticker}-{home}",
        "away_market_ticker": f"{event_ticker}-{away}",
        "market_count": 2,
        "home_result": home_result,
        "away_result": away_result,
        "open_time_utc": datetime(2025, 10, 20, 12, 0, tzinfo=UTC),
        "close_time_utc": datetime(2025, 10, 22, 2, 0, tzinfo=UTC),
        "total_volume_fp": 1000.0,
        "is_nba_matchup": True,
    }


def run(games: list[dict], events: list[dict]):
    return match_games_to_markets(pd.DataFrame(games), pd.DataFrame(events))


# --- happy path ------------------------------------------------------------


def test_exact_match_on_date_and_team_pair() -> None:
    frame, summary = run(
        [game(1, "2025-10-21", "OKC", "HOU")],
        [event("KXNBAGAME-25OCT21HOUOKC", "2025-10-21", "OKC", "HOU")],
    )

    assert summary.to_dict() == {
        STATUS_MATCHED: 1,
        STATUS_UNMATCHED_NBA: 0,
        STATUS_UNMATCHED_KALSHI: 0,
        STATUS_AMBIGUOUS: 0,
    }
    row = frame.iloc[0]
    assert row["match_tier"] == TIER_EXACT
    assert row["date_offset_days"] == 0
    assert row["nba_game_id"] == 1
    assert row["kalshi_event_ticker"] == "KXNBAGAME-25OCT21HOUOKC"
    assert row["kalshi_home_market_ticker"] == "KXNBAGAME-25OCT21HOUOKC-OKC"
    assert row["kalshi_away_market_ticker"] == "KXNBAGAME-25OCT21HOUOKC-HOU"
    assert bool(row["orientation_agrees"]) is True
    assert bool(row["settlement_agrees_with_score"]) is True


def test_output_has_the_documented_schema() -> None:
    frame, _ = run([game(1, "2025-10-21", "OKC", "HOU")], [])
    assert list(frame.columns) == MATCH_COLUMNS


def test_every_record_from_both_sides_appears_exactly_once() -> None:
    games = [game(1, "2025-10-21", "OKC", "HOU"), game(2, "2025-10-22", "LAL", "GSW")]
    events = [
        event("KXNBAGAME-25OCT21HOUOKC", "2025-10-21", "OKC", "HOU"),
        event("KXNBAGAME-25OCT23BOSNYK", "2025-10-23", "NYK", "BOS"),
    ]
    frame, summary = run(games, events)

    assert summary.matched == 1
    assert summary.unmatched_nba == 1
    assert summary.unmatched_kalshi == 1
    assert frame["nba_game_id"].dropna().nunique() == 2
    assert frame["kalshi_event_ticker"].dropna().nunique() == 2


def test_orientation_disagreement_is_flagged_but_still_matched() -> None:
    """Same teams, same day, opposite home/away: report it rather than drop it."""
    frame, summary = run(
        [game(1, "2025-10-21", "OKC", "HOU")],
        [event("KXNBAGAME-25OCT21OKCHOU", "2025-10-21", "HOU", "OKC")],
    )
    assert summary.matched == 1
    assert bool(frame.iloc[0]["orientation_agrees"]) is False


def test_settlement_disagreement_with_the_final_score_is_flagged() -> None:
    frame, _ = run(
        [game(1, "2025-10-21", "OKC", "HOU", home_win=True)],
        [event("E", "2025-10-21", "OKC", "HOU", home_result="no", away_result="yes")],
    )
    assert bool(frame.iloc[0]["settlement_agrees_with_score"]) is False


@pytest.mark.parametrize(
    ("home_win", "home_result"),
    [(None, "yes"), (True, None), (True, ""), (None, None)],
)
def test_settlement_check_is_none_when_either_side_is_unknown(home_win, home_result) -> None:
    frame, _ = run(
        [game(1, "2025-10-21", "OKC", "HOU", home_win=home_win)],
        [event("E", "2025-10-21", "OKC", "HOU", home_result=home_result)],
    )
    assert frame.iloc[0]["settlement_agrees_with_score"] is None


# --- unmatched -------------------------------------------------------------


def test_unmatched_nba_game() -> None:
    frame, summary = run([game(1, "2025-10-21", "OKC", "HOU")], [])
    assert summary.unmatched_nba == 1
    row = frame.iloc[0]
    assert row["match_status"] == STATUS_UNMATCHED_NBA
    assert row["nba_game_id"] == 1
    assert pd.isna(row["kalshi_event_ticker"])


def test_unmatched_kalshi_event() -> None:
    frame, summary = run([], [event("E", "2025-10-21", "OKC", "HOU")])
    assert summary.unmatched_kalshi == 1
    row = frame.iloc[0]
    assert row["match_status"] == STATUS_UNMATCHED_KALSHI
    assert row["kalshi_event_ticker"] == "E"
    assert pd.isna(row["nba_game_id"])


def test_a_different_team_pair_on_the_same_date_does_not_match() -> None:
    _frame, summary = run(
        [game(1, "2025-10-21", "OKC", "HOU")],
        [event("E", "2025-10-21", "LAL", "GSW")],
    )
    assert summary.matched == 0
    assert summary.unmatched_nba == 1
    assert summary.unmatched_kalshi == 1


def test_records_without_a_usable_key_are_never_matched() -> None:
    """A row that could not be keyed must fall through to unmatched, not join."""
    keyless_game = {**game(1, "2025-10-21", "OKC", "HOU"), "matchup_key": None}
    keyless_event = {**event("E", "2025-10-21", "LAL", "GSW"), "matchup_key": None,
                     "home_team_code": None, "away_team_code": None}
    _frame, summary = run([keyless_game], [keyless_event])

    assert summary.matched == 0
    assert summary.unmatched_nba == 1
    assert summary.unmatched_kalshi == 1


# --- the relaxed tier ------------------------------------------------------


@pytest.mark.parametrize("event_date", ["2025-10-20", "2025-10-22"])
def test_matches_across_a_one_day_labelling_difference(event_date: str) -> None:
    """Late tip-offs can land the two sources on adjacent calendar days."""
    frame, summary = run(
        [game(1, "2025-10-21", "LAL", "GSW")],
        [event("E", event_date, "LAL", "GSW")],
    )
    assert summary.matched == 1
    row = frame.iloc[0]
    assert row["match_tier"] == TIER_DATE_OFFSET
    assert row["date_offset_days"] == (date.fromisoformat(event_date) - date(2025, 10, 21)).days


def test_does_not_bridge_a_two_day_gap() -> None:
    _frame, summary = run(
        [game(1, "2025-10-21", "LAL", "GSW")],
        [event("E", "2025-10-23", "LAL", "GSW")],
    )
    assert summary.matched == 0
    assert summary.unmatched_nba == 1
    assert summary.unmatched_kalshi == 1


def test_relaxed_tier_refuses_when_two_events_are_equally_close() -> None:
    """Two candidates one day either side: report ambiguity, do not pick one."""
    frame, summary = run(
        [game(1, "2025-10-21", "LAL", "GSW")],
        [event("E-BEFORE", "2025-10-20", "LAL", "GSW"),
         event("E-AFTER", "2025-10-22", "LAL", "GSW")],
    )

    assert summary.matched == 0
    ambiguous = frame[frame["match_status"] == STATUS_AMBIGUOUS]
    assert len(ambiguous) == 1
    assert set(ambiguous.iloc[0]["candidate_event_tickers"].split(",")) == {"E-AFTER", "E-BEFORE"}
    assert summary.unmatched_kalshi == 2


def test_relaxed_tier_requires_mutual_uniqueness() -> None:
    """One event a day from two different games must not be handed to either.

    A back-to-back between the same two clubs is the real-world shape here.
    """
    _frame, summary = run(
        [game(1, "2025-10-21", "LAL", "GSW"), game(2, "2025-10-23", "LAL", "GSW")],
        [event("E", "2025-10-22", "LAL", "GSW")],
    )
    assert summary.matched == 0
    assert summary.unmatched_nba == 2
    assert summary.unmatched_kalshi == 1


# --- ambiguity -------------------------------------------------------------


def test_two_games_and_two_events_on_one_key_are_all_reported_ambiguous() -> None:
    """A doubleheader-shaped collision: every involved record is surfaced."""
    games = [game(1, "2025-10-21", "OKC", "HOU"), game(2, "2025-10-21", "HOU", "OKC")]
    events = [
        event("E1", "2025-10-21", "OKC", "HOU"),
        event("E2", "2025-10-21", "HOU", "OKC"),
    ]
    frame, summary = run(games, events)

    assert summary.matched == 0
    assert summary.ambiguous == 4
    ambiguous = frame[frame["match_status"] == STATUS_AMBIGUOUS]
    assert set(ambiguous["candidate_nba_game_ids"].iloc[0].split(",")) == {"1", "2"}
    assert set(ambiguous["candidate_event_tickers"].iloc[0].split(",")) == {"E1", "E2"}
    assert "share matchup key" in ambiguous["ambiguity_reason"].iloc[0]


def test_two_games_one_event_is_ambiguous_not_a_coin_flip() -> None:
    _frame, summary = run(
        [game(1, "2025-10-21", "OKC", "HOU"), game(2, "2025-10-21", "HOU", "OKC")],
        [event("E1", "2025-10-21", "OKC", "HOU")],
    )
    assert summary.matched == 0
    assert summary.ambiguous == 3  # both games plus the event


def test_duplicate_events_on_one_key_do_not_produce_a_match() -> None:
    _frame, summary = run(
        [game(1, "2025-10-21", "OKC", "HOU")],
        [event("E1", "2025-10-21", "OKC", "HOU"), event("E2", "2025-10-21", "OKC", "HOU")],
    )
    assert summary.matched == 0
    assert summary.ambiguous == 3


# --- determinism -----------------------------------------------------------


def test_matching_is_order_independent() -> None:
    games = [
        game(1, "2025-10-21", "OKC", "HOU"),
        game(2, "2025-10-22", "LAL", "GSW"),
        game(3, "2025-10-23", "BOS", "NYK"),
    ]
    events = [
        event("E1", "2025-10-21", "OKC", "HOU"),
        event("E2", "2025-10-22", "LAL", "GSW"),
        event("E3", "2025-11-01", "MIA", "ORL"),
    ]
    forward, summary_a = run(games, events)
    reversed_, summary_b = run(list(reversed(games)), list(reversed(events)))

    assert summary_a == summary_b
    key_cols = ["match_status", "nba_game_id", "kalshi_event_ticker"]
    # Compare serialised, since NaN != NaN would defeat a record-wise comparison.
    assert forward[key_cols].to_csv(index=False) == reversed_[key_cols].to_csv(index=False)


def test_empty_inputs_produce_an_empty_but_well_formed_result() -> None:
    frame, summary = run([], [])
    assert list(frame.columns) == MATCH_COLUMNS
    assert len(frame) == 0
    assert summary.to_dict() == {
        STATUS_MATCHED: 0,
        STATUS_UNMATCHED_NBA: 0,
        STATUS_UNMATCHED_KALSHI: 0,
        STATUS_AMBIGUOUS: 0,
    }


# --- collapsing markets into events ---------------------------------------


def market_row(ticker: str, event_ticker: str, team: str, is_home: bool, **kwargs) -> dict:
    row = {
        "ticker": ticker,
        "event_ticker": event_ticker,
        "matchup_key": "2025-10-21|HOU|OKC",
        "scheduled_game_date": date(2025, 10, 21),
        "home_team_code": "OKC",
        "away_team_code": "HOU",
        "title": "HOU at OKC Winner?",
        "event_sub_title": "HOU at OKC (Oct 21)",
        "market_team_code": team,
        "market_team_is_home": is_home,
        "result": "yes" if is_home else "no",
        "open_time_utc": datetime(2025, 10, 20, 12, 0, tzinfo=UTC),
        "close_time_utc": datetime(2025, 10, 22, 2, 0, tzinfo=UTC),
        "volume_fp": 500.0,
        "is_nba_matchup": True,
    }
    row.update(kwargs)
    return row


def test_two_markets_collapse_into_one_event_with_both_sides_named() -> None:
    markets = pd.DataFrame(
        [
            market_row("E-OKC", "E", "OKC", True),
            market_row("E-HOU", "E", "HOU", False),
        ]
    )
    events = build_kalshi_events_frame(markets)

    assert len(events) == 1
    row = events.iloc[0]
    assert row["home_market_ticker"] == "E-OKC"
    assert row["away_market_ticker"] == "E-HOU"
    assert row["market_count"] == 2
    assert row["home_result"] == "yes"
    assert row["away_result"] == "no"
    assert row["total_volume_fp"] == 1000.0


def test_event_with_an_undetermined_side_leaves_that_ticker_null() -> None:
    """If orientation is unknown for both rows, neither side is asserted."""
    markets = pd.DataFrame(
        [
            market_row("E-A", "E", "OKC", None),
            market_row("E-B", "E", "HOU", None),
        ]
    )
    events = build_kalshi_events_frame(markets)
    row = events.iloc[0]
    assert row["home_market_ticker"] is None
    assert row["away_market_ticker"] is None
    assert row["market_count"] == 2


def test_empty_markets_frame_yields_an_empty_events_frame() -> None:
    events = build_kalshi_events_frame(pd.DataFrame())
    assert len(events) == 0
