"""Candle parsing, price handling, and lookahead-safe quote selection.

Covers requirements 1-13 and 21-22: prediction timestamps, timezone handling,
candle selection boundaries, staleness, missing sides, midpoint/spread, price
parsing, duplicate candles, and malformed responses.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta, timezone

import pytest

from nba_prediction_market.ingestion.candlesticks import (
    DERIVED_PRICE_DECIMALS,
    ISSUE_MISSING_ASK,
    ISSUE_MISSING_BID,
    ISSUE_NO_CANDLE_BEFORE,
    ISSUE_NO_CANDLES,
    ISSUE_NO_QUOTE_DATA,
    ISSUE_STALE,
    Candle,
    parse_candle,
    parse_candles,
    parse_price,
    prediction_timestamp,
    select_pregame_quote,
)

PRED = datetime(2026, 1, 15, 23, 30, tzinfo=UTC)
PRED_TS = int(PRED.timestamp())
MAX_AGE = 600.0  # 10 minutes


def candle(offset_seconds: int, *, bid=0.60, ask=0.61, last=0.60, prev=0.59) -> Candle:
    """A candle whose period ended ``offset_seconds`` before the prediction time."""
    return Candle(
        end_period_ts=PRED_TS - offset_seconds,
        yes_bid=bid,
        yes_ask=ask,
        last_trade_price=last,
        previous_trade_price=prev,
        volume=1000.0,
        open_interest=50000.0,
    )


# --- 1 & 2: prediction timestamp and timezones -----------------------------


def test_prediction_ts_is_tipoff_minus_exactly_30_minutes() -> None:
    tipoff = datetime(2026, 1, 15, 23, 30, tzinfo=UTC)
    assert prediction_timestamp(tipoff, 30) == datetime(2026, 1, 15, 23, 0, tzinfo=UTC)
    assert (tipoff - prediction_timestamp(tipoff, 30)) == timedelta(minutes=30)


@pytest.mark.parametrize("minutes", [0, 5, 30, 60, 120])
def test_prediction_ts_honours_any_offset(minutes: int) -> None:
    tipoff = datetime(2026, 1, 15, 23, 30, tzinfo=UTC)
    assert tipoff - prediction_timestamp(tipoff, minutes) == timedelta(minutes=minutes)


def test_prediction_ts_converts_non_utc_input_to_utc() -> None:
    """A tipoff given in another zone must anchor to the same instant."""
    eastern = timezone(timedelta(hours=-5))
    tipoff = datetime(2026, 1, 15, 18, 30, tzinfo=eastern)  # 23:30 UTC
    result = prediction_timestamp(tipoff, 30)
    assert result == datetime(2026, 1, 15, 23, 0, tzinfo=UTC)
    assert result.tzinfo is not None
    assert result.utcoffset() == timedelta(0)


def test_prediction_ts_rejects_naive_datetimes() -> None:
    """Guessing a zone here would silently shift every anchor."""
    with pytest.raises(ValueError, match="timezone-aware"):
        prediction_timestamp(datetime(2026, 1, 15, 23, 30), 30)


def test_prediction_ts_rejects_negative_offset() -> None:
    with pytest.raises(ValueError, match="must be >= 0"):
        prediction_timestamp(datetime(2026, 1, 15, tzinfo=UTC), -5)


def test_select_rejects_naive_prediction_ts() -> None:
    with pytest.raises(ValueError, match="timezone-aware"):
        select_pregame_quote([candle(0)], datetime(2026, 1, 15, 23, 30), max_age_seconds=MAX_AGE)


# --- 3, 4, 5: selection boundaries -----------------------------------------


def test_candle_exactly_at_prediction_ts_is_selected() -> None:
    """end_period_ts == prediction_ts contains no post-anchor data, so it counts."""
    chosen = candle(0)
    result = select_pregame_quote([candle(120), chosen], PRED, max_age_seconds=MAX_AGE)
    assert result.candle == chosen
    assert result.quote_age_seconds == 0.0
    assert result.usable is True
    assert result.issue is None


def test_most_recent_candle_before_prediction_ts_wins() -> None:
    newest = candle(60)
    result = select_pregame_quote(
        [candle(600), candle(300), newest], PRED, max_age_seconds=MAX_AGE
    )
    assert result.candle == newest
    assert result.quote_age_seconds == 60.0
    assert result.usable is True


def test_candle_after_prediction_ts_is_never_selected() -> None:
    """The core lookahead guard."""
    future = candle(-60)   # one minute AFTER the anchor
    past = candle(120)
    result = select_pregame_quote([past, future], PRED, max_age_seconds=MAX_AGE)

    assert result.candle == past
    assert result.candle.end_period_ts < PRED_TS
    assert result.quote_age_seconds == 120.0


def test_only_future_candles_yields_no_quote() -> None:
    result = select_pregame_quote([candle(-60), candle(-120)], PRED, max_age_seconds=MAX_AGE)
    assert result.candle is None
    assert result.issue == ISSUE_NO_CANDLE_BEFORE
    assert result.usable is False


def test_unsorted_input_still_selects_the_latest_eligible_candle() -> None:
    result = select_pregame_quote(
        [candle(60), candle(600), candle(-30), candle(300)], PRED, max_age_seconds=MAX_AGE
    )
    assert result.quote_age_seconds == 60.0


# --- 6: no candles ---------------------------------------------------------


def test_no_candles_marks_the_quote_missing() -> None:
    result = select_pregame_quote([], PRED, max_age_seconds=MAX_AGE)
    assert result.candle is None
    assert result.issue == ISSUE_NO_CANDLES
    assert result.usable is False
    assert result.quote_age_seconds is None


# --- 7 & 8: staleness ------------------------------------------------------


def test_quote_older_than_the_limit_is_kept_but_unusable() -> None:
    """Preserved for diagnostics, never promoted to usable."""
    stale = candle(900)  # 15 minutes old, limit is 10
    result = select_pregame_quote([stale], PRED, max_age_seconds=MAX_AGE)

    assert result.candle == stale
    assert result.quote_age_seconds == 900.0
    assert result.issue == ISSUE_STALE
    assert result.usable is False
    # The prices are still there to inspect.
    assert result.candle.yes_bid == 0.60


def test_quote_age_exactly_at_the_threshold_is_usable() -> None:
    result = select_pregame_quote([candle(600)], PRED, max_age_seconds=600.0)
    assert result.quote_age_seconds == 600.0
    assert result.usable is True
    assert result.issue is None


def test_one_second_past_the_threshold_is_stale() -> None:
    result = select_pregame_quote([candle(601)], PRED, max_age_seconds=600.0)
    assert result.usable is False
    assert result.issue == ISSUE_STALE


def test_a_stale_quote_is_not_rescued_by_an_older_fresh_looking_one() -> None:
    """There is no reaching further back to find a 'better' quote."""
    result = select_pregame_quote(
        [candle(1200, bid=0.5, ask=0.5), candle(900)], PRED, max_age_seconds=MAX_AGE
    )
    assert result.quote_age_seconds == 900.0
    assert result.issue == ISSUE_STALE


# --- 9 & 10: missing sides -------------------------------------------------


def test_missing_bid_is_flagged_and_blocks_midpoint() -> None:
    result = select_pregame_quote([candle(0, bid=None)], PRED, max_age_seconds=MAX_AGE)
    assert result.issue == ISSUE_MISSING_BID
    assert result.usable is False
    assert result.candle.yes_ask == 0.61
    assert result.candle.midpoint is None
    assert result.candle.spread is None


def test_missing_ask_is_flagged_and_blocks_midpoint() -> None:
    result = select_pregame_quote([candle(0, ask=None)], PRED, max_age_seconds=MAX_AGE)
    assert result.issue == ISSUE_MISSING_ASK
    assert result.usable is False
    assert result.candle.yes_bid == 0.60
    assert result.candle.midpoint is None


def test_a_candle_with_neither_side_is_skipped_in_favour_of_an_older_quoted_one() -> None:
    """An empty book is not a quote; the last real quote is used instead."""
    quoted = candle(180)
    result = select_pregame_quote(
        [quoted, candle(60, bid=None, ask=None)], PRED, max_age_seconds=MAX_AGE
    )
    assert result.candle == quoted
    assert result.quote_age_seconds == 180.0
    assert result.usable is True


def test_all_candles_unquoted_reports_no_quote_data() -> None:
    newest = candle(60, bid=None, ask=None)
    result = select_pregame_quote(
        [candle(300, bid=None, ask=None), newest], PRED, max_age_seconds=MAX_AGE
    )
    assert result.issue == ISSUE_NO_QUOTE_DATA
    assert result.usable is False
    # Newest eligible candle is retained so the gap is visible.
    assert result.candle == newest


def test_trade_price_alone_never_manufactures_a_quote() -> None:
    """last_trade_price must not stand in for bid/ask."""
    result = select_pregame_quote(
        [candle(0, bid=None, ask=None, last=0.62)], PRED, max_age_seconds=MAX_AGE
    )
    assert result.usable is False
    assert result.candle.midpoint is None
    assert result.candle.last_trade_price == 0.62


# --- 11 & 12: midpoint and spread ------------------------------------------


@pytest.mark.parametrize(
    ("bid", "ask", "midpoint", "spread"),
    [
        (0.60, 0.61, 0.605, pytest.approx(0.01)),
        (0.00, 1.00, 0.50, pytest.approx(1.00)),
        (0.49, 0.49, 0.49, pytest.approx(0.0)),
        (0.12, 0.15, 0.135, pytest.approx(0.03)),
    ],
)
def test_midpoint_and_spread(bid, ask, midpoint, spread) -> None:
    c = candle(0, bid=bid, ask=ask)
    assert c.midpoint == pytest.approx(midpoint)
    assert c.spread == spread


@pytest.mark.parametrize(("bid", "ask"), [(None, 0.61), (0.60, None), (None, None)])
def test_midpoint_and_spread_require_both_sides(bid, ask) -> None:
    c = candle(0, bid=bid, ask=ask)
    assert c.midpoint is None
    assert c.spread is None


# --- 13: price parsing -----------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("0.6500", 0.65),
        ("0.0000", 0.0),
        ("1.0000", 1.0),
        (0.42, 0.42),
        ("0.01", 0.01),
    ],
)
def test_parse_price_accepts_decimal_dollars(raw, expected) -> None:
    assert parse_price(raw) == pytest.approx(expected)


@pytest.mark.parametrize(
    "raw",
    [
        None,
        "",
        "abc",
        "65",       # cents would be out of range -- never silently rescaled
        "-0.10",
        "1.5",
        float("nan"),
        True,       # bool is not a price
    ],
)
def test_parse_price_rejects_anything_outside_zero_to_one(raw) -> None:
    assert parse_price(raw) is None


# --- candle parsing, both endpoint schemas ---------------------------------

HISTORICAL_CANDLE = {
    "end_period_ts": 1781406000,
    "open_interest": "19046657.69",
    "price": {"close": "0.7200", "high": "0.7600", "low": "0.7200",
              "mean": "0.7494", "open": "0.7600", "previous": "0.7600"},
    "volume": "60555.79",
    "yes_ask": {"close": "0.7300", "high": "0.7600", "low": "0.7300", "open": "0.7600"},
    "yes_bid": {"close": "0.7200", "high": "0.7500", "low": "0.7200", "open": "0.7500"},
}

SERIES_CANDLE = {
    "end_period_ts": 1781406000,
    "open_interest_fp": "19046657.69",
    "price": {"close_dollars": "0.7200", "high_dollars": "0.7600", "low_dollars": "0.7200",
              "mean_dollars": "0.7494", "open_dollars": "0.7600", "previous_dollars": "0.7600"},
    "volume_fp": "60555.79",
    "yes_ask": {"close_dollars": "0.7300", "high_dollars": "0.7600",
                "low_dollars": "0.7300", "open_dollars": "0.7600"},
    "yes_bid": {"close_dollars": "0.7200", "high_dollars": "0.7500",
                "low_dollars": "0.7200", "open_dollars": "0.7500"},
}


@pytest.mark.parametrize("raw", [HISTORICAL_CANDLE, SERIES_CANDLE])
def test_both_endpoint_schemas_parse_identically(raw: dict) -> None:
    """The two endpoints name the same values differently; neither may go null."""
    parsed = parse_candle(raw)
    assert parsed is not None
    assert parsed.end_period_ts == 1781406000
    assert parsed.yes_bid == pytest.approx(0.72)
    assert parsed.yes_ask == pytest.approx(0.73)
    assert parsed.last_trade_price == pytest.approx(0.72)
    assert parsed.previous_trade_price == pytest.approx(0.76)
    assert parsed.volume == pytest.approx(60555.79)
    assert parsed.open_interest == pytest.approx(19046657.69)
    assert parsed.midpoint == pytest.approx(0.725)
    assert parsed.spread == pytest.approx(0.01)


def test_the_two_schemas_produce_equal_candles() -> None:
    assert parse_candle(HISTORICAL_CANDLE) == parse_candle(SERIES_CANDLE)


def test_bid_and_ask_come_from_the_period_close_not_open_or_high() -> None:
    """The quote 'as of' end_period_ts is the close, not the period's extremes."""
    parsed = parse_candle(HISTORICAL_CANDLE)
    assert parsed.yes_ask == pytest.approx(0.73)   # close, not high 0.76 or open 0.76
    assert parsed.yes_bid == pytest.approx(0.72)   # close, not high 0.75 or open 0.75


def test_candle_end_ts_is_utc_aware() -> None:
    parsed = parse_candle(HISTORICAL_CANDLE)
    assert parsed.end_ts_utc == datetime(2026, 6, 14, 3, 0, tzinfo=UTC)
    assert parsed.end_ts_utc.tzinfo is not None


# --- 22: malformed responses -----------------------------------------------


@pytest.mark.parametrize(
    "raw",
    [
        None,
        "not a dict",
        {},                                 # no end_period_ts
        {"end_period_ts": None},
        {"end_period_ts": "not a number"},
        {"end_period_ts": True},            # bool is not a timestamp
        [1, 2, 3],
    ],
)
def test_malformed_candles_are_rejected_not_guessed(raw) -> None:
    assert parse_candle(raw) is None


def test_candle_with_missing_subobjects_parses_with_null_prices() -> None:
    parsed = parse_candle({"end_period_ts": 1781406000})
    assert parsed is not None
    assert parsed.yes_bid is None and parsed.yes_ask is None
    assert parsed.has_quote is False


def test_candle_with_non_dict_subobjects_does_not_crash() -> None:
    parsed = parse_candle({"end_period_ts": 1, "price": "nope", "yes_bid": 5, "yes_ask": None})
    assert parsed is not None
    assert parsed.yes_bid is None


@pytest.mark.parametrize(
    "payload",
    [None, "text", {}, {"candlesticks": None}, {"candlesticks": "nope"}, {"other": []}],
)
def test_malformed_payloads_yield_no_candles(payload) -> None:
    candles, malformed = parse_candles(payload)
    assert candles == []
    assert malformed == 0


def test_malformed_candles_are_counted_not_silently_dropped() -> None:
    payload = {"candlesticks": [HISTORICAL_CANDLE, {"bad": True}, None, "x"]}
    candles, malformed = parse_candles(payload)
    assert len(candles) == 1
    assert malformed == 3


# --- 21: duplicate candles -------------------------------------------------


def test_duplicate_timestamps_collapse_to_one_candle() -> None:
    first = {**HISTORICAL_CANDLE, "yes_bid": {"close": "0.7000"}}
    second = {**HISTORICAL_CANDLE, "yes_bid": {"close": "0.7200"}}
    candles, malformed = parse_candles({"candlesticks": [first, second]})

    assert len(candles) == 1
    assert malformed == 0
    # The later record in the response wins, deterministically.
    assert candles[0].yes_bid == pytest.approx(0.72)


def test_candles_are_returned_in_timestamp_order() -> None:
    payload = {
        "candlesticks": [
            {**HISTORICAL_CANDLE, "end_period_ts": 300},
            {**HISTORICAL_CANDLE, "end_period_ts": 100},
            {**HISTORICAL_CANDLE, "end_period_ts": 200},
        ]
    }
    candles, _ = parse_candles(payload)
    assert [c.end_period_ts for c in candles] == [100, 200, 300]


def test_duplicates_do_not_change_the_selected_quote() -> None:
    duplicated = [candle(60), candle(60), candle(120)]
    result = select_pregame_quote(duplicated, PRED, max_age_seconds=MAX_AGE)
    assert result.quote_age_seconds == 60.0


def test_selection_reports_how_much_it_looked_at() -> None:
    result = select_pregame_quote(
        [candle(300), candle(60)], PRED, max_age_seconds=MAX_AGE, malformed_candles=2
    )
    assert result.candles_considered == 2
    assert result.malformed_candles == 2


# --- derived-price precision ----------------------------------------------


def test_derived_prices_carry_no_binary_float_noise() -> None:
    """0.605 + 0.385 must be 0.99, not 0.9900000000000001.

    Real Kalshi quotes are whole cents, so midpoints land on half-cents. Left
    unrounded, ``abs(sum - 1)`` produced 0.010000000000000009, which compares
    greater than 0.01 and silently inflated every threshold count.
    """
    home = candle(0, bid=0.60, ask=0.61)
    away = candle(0, bid=0.38, ask=0.39)
    assert home.midpoint == 0.605
    assert away.midpoint == 0.385
    total = round(home.midpoint + away.midpoint, DERIVED_PRICE_DECIMALS)
    assert total == 0.99

    # The deviation needs rounding too: abs(0.99 - 1.0) is 0.010000000000000009
    # in binary floating point, which compares greater than 0.01.
    raw_deviation = abs(total - 1.0)
    assert raw_deviation > 0.01                       # the trap
    deviation = round(raw_deviation, DERIVED_PRICE_DECIMALS)
    assert deviation == 0.01
    assert not deviation > 0.01                       # what the report counts


@pytest.mark.parametrize(
    ("bid", "ask", "expected_spread"),
    [(0.60, 0.61, 0.01), (0.38, 0.39, 0.01), (0.07, 0.09, 0.02), (0.99, 1.00, 0.01)],
)
def test_spread_is_exact_for_whole_cent_quotes(bid, ask, expected_spread) -> None:
    assert candle(0, bid=bid, ask=ask).spread == expected_spread


def test_equal_midpoints_compare_equal_across_different_quotes() -> None:
    """Grouping on midpoint must not fragment on representation."""
    assert candle(0, bid=0.10, ask=0.20).midpoint == candle(0, bid=0.05, ask=0.25).midpoint
