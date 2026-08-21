"""Multi-anchor reconstruction from a single cached candle stream."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from nba_prediction_market.availability.kalshi_anchors import (
    CAPTURE_WINDOW_HOURS,
    RESEARCH_ANCHORS_MINUTES,
    anchor_label,
    capture_window,
    quotes_at_anchors,
)
from nba_prediction_market.ingestion.candlesticks import Candle

TIP = datetime(2026, 1, 15, 0, 30, tzinfo=UTC)


def _candle(minutes_before_tip: int, bid: float, ask: float) -> Candle:
    ts = int((TIP - timedelta(minutes=minutes_before_tip)).timestamp())
    return Candle(
        end_period_ts=ts,
        yes_bid=bid,
        yes_ask=ask,
        last_trade_price=(bid + ask) / 2,
        previous_trade_price=None,
        volume=10.0,
        open_interest=100.0,
    )


# One candle per minute across the full day before tipoff. The price drifts
# upward so each anchor has a distinguishable value.
STREAM = [_candle(m, 0.30 + m / 10_000, 0.32 + m / 10_000) for m in range(1441, 0, -1)]


class TestAnchorLabels:
    @pytest.mark.parametrize(
        ("minutes", "expected"),
        [(1440, "T-24h"), (360, "T-6h"), (180, "T-3h"), (60, "T-1h"),
         (30, "T-30m"), (15, "T-15m"), (5, "T-5m")],
    )
    def test_labels_match_the_documented_anchor_names(self, minutes, expected):
        assert anchor_label(minutes) == expected


class TestCaptureWindow:
    def test_window_covers_every_research_anchor(self):
        start, end = capture_window(TIP)
        assert end == int(TIP.timestamp())
        assert start == int((TIP - timedelta(hours=CAPTURE_WINDOW_HOURS)).timestamp())
        for minutes in RESEARCH_ANCHORS_MINUTES:
            assert start <= int((TIP - timedelta(minutes=minutes)).timestamp()) <= end

    def test_a_naive_tipoff_is_rejected(self):
        with pytest.raises(ValueError, match="timezone-aware"):
            capture_window(datetime(2026, 1, 15, 0, 30))


class TestAnchorQuotes:
    def test_every_anchor_is_produced_newest_last(self):
        quotes = quotes_at_anchors(STREAM, TIP)
        assert [q.minutes_before_tip for q in quotes] == sorted(
            RESEARCH_ANCHORS_MINUTES, reverse=True
        )

    def test_no_anchor_sees_a_candle_from_after_itself(self):
        for quote in quotes_at_anchors(STREAM, TIP):
            assert quote.quote_ts_utc is not None
            assert quote.quote_ts_utc <= quote.anchor_utc

    def test_each_anchor_resolves_to_its_own_distinct_quote(self):
        quotes = quotes_at_anchors(STREAM, TIP)
        assert len({q.quote_ts_utc for q in quotes}) == len(quotes)
        assert all(q.usable for q in quotes)

    def test_the_t30_anchor_agrees_with_the_phase_2_selector(self):
        from nba_prediction_market.ingestion.candlesticks import select_pregame_quote

        anchor = TIP - timedelta(minutes=30)
        expected = select_pregame_quote(list(STREAM), anchor, max_age_seconds=600.0)
        t30 = next(q for q in quotes_at_anchors(STREAM, TIP) if q.minutes_before_tip == 30)
        assert t30.quote_ts_utc == expected.candle.end_ts_utc
        assert t30.midpoint == expected.candle.midpoint

    def test_a_market_that_opened_late_marks_early_anchors_unusable(self):
        # Trading only starts three hours before tip: the 24h and 6h anchors
        # have nothing to see and must say so rather than borrowing later data.
        late = [c for c in STREAM if c.end_period_ts >= int((TIP - timedelta(hours=3)).timestamp())]
        quotes = {q.anchor_label: q for q in quotes_at_anchors(late, TIP)}
        assert not quotes["T-24h"].usable
        assert quotes["T-24h"].midpoint is None
        assert not quotes["T-6h"].usable
        assert quotes["T-3h"].usable
        assert quotes["T-30m"].usable

    def test_a_stale_quote_is_flagged_but_its_values_are_kept(self):
        # Last candle is 40 minutes before tip, so T-5m is 35 minutes stale.
        cutoff = int((TIP - timedelta(minutes=40)).timestamp())
        sparse = [c for c in STREAM if c.end_period_ts <= cutoff]
        t5 = next(q for q in quotes_at_anchors(sparse, TIP) if q.minutes_before_tip == 5)
        assert not t5.usable
        assert t5.quote_age_seconds is not None and t5.quote_age_seconds > 600
        assert t5.midpoint is not None

    def test_a_naive_tipoff_is_rejected(self):
        with pytest.raises(ValueError, match="timezone-aware"):
            quotes_at_anchors(STREAM, datetime(2026, 1, 15, 0, 30))

    def test_quotes_serialise_to_json_safe_primitives(self):
        row = quotes_at_anchors(STREAM, TIP)[0].to_dict()
        assert row["anchor"] == "T-24h"
        assert isinstance(row["anchor_utc"], str)
        assert isinstance(row["quote_ts_utc"], str)
