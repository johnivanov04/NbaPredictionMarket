"""Phase 4A1: anchor discipline, event timing, and reaction measurement."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from nba_prediction_market.pipelines.build_anchor_models import (
    ANCHOR_BUNDLE,
    ANCHORS,
    derive_anchor_features,
)
from nba_prediction_market.pipelines.build_availability_event_study import (
    HORIZONS_MINUTES,
    MEANINGFUL_TRANSITIONS,
    ROLE_BANDS,
    WINDOW_END_MINUTES,
    WINDOW_START_MINUTES,
    Event,
    cluster_bootstrap_mean,
    market_around,
    role_band,
    signed_move,
)
from nba_prediction_market.pipelines.build_multi_anchor_market import (
    ANCHOR_MINUTES,
    LOOKBACK_MINUTES,
    window_for,
)

TIP = datetime(2026, 1, 15, 0, 30, tzinfo=UTC)


class TestAnchorWindow:
    def test_every_anchor_lies_inside_the_fetched_window(self):
        start, end = window_for(TIP)
        for minutes in ANCHOR_MINUTES:
            anchor = int((TIP - timedelta(minutes=minutes)).timestamp())
            assert start <= anchor <= end

    def test_the_window_ends_at_the_decision_anchor(self):
        _, end = window_for(TIP)
        assert end == int((TIP - timedelta(minutes=30)).timestamp())

    def test_post_anchor_states_are_structurally_unreachable(self):
        # T-15m and T-5m fall after the window's end, so they cannot be
        # fetched at all -- not merely left unused.
        _, end = window_for(TIP)
        for minutes in (15, 5):
            assert int((TIP - timedelta(minutes=minutes)).timestamp()) > end

    def test_the_window_is_six_hours(self):
        start, end = window_for(TIP)
        assert (end - start) == (LOOKBACK_MINUTES - 30) * 60

    def test_the_cache_slug_differs_from_phase_two(self):
        from nba_prediction_market.ingestion.candle_cache import cache_slug

        phase_two = cache_slug(
            minutes_before_tip=30, lookback_minutes=60, period_interval=1
        )
        phase_four = cache_slug(
            minutes_before_tip=30, lookback_minutes=LOOKBACK_MINUTES, period_interval=1
        )
        assert phase_two != phase_four


class TestAnchorFeatures:
    def _frame(self):
        row = {"season": 2025, "nba_game_id": 1}
        for anchor in ("t30", "t1h", "t3h", "t6h"):
            row[f"avail_{anchor}_covered"] = True
            for status in ("out", "doubtful", "questionable", "probable", "available"):
                row[f"avail_home_{anchor}_{status}_expected_minutes"] = 10.0
                row[f"avail_away_{anchor}_{status}_expected_minutes"] = 4.0
        return pd.DataFrame([row])

    def test_each_anchor_reads_its_own_state(self):
        frame = self._frame()
        frame.loc[0, "avail_home_t6h_out_expected_minutes"] = 30.0
        t6h = derive_anchor_features(frame, None, "t6h")
        t30 = derive_anchor_features(frame, None, "t30")
        assert t6h["avail_out_expected_minutes_diff"].iloc[0] == 26.0
        assert t30["avail_out_expected_minutes_diff"].iloc[0] == 6.0

    def test_an_uncovered_anchor_yields_null_not_zero(self):
        # Zero would assert nobody was designated; null lets the train-only
        # imputer handle a genuinely unknown state.
        frame = self._frame()
        frame.loc[0, "avail_t6h_covered"] = False
        derived = derive_anchor_features(frame, None, "t6h")
        assert pd.isna(derived["avail_out_expected_minutes_diff"].iloc[0])

    def test_the_bundle_is_the_frozen_3a3c_allowlist(self):
        from nba_prediction_market.models.availability_bundles import (
            PHASE_3A3_FEATURES,
            ROLE_MINUTE_FAMILY,
        )

        assert ANCHOR_BUNDLE == PHASE_3A3_FEATURES + ROLE_MINUTE_FAMILY

    def test_anchors_run_from_earliest_to_the_decision_anchor(self):
        minutes = [m for _, _, m in ANCHORS]
        assert minutes == sorted(minutes, reverse=True)
        assert minutes[-1] == 30


class TestEventDefinition:
    def test_transitions_are_declared_in_advance(self):
        assert ("questionable", "out") in MEANINGFUL_TRANSITIONS
        assert ("out", "available") in MEANINGFUL_TRANSITIONS
        # A non-change is not an event.
        assert ("out", "out") not in MEANINGFUL_TRANSITIONS

    def test_role_bands_are_fixed_not_derived_from_market_movement(self):
        assert [label for _, _, label in ROLE_BANDS] == [
            "low role", "medium role", "high role"
        ]
        assert role_band(30.0) == "high role"
        assert role_band(2.0) == "low role"
        assert role_band(None) == "unknown role"

    def test_the_study_window_precedes_the_decision_anchor(self):
        assert WINDOW_START_MINUTES == 360
        assert WINDOW_END_MINUTES == 30

    def test_a_downgrade_is_signed_toward_lower_team_probability(self):
        # News that a player is out should push the team down, so a downward
        # midpoint move counts as the expected direction.
        assert signed_move(-0.01, "downgrade") == pytest.approx(0.01)
        assert signed_move(+0.01, "downgrade") == pytest.approx(-0.01)
        assert signed_move(+0.01, "upgrade") == pytest.approx(0.01)

    def test_a_missing_move_stays_missing(self):
        assert signed_move(None, "downgrade") is None


class TestEventTiming:
    def _observations(self, stamp):
        times = [stamp - timedelta(minutes=m) for m in (3, 2, 1)] + [
            stamp + timedelta(minutes=m) for m in (0, 1, 5, 15, 30)
        ]
        return pd.DataFrame({
            "nba_game_id": 1,
            "side": "home",
            "observed_at_utc": times,
            "midpoint": [0.50] * 3 + [0.49, 0.48, 0.47, 0.46, 0.46],
            "yes_bid": [0.49] * 3 + [0.48, 0.47, 0.46, 0.45, 0.45],
            "yes_ask": [0.51] * 3 + [0.50, 0.49, 0.48, 0.47, 0.47],
            "spread": [0.02] * 8,
        })

    def _event(self, stamp):
        return Event(
            nba_game_id=1, team_code="BOS", player_id=7, player_name="X, Y",
            from_status="questionable", to_status="out",
            event_ts_utc=stamp, previous_report_ts_utc=stamp - timedelta(minutes=60),
            expected_minutes=30.0, direction="downgrade",
        )

    def test_a_report_cannot_affect_a_quote_before_its_stamp(self):
        stamp = pd.Timestamp("2026-01-15T22:30:00Z")
        result = market_around(self._observations(stamp), self._event(stamp), "BOS")
        assert result["pre_ts_utc"] < stamp

    def test_the_post_quote_is_the_first_that_actually_exists(self):
        stamp = pd.Timestamp("2026-01-15T22:30:00Z")
        result = market_around(self._observations(stamp), self._event(stamp), "BOS")
        assert result["post_ts_utc"] >= stamp
        assert result["post_latency_seconds"] >= 0

    def test_latency_is_preserved_rather_than_smoothed_away(self):
        stamp = pd.Timestamp("2026-01-15T22:30:00Z")
        result = market_around(self._observations(stamp), self._event(stamp), "BOS")
        assert "pre_latency_seconds" in result
        assert result["pre_latency_seconds"] > 0

    def test_bid_and_ask_are_carried_alongside_the_midpoint(self):
        stamp = pd.Timestamp("2026-01-15T22:30:00Z")
        result = market_around(self._observations(stamp), self._event(stamp), "BOS")
        for key in ("pre_bid", "pre_ask", "post_bid", "post_ask",
                    "pre_spread", "post_spread"):
            assert key in result

    def test_an_event_with_no_prior_observation_is_dropped(self):
        stamp = pd.Timestamp("2026-01-15T22:30:00Z")
        only_after = self._observations(stamp)
        only_after = only_after[only_after["observed_at_utc"] >= stamp]
        assert market_around(only_after, self._event(stamp), "BOS") is None

    def test_the_affected_side_is_chosen_by_team(self):
        stamp = pd.Timestamp("2026-01-15T22:30:00Z")
        observations = self._observations(stamp)
        observations["side"] = "away"
        # The event team is the away team here, so the away series is read.
        result = market_around(observations, self._event(stamp), "LAL")
        assert result["side"] == "away"

    def test_horizons_are_declared(self):
        assert HORIZONS_MINUTES == (5, 15, 30, 60)


class TestClusteredEventUncertainty:
    def test_events_in_one_game_are_not_treated_as_independent(self):
        # Several players can change status in the same game and share a
        # market, so games are the resampling unit.
        values = np.repeat([0.01, -0.01], 50)
        games = np.repeat([1, 2], 50)
        result = cluster_bootstrap_mean(values, games, n_resamples=500)
        assert result["n_games"] == 2
        assert result["method"] == "cluster bootstrap over games"

    def test_missing_moves_are_dropped_not_counted_as_zero(self):
        values = np.array([0.01, np.nan, 0.02])
        games = np.array([1, 2, 3])
        assert cluster_bootstrap_mean(values, games, n_resamples=200)["n"] == 2

    def test_an_empty_set_reports_nothing(self):
        assert cluster_bootstrap_mean(np.array([]), np.array([]))["n"] == 0
