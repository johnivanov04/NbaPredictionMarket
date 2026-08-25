"""Phase 4A2: observation timing, event triggers, health, and safety."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import ClassVar
from zoneinfo import ZoneInfo

import pytest

from nba_prediction_market.capture.event_trigger import (
    BASELINE_INTERVAL_SECONDS,
    EVENT_INTERVAL_SECONDS,
    EVENT_TARGET_OFFSETS_SECONDS,
    MEANINGFUL_ROLE_MINUTES,
    CaptureScheduler,
    StatusChange,
    diff_states,
)
from nba_prediction_market.capture.health import (
    CRITICAL,
    INFO,
    WARNING,
    CollectorState,
    assess,
    overall_status,
    storage_estimate,
)
from nba_prediction_market.capture.kalshi_live import (
    WEBSOCKET_REQUIRES_AUTH,
    BookSnapshot,
    derive_yes_ask,
    parse_book,
    streaming_capability,
)
from nba_prediction_market.capture.observation import (
    ObservationTiming,
    RawObservation,
    content_hash,
    visible_at,
)

NOW = datetime(2026, 10, 20, 22, 30, tzinfo=UTC)


def _timing(**overrides) -> ObservationTiming:
    base = {
        "request_started_at_utc": NOW,
        "response_received_at_utc": NOW + timedelta(milliseconds=300),
        "first_observed_at_utc": NOW + timedelta(milliseconds=300),
        "source_timestamp_utc": NOW - timedelta(seconds=12),
    }
    return ObservationTiming(**{**base, **overrides})


class TestObservationTiming:
    def test_source_time_is_not_treated_as_availability(self):
        timing = _timing()
        # The publisher says 12 seconds earlier; we could only act on it when
        # we actually received it.
        assert timing.actionable_from() == timing.first_observed_at_utc
        assert timing.actionable_from() > timing.source_timestamp_utc

    def test_publication_latency_is_measured(self):
        assert _timing().publication_latency_seconds == pytest.approx(12.3, abs=0.01)

    def test_round_trip_is_measured_separately(self):
        assert _timing().round_trip_seconds == pytest.approx(0.3, abs=0.001)

    def test_a_source_without_a_timestamp_has_no_latency(self):
        assert _timing(source_timestamp_utc=None).publication_latency_seconds is None

    def test_negative_latency_is_surfaced_not_clamped(self):
        # Observing before the publisher's stamp means a clock problem, and
        # hiding it would hide the bug.
        timing = _timing(source_timestamp_utc=NOW + timedelta(seconds=60))
        assert timing.publication_latency_seconds < 0

    def test_naive_timestamps_are_refused(self):
        with pytest.raises(ValueError, match="timezone-aware"):
            _timing(request_started_at_utc=datetime(2026, 10, 20, 22, 30))

    def test_a_response_cannot_precede_its_request(self):
        with pytest.raises(ValueError, match="cannot precede"):
            _timing(response_received_at_utc=NOW - timedelta(seconds=1))


class TestVisibility:
    def _observation(self, first_observed: datetime, source: datetime):
        return RawObservation(
            source="nba_official", url="https://example.test/x.pdf",
            http_status=200,
            timing=_timing(
                request_started_at_utc=first_observed,
                response_received_at_utc=first_observed,
                first_observed_at_utc=first_observed,
                source_timestamp_utc=source,
            ),
            content_sha256="abc", content_bytes=10,
        )

    def test_replay_filters_on_observation_time_not_source_time(self):
        # Stamped early, seen late: at the stamp it must be invisible.
        late = self._observation(NOW + timedelta(minutes=5), NOW)
        assert visible_at([late], NOW) == []
        assert visible_at([late], NOW + timedelta(minutes=5)) == [late]

    def test_an_observation_exactly_at_the_moment_is_visible(self):
        exact = self._observation(NOW, NOW)
        assert visible_at([exact], NOW) == [exact]

    def test_a_naive_moment_is_refused(self):
        with pytest.raises(ValueError, match="timezone-aware"):
            visible_at([], datetime(2026, 10, 20, 22, 30))

    def test_content_hash_detects_an_unchanged_artefact(self):
        assert content_hash(b"same") == content_hash(b"same")
        assert content_hash(b"same") != content_hash(b"different")


class TestOrderbookDerivation:
    def test_the_yes_ask_comes_from_the_no_side(self):
        # Kalshi returns bids only: a NO bid at 0.43 is a YES ask at 0.57.
        assert derive_yes_ask([(0.38, 600.0), (0.43, 1210.0)]) == (0.57, 1210.0)

    def test_an_empty_no_side_has_no_ask(self):
        assert derive_yes_ask([]) is None

    def test_a_snapshot_reports_bid_ask_and_spread(self):
        snap = BookSnapshot(
            market_ticker="X", observed_at_utc=NOW, exchange_updated_at_utc=None,
            yes_bids=[(0.30, 10.0), (0.36, 2000.0)],
            no_bids=[(0.38, 600.0), (0.43, 1210.0)],
        )
        assert snap.best_yes_bid == (0.36, 2000.0)
        assert snap.best_yes_ask == (0.57, 1210.0)
        assert snap.spread == pytest.approx(0.21)
        assert snap.midpoint == pytest.approx(0.465)

    def test_depth_counts_only_observed_levels(self):
        snap = BookSnapshot(
            market_ticker="X", observed_at_utc=NOW, exchange_updated_at_utc=None,
            yes_bids=[(0.34, 5.0), (0.35, 7.0), (0.36, 11.0)], no_bids=[],
        )
        assert snap.depth_within(0.02)["yes_bid_size"] == pytest.approx(23.0)
        assert snap.depth_within(0.0)["yes_bid_size"] == pytest.approx(11.0)
        # No ask side observed means no ask depth invented.
        assert snap.depth_within(0.02)["yes_ask_size"] == 0.0

    def test_parsing_handles_the_real_payload_shape(self):
        snap = parse_book(
            "X",
            {"orderbook_fp": {"yes_dollars": [["0.3600", "2000.00"]],
                              "no_dollars": [["0.4300", "1210.32"]]}},
            NOW,
            {"market": {"updated_time": "2026-10-20T22:29:00Z",
                        "last_price_dollars": "0.4000", "status": "open"}},
        )
        assert snap.best_yes_ask[0] == pytest.approx(0.57)
        assert snap.exchange_updated_at_utc == datetime(2026, 10, 20, 22, 29, tzinfo=UTC)
        assert snap.status == "open"


class TestStreamingHonesty:
    def test_streaming_is_reported_unavailable_with_a_reason(self):
        capability = streaming_capability()
        assert WEBSOCKET_REQUIRES_AUTH is True
        assert capability["available_to_this_project"] is False
        assert "credentials" in capability["reason"]

    def test_a_polling_fallback_is_named(self):
        assert "REST" in streaming_capability()["fallback"]

    def test_an_upgrade_path_is_documented(self):
        assert "KALSHI_API_KEY" in streaming_capability()["upgrade_path"]


class TestStatusChangeDetection:
    ROLES: ClassVar[dict] = {(1, "p1"): 30.0, (1, "p2"): 3.0}

    def _current(self, **statuses):
        return {(1, "BOS", k): v for k, v in statuses.items()}

    def test_the_first_report_of_a_night_is_a_baseline_not_a_burst(self):
        # Calling every player in the first report a change would manufacture
        # events that never happened.
        changes = diff_states(
            None, self._current(p1="out"), detected_at_utc=NOW,
            source_report_timestamp_utc=NOW, roles=self.ROLES,
        )
        assert changes == []

    def test_a_status_change_is_detected(self):
        changes = diff_states(
            self._current(p1="questionable"), self._current(p1="out"),
            detected_at_utc=NOW, source_report_timestamp_utc=NOW, roles=self.ROLES,
        )
        assert len(changes) == 1
        assert changes[0].from_status == "questionable"
        assert changes[0].to_status == "out"
        assert changes[0].direction == "downgrade"

    def test_a_player_dropping_off_becomes_not_reported_never_available(self):
        changes = diff_states(
            self._current(p1="out"), {}, detected_at_utc=NOW,
            source_report_timestamp_utc=NOW, roles=self.ROLES,
        )
        assert changes[0].to_status == "not_reported"
        assert changes[0].to_status != "available"

    def test_an_unchanged_status_is_not_an_event(self):
        assert diff_states(
            self._current(p1="out"), self._current(p1="out"),
            detected_at_utc=NOW, source_report_timestamp_utc=NOW, roles=self.ROLES,
        ) == []

    def test_upgrades_are_captured_not_only_downgrades(self):
        changes = diff_states(
            self._current(p1="out"), self._current(p1="available"),
            detected_at_utc=NOW, source_report_timestamp_utc=NOW, roles=self.ROLES,
        )
        assert changes[0].direction == "upgrade"

    def test_role_weight_decides_whether_sampling_is_raised(self):
        big = StatusChange(1, "BOS", "p1", "A", "questionable", "out", NOW, NOW, 30.0)
        small = StatusChange(1, "BOS", "p2", "B", "questionable", "out", NOW, NOW, 3.0)
        assert big.is_meaningful
        assert not small.is_meaningful
        assert MEANINGFUL_ROLE_MINUTES == 8.0

    def test_an_unknown_role_is_treated_as_meaningful(self):
        # Missing a real event costs more than a few extra samples.
        unknown = StatusChange(1, "BOS", "p9", "C", "questionable", "out", NOW, NOW, None)
        assert unknown.is_meaningful


class TestCaptureScheduler:
    def _change(self, player="p1", role=30.0, stamp=NOW):
        return StatusChange(1, "BOS", player, "A", "questionable", "out",
                            stamp, stamp, role)

    def test_a_meaningful_change_raises_the_sampling_rate(self):
        scheduler = CaptureScheduler()
        assert scheduler.register(self._change()) is True
        assert scheduler.interval_for(1, NOW) == EVENT_INTERVAL_SECONDS

    def test_an_untouched_game_stays_at_baseline(self):
        scheduler = CaptureScheduler()
        scheduler.register(self._change())
        assert scheduler.interval_for(99, NOW) == BASELINE_INTERVAL_SECONDS

    def test_re_observing_the_same_report_does_not_retrigger(self):
        # Every restart and every unchanged poll re-observes the same report.
        scheduler = CaptureScheduler()
        assert scheduler.register(self._change()) is True
        assert scheduler.register(self._change()) is False

    def test_a_low_role_change_is_recorded_but_does_not_elevate(self):
        scheduler = CaptureScheduler()
        assert scheduler.register(self._change(player="p2", role=2.0)) is False
        assert scheduler.interval_for(1, NOW) == BASELINE_INTERVAL_SECONDS
        assert len(scheduler.triggered) == 1

    def test_the_window_expires(self):
        scheduler = CaptureScheduler()
        scheduler.register(self._change())
        later = NOW + timedelta(hours=2)
        assert scheduler.interval_for(1, later) == BASELINE_INTERVAL_SECONDS
        assert scheduler.prune(later) == 1

    def test_a_later_change_extends_the_window(self):
        scheduler = CaptureScheduler()
        scheduler.register(self._change())
        second = NOW + timedelta(minutes=20)
        scheduler.register(self._change(player="p3", stamp=second))
        assert scheduler.interval_for(1, second + timedelta(minutes=20)) == (
            EVENT_INTERVAL_SECONDS
        )

    def test_research_targets_are_declared_and_start_at_the_event(self):
        assert EVENT_TARGET_OFFSETS_SECONDS[0] == 0
        assert 10 in EVENT_TARGET_OFFSETS_SECONDS
        assert max(EVENT_TARGET_OFFSETS_SECONDS) == 1800


class TestHealth:
    def _state(self, **overrides) -> CollectorState:
        base = {
            "now_utc": NOW,
            "heartbeat_at_utc": NOW - timedelta(seconds=30),
            "last_market_observation_utc": NOW - timedelta(seconds=30),
            "last_report_observation_utc": NOW - timedelta(minutes=5),
            "upcoming_games": 8,
            "games_with_market_identity": 8,
        }
        return CollectorState(**{**base, **overrides})

    def test_a_healthy_collector_passes(self):
        assert overall_status(assess(self._state())) == "PASS"

    def test_a_stopped_collector_is_critical(self):
        issues = assess(self._state(heartbeat_at_utc=NOW - timedelta(hours=1)))
        assert any(i.code == "collector_stopped" and i.severity == CRITICAL
                   for i in issues)
        assert overall_status(issues) == "FAIL"

    def test_a_stale_market_feed_is_critical_when_games_are_upcoming(self):
        issues = assess(
            self._state(last_market_observation_utc=NOW - timedelta(hours=1))
        )
        assert any(i.code == "market_feed_stale" for i in issues)

    def test_a_stale_feed_is_not_flagged_with_no_games(self):
        issues = assess(self._state(
            last_market_observation_utc=NOW - timedelta(hours=1),
            upcoming_games=0,
        ))
        assert not any(i.code == "market_feed_stale" for i in issues)

    def test_a_blocked_canary_is_critical(self):
        issues = assess(self._state(canary_reachable=False))
        assert any(i.code == "report_source_canary_blocked"
                   and i.severity == CRITICAL for i in issues)

    def test_unwritable_storage_is_critical(self):
        issues = assess(self._state(raw_storage_writable=False))
        assert any(i.code == "raw_storage_unwritable" for i in issues)

    def test_a_game_near_its_anchor_without_a_market_is_critical(self):
        issues = assess(self._state(games_missing_identity_near_anchor=[123]))
        assert any(i.code == "missing_market_identity_near_anchor"
                   and i.severity == CRITICAL for i in issues)

    def test_a_parse_failure_is_a_warning_not_a_stop(self):
        issues = assess(self._state(parse_failures=1))
        assert any(i.code == "report_parse_failures" and i.severity == WARNING
                   for i in issues)
        assert overall_status(issues) == "WARN"

    def test_no_upcoming_games_is_informational(self):
        issues = assess(self._state(upcoming_games=0, games_with_market_identity=0))
        assert any(i.code == "no_upcoming_games" and i.severity == INFO
                   for i in issues)

    def test_issues_are_ordered_most_severe_first(self):
        issues = assess(self._state(
            canary_reachable=False, parse_failures=2, upcoming_games=0
        ))
        severities = [i.severity for i in issues]
        assert severities == sorted(
            severities, key=lambda s: {CRITICAL: 0, WARNING: 1, INFO: 2}[s]
        )

    def test_a_season_fits_on_disk(self):
        estimate = storage_estimate()
        assert estimate["total_gib"] < 20
        assert "never overwritten" in estimate["note"]


class TestTimezoneSafety:
    def test_report_slots_survive_the_dst_transition(self):
        from nba_prediction_market.availability.nba_official import (
            latest_slot_at_or_before,
        )

        eastern = ZoneInfo("America/New_York")
        # US DST began 2026-03-08. Both sides must resolve to a real slot.
        for moment in (
            datetime(2026, 3, 8, 1, 45, tzinfo=eastern),
            datetime(2026, 3, 8, 3, 15, tzinfo=eastern),
        ):
            slot = latest_slot_at_or_before(moment)
            assert slot.report_timestamp_utc <= moment

    def test_a_late_tipoff_crossing_midnight_utc_stays_ordered(self):
        from nba_prediction_market.availability.nba_official import (
            latest_slot_at_or_before,
        )

        # 10:30pm ET is the next day in UTC.
        tipoff = datetime(2026, 11, 4, 3, 30, tzinfo=UTC)
        anchor = tipoff - timedelta(minutes=30)
        assert latest_slot_at_or_before(anchor).report_timestamp_utc <= anchor

    def test_every_stored_timestamp_is_timezone_aware(self):
        timing = _timing()
        for value in (timing.request_started_at_utc, timing.response_received_at_utc,
                      timing.first_observed_at_utc):
            assert value.tzinfo is not None


class TestNoTradingCapability:
    def test_the_capture_package_cannot_place_an_order(self):
        import pkgutil

        import nba_prediction_market.capture as capture

        banned = ("create_order", "place_order", "batch_create", "submit_order",
                  "cancel_order", "portfolio", "kelly")
        for module in pkgutil.iter_modules(capture.__path__):
            source = (
                Path(capture.__path__[0]) / f"{module.name}.py"
            ).read_text().lower()
            for token in banned:
                assert token not in source, f"{module.name} mentions {token}"

    def test_no_write_endpoint_is_referenced(self):
        from nba_prediction_market.capture import kalshi_live

        source = Path(kalshi_live.__file__).read_text()
        assert "/orders" not in source
        assert "POST" not in source
