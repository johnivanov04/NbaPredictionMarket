"""The collector process: locking, wiring, preseason behaviour, and safety."""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from nba_prediction_market.capture.health import (
    CRITICAL,
    INFO,
    WARNING,
    CollectorState,
    assess,
)
from nba_prediction_market.capture.schedule import ScheduledGame, reports_expected
from nba_prediction_market.pipelines.run_capture_collector import (
    PLACES_ORDERS,
    CollectorLock,
    LockHeld,
)

NOW = datetime(2026, 10, 20, 22, 0, tzinfo=UTC)


def _game(gid="g1", *, phase="regular_season", tip=NOW):
    return ScheduledGame(
        source_game_id=gid, season=2026, phase=phase, tipoff_utc=tip,
        home_team="DET", away_team="BOS", source="test",
        first_seen_at_utc=NOW - timedelta(days=1),
    )


class TestProcessLock:
    def test_acquiring_writes_pid_and_heartbeat(self, tmp_path):
        lock = CollectorLock(tmp_path / "collector.lock").acquire(NOW)
        payload = json.loads((tmp_path / "collector.lock").read_text())
        assert payload["pid"] == os.getpid()
        assert payload["heartbeat_at_utc"] == NOW.isoformat()
        lock.release()

    def test_releasing_removes_the_lock(self, tmp_path):
        path = tmp_path / "collector.lock"
        with CollectorLock(path):
            assert path.is_file()
        assert not path.is_file()

    def test_a_live_holder_blocks_a_second_collector(self, tmp_path):
        path = tmp_path / "collector.lock"
        # A pid that is definitely alive but is not us.
        path.write_text(json.dumps({"pid": 1, "heartbeat_at_utc": NOW.isoformat()}))
        with pytest.raises(LockHeld, match="already running"):
            CollectorLock(path).acquire(NOW)

    def test_a_stale_lock_from_a_dead_pid_is_taken_over(self, tmp_path):
        """A crash must not need a human before capture can resume."""
        path = tmp_path / "collector.lock"
        dead = 999_999
        path.write_text(json.dumps({"pid": dead, "heartbeat_at_utc": NOW.isoformat()}))
        lock = CollectorLock(path).acquire(NOW)
        assert json.loads(path.read_text())["pid"] == os.getpid()
        lock.release()

    def test_an_unreadable_lock_is_not_treated_as_held(self, tmp_path):
        path = tmp_path / "collector.lock"
        path.write_text("{ truncated")
        lock = CollectorLock(path).acquire(NOW)
        assert json.loads(path.read_text())["pid"] == os.getpid()
        lock.release()

    def test_heartbeat_advances_but_start_time_does_not(self, tmp_path):
        path = tmp_path / "collector.lock"
        lock = CollectorLock(path).acquire(NOW)
        lock.beat(NOW + timedelta(seconds=90))
        payload = json.loads(path.read_text())
        assert payload["started_at_utc"] == NOW.isoformat()
        assert payload["heartbeat_at_utc"] == (NOW + timedelta(seconds=90)).isoformat()
        lock.release()

    def test_restart_reacquires_cleanly(self, tmp_path):
        path = tmp_path / "collector.lock"
        CollectorLock(path).acquire(NOW).release()
        lock = CollectorLock(path).acquire(NOW + timedelta(minutes=5))
        assert path.is_file()
        lock.release()


class TestNoOrderPath:
    """Research-only, asserted mechanically rather than by docstring."""

    def test_flag_is_false(self):
        assert PLACES_ORDERS is False

    def test_no_module_references_an_order_endpoint(self):
        src = Path(__file__).resolve().parents[2] / "src"
        forbidden = ("/portfolio/orders", "create_order", "place_order", "OrderRequest")
        offenders = []
        for path in src.rglob("*.py"):
            text = path.read_text(encoding="utf-8")
            for needle in forbidden:
                if needle in text:
                    offenders.append(f"{path.name}: {needle}")
        assert offenders == [], f"order path present: {offenders}"

    def test_collector_only_uses_read_endpoints(self):
        from nba_prediction_market.pipelines import run_capture_collector as mod

        text = Path(mod.__file__).read_text(encoding="utf-8")
        assert "client.post" not in text
        assert "client.delete" not in text


class TestPreseasonReportExpectations:
    def test_preseason_only_slate_expects_no_report(self):
        assert reports_expected([_game(phase="preseason")]) is False

    def test_regular_season_slate_expects_a_report(self):
        assert reports_expected([_game(phase="regular_season")]) is True

    def test_one_research_game_is_enough_to_expect_publication(self):
        """Erring lenient here would silence a real alarm on a real slate."""
        mixed = [_game("a", phase="preseason"), _game("b", phase="regular_season")]
        assert reports_expected(mixed) is True

    def _state(self, **kw):
        base = {
            "now_utc": NOW,
            "heartbeat_at_utc": NOW,
            "last_market_observation_utc": NOW,
            "upcoming_games": 4,
            "games_with_market_identity": 4,
        }
        base.update(kw)
        return CollectorState(**base)

    def test_absent_preseason_report_is_informational_not_a_fault(self):
        issues = assess(self._state(
            last_report_observation_utc=None, report_publication_expected=False
        ))
        codes = {i.code: i.severity for i in issues}
        assert codes["report_publication_not_expected"] == INFO
        assert "no_report_observed_yet" not in codes
        assert not [i for i in issues if i.severity == CRITICAL]

    def test_absent_regular_season_report_still_reported(self):
        issues = assess(self._state(last_report_observation_utc=None))
        assert any(i.code == "no_report_observed_yet" for i in issues)

    def test_stale_report_warning_suppressed_only_when_unexpected(self):
        old = NOW - timedelta(hours=20)
        expected = assess(self._state(latest_report_source_timestamp_utc=old))
        assert any(i.code == "availability_report_unusually_old" for i in expected)

        unexpected = assess(self._state(
            latest_report_source_timestamp_utc=old,
            report_publication_expected=False,
        ))
        assert not any(
            i.code == "availability_report_unusually_old" for i in unexpected
        )

    def test_canary_block_is_still_critical_during_preseason(self):
        """Expected absence must never mask a source that is actually blocked."""
        issues = assess(self._state(
            canary_reachable=False, report_publication_expected=False
        ))
        assert any(
            i.code == "report_source_canary_blocked" and i.severity == CRITICAL
            for i in issues
        )


class TestPreseasonMarketExpectations:
    def _state(self, **kw):
        base = {"now_utc": NOW, "heartbeat_at_utc": NOW, "last_report_observation_utc": NOW}
        base.update(kw)
        return CollectorState(**base)

    def test_unlisted_preseason_markets_are_not_critical(self):
        issues = assess(self._state(
            upcoming_games=6, research_games=0,
            games_with_market_identity=0, last_market_observation_utc=None,
        ))
        assert not [i for i in issues if i.severity == CRITICAL]
        assert any(i.code == "markets_not_yet_listed" for i in issues)

    def test_unlisted_regular_season_markets_stay_critical(self):
        issues = assess(self._state(
            upcoming_games=6, research_games=6,
            games_with_market_identity=0, last_market_observation_utc=None,
        ))
        assert any(
            i.code == "no_market_observations" and i.severity == CRITICAL
            for i in issues
        )

    def test_default_research_count_preserves_regular_season_behaviour(self):
        """research_games=None must behave exactly as before it existed."""
        issues = assess(self._state(
            upcoming_games=6, games_with_market_identity=0,
            last_market_observation_utc=None,
        ))
        assert any(i.code == "no_market_observations" for i in issues)

    def test_stale_market_feed_is_not_critical_on_a_preseason_night(self):
        issues = assess(self._state(
            upcoming_games=6, research_games=0, games_with_market_identity=6,
            last_market_observation_utc=NOW - timedelta(hours=1),
        ))
        assert not [i for i in issues if i.severity == CRITICAL]

    def test_stale_market_feed_stays_critical_in_the_regular_season(self):
        issues = assess(self._state(
            upcoming_games=6, research_games=6, games_with_market_identity=6,
            last_market_observation_utc=NOW - timedelta(hours=1),
        ))
        assert any(
            i.code == "market_feed_stale" and i.severity == CRITICAL for i in issues
        )

    def test_low_disk_is_still_a_warning_during_preseason(self):
        issues = assess(self._state(
            upcoming_games=6, research_games=0, games_with_market_identity=6,
            last_market_observation_utc=NOW, disk_free_bytes=2 * 2**30,
        ))
        assert any(
            i.code == "low_disk_space" and i.severity == WARNING for i in issues
        )
