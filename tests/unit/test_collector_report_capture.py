"""The collector's report-capture tick, against the real CaptureResult shape.

This file exists because of a production crash on the first tick of the
2026-27 preseason soak: the collector read ``CaptureResult.status``, which has
never existed -- the field is ``outcome``. No test caught it because every
earlier smoke run had no game inside the capture horizon, so no capture was
ever *due* and the code path was unreachable.

Every test here therefore drives ``_capture_reports`` with a runner that
returns genuine ``CaptureResult`` objects built by the real ``AvailabilityRunner``.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import httpx

from nba_prediction_market.availability.capture_schedule import plan_captures
from nba_prediction_market.availability.nba_official import ReportArchive
from nba_prediction_market.availability.runner import (
    AvailabilityRunner,
    CaptureResult,
)
from nba_prediction_market.availability.snapshot_store import SnapshotStore
from nba_prediction_market.capture.schedule import ScheduledGame
from nba_prediction_market.pipelines import run_capture_collector as mod

TIP = datetime(2026, 10, 3, 23, 0, tzinfo=UTC)
NOW = TIP - timedelta(minutes=45)


def _settings(tmp_path):
    class Paths:
        root = tmp_path

        @property
        def processed(self):
            return tmp_path / "processed"

        @property
        def reports(self):
            return tmp_path / "reports"

        def ensure(self):
            for p in (self.processed, self.reports):
                p.mkdir(parents=True, exist_ok=True)

    class Settings:
        paths = Paths()

    Settings.paths.ensure()
    return Settings()


def _game(gid="g1", *, phase="preseason", tip=TIP):
    return ScheduledGame(
        source_game_id=gid, season=2026, phase=phase, tipoff_utc=tip,
        home_team="DET", away_team="BOS", source="test",
        first_seen_at_utc=TIP - timedelta(days=7),
    )


def _collector(tmp_path, *, responses):
    """A collector whose runner answers with the given HTTP responses."""
    collector = mod.Collector(_settings(tmp_path), capture_referees=False)
    calls: list[str] = []

    def fetch(url: str):
        calls.append(url)
        result = responses.pop(0) if isinstance(responses, list) else responses
        if isinstance(result, Exception):
            raise result
        return result

    def make_runner(archive, store, **_kwargs):
        return AvailabilityRunner(archive, store, fetch=fetch, sleep=lambda _s: None)

    return collector, calls, make_runner


class TestCaptureResultContract:
    def test_capture_result_has_outcome_and_not_status(self):
        """Pins the field the collector must read."""
        fields = set(CaptureResult.__dataclass_fields__)
        assert "outcome" in fields
        assert "status" not in fields
        assert not hasattr(CaptureResult, "status"), (
            "a status property must not be added just to satisfy a caller"
        )

    def test_collector_source_reads_outcome(self):
        from pathlib import Path

        text = Path(mod.__file__).read_text(encoding="utf-8")
        assert "r.status ==" not in text
        assert 'r.outcome == "captured"' in text
        assert 'r.outcome == "source_unavailable"' in text


class TestPreseason403IsNonFatal:
    """The exact production failure: 403 on every preseason slot."""

    def test_tick_does_not_raise_attribute_error(self, tmp_path, monkeypatch):
        collector, _calls, make_runner = _collector(tmp_path, responses=(403, b"", {}))
        monkeypatch.setattr(mod, "AvailabilityRunner", make_runner)
        monkeypatch.setattr(mod, "canary_is_reachable", lambda _c: True)
        with httpx.Client() as client:
            collector._capture_reports(client, [_game()], NOW)  # must not raise

    def test_an_unavailable_report_is_not_counted_as_captured(
        self, tmp_path, monkeypatch
    ):
        collector, _calls, make_runner = _collector(tmp_path, responses=(403, b"", {}))
        monkeypatch.setattr(mod, "AvailabilityRunner", make_runner)
        monkeypatch.setattr(mod, "canary_is_reachable", lambda _c: True)
        with httpx.Client() as client:
            collector._capture_reports(client, [_game()], NOW)
        assert collector.report_captures == 0
        assert collector.last_report_observation is None

    def test_a_403_is_not_counted_as_a_fetch_failure(self, tmp_path, monkeypatch):
        """403 means 'not published', which is the normal preseason answer."""
        collector, _calls, make_runner = _collector(tmp_path, responses=(403, b"", {}))
        monkeypatch.setattr(mod, "AvailabilityRunner", make_runner)
        monkeypatch.setattr(mod, "canary_is_reachable", lambda _c: True)
        with httpx.Client() as client:
            collector._capture_reports(client, [_game()], NOW)
        assert collector.failed_fetches == 0

    def test_the_canary_is_consulted_when_a_slot_is_unavailable(
        self, tmp_path, monkeypatch
    ):
        asked = []
        collector, _calls, make_runner = _collector(tmp_path, responses=(403, b"", {}))
        monkeypatch.setattr(mod, "AvailabilityRunner", make_runner)
        monkeypatch.setattr(
            mod, "canary_is_reachable", lambda _c: asked.append(1) or True
        )
        with httpx.Client() as client:
            collector._capture_reports(client, [_game()], NOW)
        assert asked, "an unavailable slot must trigger the blocked-source check"
        assert collector.canary_ok is True

    def test_a_blocked_canary_is_recorded(self, tmp_path, monkeypatch):
        collector, _calls, make_runner = _collector(tmp_path, responses=(403, b"", {}))
        monkeypatch.setattr(mod, "AvailabilityRunner", make_runner)
        monkeypatch.setattr(mod, "canary_is_reachable", lambda _c: False)
        with httpx.Client() as client:
            collector._capture_reports(client, [_game()], NOW)
        assert collector.canary_ok is False


class TestTransientFailureIsNotSuccess:
    def test_a_raising_fetch_is_not_a_capture(self, tmp_path, monkeypatch):
        collector, _calls, make_runner = _collector(
            tmp_path, responses=httpx.ConnectError("boom")
        )
        monkeypatch.setattr(mod, "AvailabilityRunner", make_runner)
        monkeypatch.setattr(mod, "canary_is_reachable", lambda _c: True)
        with httpx.Client() as client:
            collector._capture_reports(client, [_game()], NOW)
        assert collector.report_captures == 0
        assert collector.last_report_observation is None
        assert collector.failed_fetches > 0

    def test_an_unexpected_status_is_a_failure_not_a_capture(
        self, tmp_path, monkeypatch
    ):
        collector, _calls, make_runner = _collector(tmp_path, responses=(500, b"", {}))
        monkeypatch.setattr(mod, "AvailabilityRunner", make_runner)
        monkeypatch.setattr(mod, "canary_is_reachable", lambda _c: True)
        with httpx.Client() as client:
            collector._capture_reports(client, [_game()], NOW)
        assert collector.report_captures == 0
        assert collector.failed_fetches > 0


class TestSlotDeduplication:
    """One report covers many games; it must be requested once per run."""

    def test_a_games_nine_offsets_resolve_to_one_slot(self):
        tasks = plan_captures(
            [{"game_id": "g1", "scheduled_tipoff_utc": TIP}],
            ["nba_official_injury_report"],
        )
        anchors = {t.anchor_utc for t in tasks}
        assert len(tasks) > 1
        assert len(anchors) == 1, "the anchor is one instant per game"

    def test_an_unavailable_slot_is_fetched_once_not_once_per_task(self, tmp_path):
        runner = AvailabilityRunner(
            ReportArchive(tmp_path / "a"), SnapshotStore(tmp_path / "s"),
            fetch=lambda url: (403, b"", {}), sleep=lambda _s: None,
        )
        calls: list[str] = []
        runner._fetch = lambda url: (calls.append(url), (403, b"", {}))[1]
        tasks = plan_captures(
            [{"game_id": "g1", "scheduled_tipoff_utc": TIP}],
            ["nba_official_injury_report"],
        )
        results = runner.run(tasks, now=NOW)
        assert len(set(calls)) == 1
        assert len(calls) == 1, f"refetched the same slot {len(calls)} times"
        # Every task still gets its own result, so anchor_health stays correct.
        assert len(results) == len(tasks)
        assert {r.outcome for r in results} == {"source_unavailable"}

    def test_dedup_keeps_stats_consistent_with_results(self, tmp_path):
        runner = AvailabilityRunner(
            ReportArchive(tmp_path / "a"), SnapshotStore(tmp_path / "s"),
            fetch=lambda url: (403, b"", {}), sleep=lambda _s: None,
        )
        tasks = plan_captures(
            [{"game_id": "g1", "scheduled_tipoff_utc": TIP}],
            ["nba_official_injury_report"],
        )
        results = runner.run(tasks, now=NOW)
        assert runner.stats.unavailable == len(results)
        assert runner.stats.deduplicated == len(results) - 1
        assert runner.stats.attempted == len(results)

    def test_a_captured_slot_is_also_fetched_only_once(self, tmp_path):
        calls: list[str] = []
        runner = AvailabilityRunner(
            ReportArchive(tmp_path / "a"), SnapshotStore(tmp_path / "s"),
            fetch=lambda url: (calls.append(url), (200, b"%PDF", {}))[1],
            sleep=lambda _s: None,
        )
        tasks = plan_captures(
            [{"game_id": "g1", "scheduled_tipoff_utc": TIP}],
            ["nba_official_injury_report"],
        )
        results = runner.run(tasks, now=NOW)
        assert len(calls) == 1
        outcomes = [r.outcome for r in results]
        assert outcomes.count("captured") == 1
        assert outcomes.count("already_present") == len(results) - 1

    def test_distinct_slots_are_still_fetched_separately(self, tmp_path):
        """Dedup must not collapse genuinely different reports."""
        calls: list[str] = []
        runner = AvailabilityRunner(
            ReportArchive(tmp_path / "a"), SnapshotStore(tmp_path / "s"),
            fetch=lambda url: (calls.append(url), (403, b"", {}))[1],
            sleep=lambda _s: None,
        )
        tasks = plan_captures(
            [
                {"game_id": "g1", "scheduled_tipoff_utc": TIP},
                {"game_id": "g2", "scheduled_tipoff_utc": TIP + timedelta(hours=5)},
            ],
            ["nba_official_injury_report"],
        )
        runner.run(tasks, now=NOW)
        assert len(set(calls)) == 2, "two different slots, two fetches"

    def test_a_fresh_runner_retries_a_previously_unavailable_slot(self, tmp_path):
        """The memo is per run, so the next tick tries again."""
        archive = ReportArchive(tmp_path / "a")
        store = SnapshotStore(tmp_path / "s")
        tasks = plan_captures(
            [{"game_id": "g1", "scheduled_tipoff_utc": TIP}],
            ["nba_official_injury_report"],
        )
        first_calls: list[str] = []
        AvailabilityRunner(
            archive, store,
            fetch=lambda u: (first_calls.append(u), (403, b"", {}))[1],
            sleep=lambda _s: None,
        ).run(tasks, now=NOW)

        second_calls: list[str] = []
        results = AvailabilityRunner(
            archive, store,
            fetch=lambda u: (second_calls.append(u), (200, b"%PDF", {}))[1],
            sleep=lambda _s: None,
        ).run(tasks, now=NOW)
        assert len(first_calls) == 1 and len(second_calls) == 1
        assert "captured" in {r.outcome for r in results}


class TestNoCaptureDue:
    def test_a_game_outside_the_horizon_plans_nothing(self, tmp_path, monkeypatch):
        """The condition that hid this bug through every earlier smoke test."""
        collector, calls, make_runner = _collector(tmp_path, responses=(200, b"", {}))
        monkeypatch.setattr(mod, "AvailabilityRunner", make_runner)
        far = TIP + timedelta(days=40)
        with httpx.Client() as client:
            collector._capture_reports(client, [_game(tip=far)], NOW)
        assert calls == []
        assert collector.report_captures == 0

    def test_an_empty_slate_is_a_no_op(self, tmp_path, monkeypatch):
        collector, calls, make_runner = _collector(tmp_path, responses=(200, b"", {}))
        monkeypatch.setattr(mod, "AvailabilityRunner", make_runner)
        with httpx.Client() as client:
            collector._capture_reports(client, [], NOW)
        assert calls == []


class TestAPlannedCaptureIsAttemptedOnce:
    """Across ticks, not just within one.

    The live preseason run issued 162 identical report requests in 90 seconds:
    27 ticks x 6 slots. ``due`` meant "its scheduled moment has passed", which
    stays true for the rest of the slate, so every task re-ran every second.
    """

    def test_a_second_tick_does_not_refetch_the_same_slot(
        self, tmp_path, monkeypatch
    ):
        collector, calls, make_runner = _collector(tmp_path, responses=(403, b"", {}))
        monkeypatch.setattr(mod, "AvailabilityRunner", make_runner)
        monkeypatch.setattr(mod, "canary_is_reachable", lambda _c: True)
        games = [_game()]
        with httpx.Client() as client:
            collector._capture_reports(client, games, NOW)
            first = len(calls)
            collector._capture_reports(client, games, NOW + timedelta(seconds=1))
            collector._capture_reports(client, games, NOW + timedelta(seconds=2))
        assert first > 0, "the first tick must actually attempt the capture"
        assert len(calls) == first, (
            f"later ticks refetched: {len(calls)} calls vs {first} after one tick"
        )

    def test_one_fetch_per_slot_for_a_whole_game(self, tmp_path, monkeypatch):
        collector, calls, make_runner = _collector(tmp_path, responses=(403, b"", {}))
        monkeypatch.setattr(mod, "AvailabilityRunner", make_runner)
        monkeypatch.setattr(mod, "canary_is_reachable", lambda _c: True)
        with httpx.Client() as client:
            for second in range(10):
                collector._capture_reports(
                    client, [_game()], NOW + timedelta(seconds=second)
                )
        assert len(calls) == 1, f"expected one fetch, made {len(calls)}"

    def test_a_later_planned_offset_is_still_attempted(self, tmp_path, monkeypatch):
        """The offsets are the retry schedule; they must not be suppressed."""
        collector, _calls, make_runner = _collector(tmp_path, responses=(403, b"", {}))
        monkeypatch.setattr(mod, "AvailabilityRunner", make_runner)
        monkeypatch.setattr(mod, "canary_is_reachable", lambda _c: True)
        early = TIP - timedelta(hours=7)   # only the 24h offset is due
        late = TIP - timedelta(minutes=40)  # several more offsets now due
        with httpx.Client() as client:
            collector._capture_reports(client, [_game()], early)
            after_early = len(collector.attempted_captures)
            collector._capture_reports(client, [_game()], late)
        assert after_early >= 1
        assert len(collector.attempted_captures) > after_early, (
            "offsets that became due later must still be attempted"
        )

    def test_a_new_game_is_not_suppressed_by_an_earlier_one(
        self, tmp_path, monkeypatch
    ):
        collector, _calls, make_runner = _collector(tmp_path, responses=(403, b"", {}))
        monkeypatch.setattr(mod, "AvailabilityRunner", make_runner)
        monkeypatch.setattr(mod, "canary_is_reachable", lambda _c: True)
        with httpx.Client() as client:
            collector._capture_reports(client, [_game("g1")], NOW)
            before = len(collector.attempted_captures)
            collector._capture_reports(
                client, [_game("g1"), _game("g2", tip=TIP + timedelta(hours=5))], NOW
            )
        assert len(collector.attempted_captures) > before

    def test_a_captured_report_is_still_recognised_once(self, tmp_path, monkeypatch):
        collector, calls, make_runner = _collector(
            tmp_path, responses=(200, b"%PDF", {})
        )
        monkeypatch.setattr(mod, "AvailabilityRunner", make_runner)
        monkeypatch.setattr(mod, "canary_is_reachable", lambda _c: True)
        with httpx.Client() as client:
            collector._capture_reports(client, [_game()], NOW)
        assert collector.report_captures == 1
        assert collector.last_report_observation == NOW
        assert len(calls) == 1
