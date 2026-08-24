"""Historical backfill: pacing, restart-safety, and throttle detection."""

from __future__ import annotations

from datetime import date

import httpx
import pytest

from nba_prediction_market.availability.nba_official import ReportArchive, slots_for_date
from nba_prediction_market.pipelines.build_availability_backfill import (
    CANARY_AFTER_CONSECUTIVE_MISSES,
    MIN_REQUEST_INTERVAL_SECONDS,
    BackfillRunner,
    ThrottleDetected,
)

PDF = b"%PDF-1.4 body"
LEGACY_DAY = date(2025, 1, 15)


class _Clock:
    def __init__(self) -> None:
        self.t = 0.0
        self.slept: list[float] = []

    def monotonic(self) -> float:
        return self.t

    def sleep(self, seconds: float) -> None:
        self.slept.append(seconds)
        self.t += seconds


def _runner(tmp_path, handler, clock=None):
    clock = clock or _Clock()
    client = httpx.Client(transport=httpx.MockTransport(handler))
    return BackfillRunner(
        ReportArchive(tmp_path), client,
        sleep=clock.sleep, monotonic=clock.monotonic,
    ), clock


class TestPacing:
    def test_requests_are_spaced_by_the_floor(self, tmp_path):
        runner, clock = _runner(tmp_path, lambda r: httpx.Response(200, content=PDF))
        slots = slots_for_date(LEGACY_DAY)[:3]
        for slot in slots:
            runner.fetch_slot(slot)
        assert clock.slept
        assert all(s == pytest.approx(MIN_REQUEST_INTERVAL_SECONDS) for s in clock.slept)


class TestRestartSafety:
    def test_an_archived_slot_is_never_refetched(self, tmp_path):
        calls: list[str] = []

        def handler(request):
            calls.append(str(request.url))
            return httpx.Response(200, content=PDF)

        runner, _ = _runner(tmp_path, handler)
        slot = slots_for_date(LEGACY_DAY)[0]
        assert runner.fetch_slot(slot) == "archived"
        assert runner.fetch_slot(slot) == "already_present"
        assert len(calls) == 1
        assert runner.stats.already_present == 1

    def test_a_restarted_runner_skips_what_is_already_held(self, tmp_path):
        runner, _ = _runner(tmp_path, lambda r: httpx.Response(200, content=PDF))
        slot = slots_for_date(LEGACY_DAY)[0]
        runner.fetch_slot(slot)

        restarted, _ = _runner(tmp_path, lambda r: httpx.Response(200, content=PDF))
        assert restarted.fetch_slot(slot) == "already_present"
        assert restarted.stats.archived == 0


class TestMissingReports:
    def test_a_403_is_recorded_as_unavailable_not_an_error(self, tmp_path):
        # 403 is how the CDN says "never published"; it is data, not a fault.
        runner, _ = _runner(tmp_path, lambda r: httpx.Response(403))
        assert runner.fetch_slot(slots_for_date(LEGACY_DAY)[0]) == "unavailable"
        assert runner.stats.unavailable == 1
        assert runner.stats.errors == 0

    def test_a_non_pdf_body_is_an_error_not_an_archive_entry(self, tmp_path):
        runner, _ = _runner(tmp_path, lambda r: httpx.Response(200, content=b"<html>"))
        assert runner.fetch_slot(slots_for_date(LEGACY_DAY)[0]) == "error"
        assert runner.stats.archived == 0


class TestThrottleDetection:
    def _handler(self, canary_status):
        def handler(request):
            if "2026-04-10_04_00PM" in str(request.url):
                return httpx.Response(canary_status)
            return httpx.Response(403)

        return handler

    def test_a_run_of_403s_with_a_live_canary_is_accepted_as_real(self, tmp_path):
        # Genuine non-publication: many 403s, but the canary still answers.
        runner, _ = _runner(tmp_path, self._handler(200))
        for slot in slots_for_date(LEGACY_DAY)[: CANARY_AFTER_CONSECUTIVE_MISSES + 2]:
            assert runner.fetch_slot(slot) == "unavailable"
        assert runner.stats.canary_checks >= 1

    def test_a_run_of_403s_with_a_dead_canary_stops_the_run(self, tmp_path):
        # Throttling looks identical in the 403s alone, so the canary is the
        # only thing separating "not published" from "we were blocked".
        runner, _ = _runner(tmp_path, self._handler(403))
        with pytest.raises(ThrottleDetected, match="not evidence"):
            for slot in slots_for_date(LEGACY_DAY):
                runner.fetch_slot(slot)

    def test_a_dead_canary_before_the_run_refuses_to_start(self, tmp_path):
        runner, _ = _runner(tmp_path, lambda r: httpx.Response(403))
        with pytest.raises(ThrottleDetected, match="before the run started"):
            runner.run_range(LEGACY_DAY, LEGACY_DAY)

    def test_a_success_resets_the_miss_streak(self, tmp_path):
        state = {"n": 0}

        def handler(request):
            if "2026-04-10_04_00PM" in str(request.url):
                return httpx.Response(200)
            state["n"] += 1
            return httpx.Response(200, content=PDF) if state["n"] % 3 == 0 else httpx.Response(403)

        runner, _ = _runner(tmp_path, handler)
        for slot in slots_for_date(LEGACY_DAY):
            runner.fetch_slot(slot)
        # Interleaved successes mean the streak never reaches the threshold.
        assert runner.stats.canary_checks == 0
        assert runner.stats.archived > 0


class TestConventionCoverage:
    def test_the_runner_requests_whichever_convention_the_date_uses(self, tmp_path):
        seen: list[str] = []

        def handler(request):
            seen.append(str(request.url))
            return httpx.Response(200, content=PDF)

        runner, _ = _runner(tmp_path, handler)
        runner.fetch_slot(slots_for_date(date(2025, 1, 15))[10])
        runner.fetch_slot(slots_for_date(date(2026, 1, 15))[10])
        assert any(u.endswith("AM.pdf") or u.endswith("PM.pdf") for u in seen)
        legacy, modern = seen[0], seen[1]
        assert "_05AM.pdf" in legacy or legacy.count("_") == 2
        assert modern.count("_") == 3
