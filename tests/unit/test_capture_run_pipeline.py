"""The prospective capture entry point: pacing, canary, and the anchor rule."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import httpx
import pytest

from nba_prediction_market.pipelines.run_availability_capture import (
    MIN_REQUEST_INTERVAL_SECONDS,
    PacedFetcher,
    canary_is_reachable,
)

NOW = datetime(2026, 1, 15, 18, 0, tzinfo=UTC)


class _Clock:
    def __init__(self) -> None:
        self.t = 0.0
        self.slept: list[float] = []

    def monotonic(self) -> float:
        return self.t

    def sleep(self, seconds: float) -> None:
        self.slept.append(seconds)
        self.t += seconds


def _client(handler) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))


class TestPacing:
    def test_the_first_request_is_not_delayed(self):
        clock = _Clock()
        client = _client(lambda r: httpx.Response(200, content=b"%PDF"))
        fetch = PacedFetcher(client, sleep=clock.sleep, monotonic=clock.monotonic)
        status, body, _ = fetch("https://example.test/a.pdf")
        assert (status, body) == (200, b"%PDF")
        assert clock.slept == []

    def test_back_to_back_requests_are_spaced_out(self):
        # Concurrency gets this CDN to answer 403, which is indistinguishable
        # from "not published", so the floor between requests is load-bearing.
        clock = _Clock()
        client = _client(lambda r: httpx.Response(200, content=b"%PDF"))
        fetch = PacedFetcher(client, sleep=clock.sleep, monotonic=clock.monotonic)
        fetch("https://example.test/a.pdf")
        fetch("https://example.test/b.pdf")
        assert clock.slept == [pytest.approx(MIN_REQUEST_INTERVAL_SECONDS)]

    def test_no_extra_wait_when_enough_time_already_passed(self):
        clock = _Clock()
        client = _client(lambda r: httpx.Response(200, content=b"%PDF"))
        fetch = PacedFetcher(client, sleep=clock.sleep, monotonic=clock.monotonic)
        fetch("https://example.test/a.pdf")
        clock.t += 5.0
        fetch("https://example.test/b.pdf")
        assert clock.slept == []

    def test_a_403_is_returned_not_raised(self):
        # 403 is data, not an error: it is how the CDN says "no such report".
        client = _client(lambda r: httpx.Response(403))
        status, _, _ = PacedFetcher(client)("https://example.test/x.pdf")
        assert status == 403


class TestCanary:
    def test_a_reachable_canary_means_the_run_was_not_blocked(self):
        assert canary_is_reachable(_client(lambda r: httpx.Response(200)))

    def test_a_403_canary_means_the_run_was_blocked(self):
        assert not canary_is_reachable(_client(lambda r: httpx.Response(403)))

    def test_a_transport_failure_is_treated_as_blocked_not_as_proof(self):
        def boom(request):
            raise httpx.ConnectError("refused")

        assert not canary_is_reachable(_client(boom))

    def test_the_canary_targets_a_slot_known_to_exist(self):
        seen: list[str] = []

        def handler(request):
            seen.append(str(request.url))
            return httpx.Response(200)

        canary_is_reachable(_client(handler))
        assert seen and seen[0].endswith(".pdf")
        assert "ak-static.cms.nba.com" in seen[0]


class TestHorizon:
    def test_only_games_inside_the_horizon_are_planned(self, tmp_path, monkeypatch):
        import pandas as pd

        from nba_prediction_market.pipelines import run_availability_capture as mod

        processed = tmp_path / "processed"
        processed.mkdir()
        pd.DataFrame([
            {"nba_game_id": 1, "game_datetime_utc": NOW + timedelta(hours=2)},
            {"nba_game_id": 2, "game_datetime_utc": NOW + timedelta(hours=100)},
            {"nba_game_id": 3, "game_datetime_utc": NOW - timedelta(hours=2)},
        ]).to_parquet(processed / "nba_regular_season_games_2006_26.parquet")

        class Paths:
            def __init__(self):
                self.processed = processed

        class Settings:
            paths = Paths()

        games = mod.upcoming_games(Settings(), now=NOW, horizon_hours=36.0)
        assert [g["game_id"] for g in games] == [1]

    def test_a_missing_schedule_is_a_config_error(self, tmp_path):
        from nba_prediction_market.config import ConfigError
        from nba_prediction_market.pipelines import run_availability_capture as mod

        class Paths:
            processed = tmp_path

        class Settings:
            paths = Paths()

        with pytest.raises(ConfigError, match="Missing"):
            mod.upcoming_games(Settings(), now=NOW, horizon_hours=36.0)
