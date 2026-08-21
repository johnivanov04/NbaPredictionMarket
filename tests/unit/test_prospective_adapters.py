"""Secondary availability feeds are captured forward, never backward."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from nba_prediction_market.availability.prospective_adapters import (
    BallDontLieInjuriesAdapter,
    ProspectiveAdapter,
    SportsDataIoAdapter,
    capture_all,
)
from nba_prediction_market.availability.snapshot_store import SnapshotStore

NOW = datetime(2026, 1, 15, 18, 0, tzinfo=UTC)
PAYLOAD = {"data": [{"player": {"id": 1}, "status": "Out", "description": "knee"}]}


@pytest.fixture
def store(tmp_path: Path) -> SnapshotStore:
    return SnapshotStore(tmp_path / "snapshots")


def _docs(store: SnapshotStore, source: str = "stub_feed") -> list[dict]:
    return store.iter_snapshots(source)


class _Stub(ProspectiveAdapter):
    source_name = "stub_feed"


class TestObservationTime:
    def test_the_observation_time_is_the_fetch_time(self, store):
        outcome = _Stub(store, fetch=lambda: PAYLOAD, now=lambda: NOW).capture()
        assert outcome.status == "captured"
        assert outcome.retrieved_at_utc == NOW
        assert _docs(store)[0]["retrieved_at_utc"] == NOW.isoformat()

    def test_no_as_of_timestamp_is_invented_from_the_payload(self, store):
        # The feed carries no as-of stamp. Leaving these empty is what keeps a
        # consumer from back-dating today's state onto an earlier game.
        payload = dict(PAYLOAD, updated_at="2025-11-01T00:00:00Z")
        _Stub(store, fetch=lambda: payload, now=lambda: NOW).capture()
        document = _docs(store)[0]
        assert document["source_report_timestamp"] is None
        assert document["source_effective_date"] is None

    def test_the_snapshot_is_marked_prospective_only(self, store):
        _Stub(store, fetch=lambda: PAYLOAD, now=lambda: NOW).capture()
        metadata = _docs(store)[0]["metadata"]
        assert metadata["prospective_only"] is True
        assert metadata["precision"] == "no_timestamp"

    def test_a_naive_clock_is_rejected(self, store):
        naive = _Stub(store, fetch=lambda: PAYLOAD, now=lambda: datetime(2026, 1, 15, 18, 0))
        with pytest.raises(ValueError, match="timezone-aware"):
            naive.capture()


class TestFailureHandling:
    def test_a_source_outage_is_reported_not_raised(self, store):
        def boom():
            raise ConnectionError("feed down")

        outcome = _Stub(store, fetch=boom, now=lambda: NOW).capture()
        assert outcome.status == "failed"
        assert "feed down" in outcome.detail
        assert _docs(store) == []

    def test_one_failing_adapter_does_not_stop_the_others(self, store):
        def boom():
            raise ConnectionError("down")

        good = _Stub(store, fetch=lambda: PAYLOAD, now=lambda: NOW)
        bad = _Stub(store, fetch=boom, now=lambda: NOW)
        statuses = [o.status for o in capture_all([bad, good, bad])]
        assert statuses == ["failed", "captured", "failed"]
        assert len(_docs(store)) == 1


class TestRecordCounting:
    def test_a_wrapped_data_list_is_counted(self, store):
        outcome = _Stub(store, fetch=lambda: PAYLOAD, now=lambda: NOW).capture()
        assert outcome.records == 1

    def test_a_bare_list_is_counted(self, store):
        outcome = _Stub(store, fetch=lambda: [1, 2, 3], now=lambda: NOW).capture()
        assert outcome.records == 3


class TestBallDontLieAdapter:
    def test_all_pages_are_gathered_into_one_snapshot(self, store):
        class Client:
            def iter_paid_records(self, path, season, **kwargs):
                assert path == "/v1/player_injuries"
                assert season == 2025
                yield [{"player": {"id": 1}}]
                yield [{"player": {"id": 2}}]

        adapter = BallDontLieInjuriesAdapter.from_client(
            store, Client(), season=2025, now=lambda: NOW
        )
        outcome = adapter.capture()
        assert outcome.records == 2
        assert _docs(store, "balldontlie_player_injuries")[0]["source"] == (
            "balldontlie_player_injuries"
        )

    def test_captures_accumulate_rather_than_overwrite(self, store):
        later = datetime(2026, 1, 15, 19, 0, tzinfo=UTC)
        _Stub(store, fetch=lambda: PAYLOAD, now=lambda: NOW).capture()
        _Stub(store, fetch=lambda: {"data": [{"x": 1}]}, now=lambda: later).capture()
        assert len(_docs(store)) == 2


class TestSportsDataIoPlaceholder:
    def test_it_is_inert_until_someone_verifies_the_contract(self, store):
        outcome = SportsDataIoAdapter(store).capture()
        assert outcome.status == "disabled"
        assert "verified response shape" in outcome.detail
        assert _docs(store, "sportsdataio_injuries") == []

    def test_it_writes_nothing_even_when_handed_a_fetcher(self, store):
        outcome = SportsDataIoAdapter(store, fetch=lambda: PAYLOAD).capture()
        assert outcome.status == "disabled"
        assert _docs(store, "sportsdataio_injuries") == []
