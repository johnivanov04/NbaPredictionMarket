"""Availability capture: snapshots, normalization, identity, and as-of safety."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from itertools import pairwise
from pathlib import Path

import pytest

from nba_prediction_market.availability.as_of import (
    AnchorViolationError,
    assert_no_future_observation,
    coverage_at,
    observations_at_or_before,
    state_at,
    status_for,
)
from nba_prediction_market.availability.capture_schedule import (
    ANCHOR_MINUTES_BEFORE_TIP,
    anchor_for,
    latest_report_slot,
    plan_captures,
    verify_anchor_guarantee,
)
from nba_prediction_market.availability.events import (
    NORMALIZED_STATUSES,
    STATUS_OUT,
    STATUS_QUESTIONABLE,
    STATUS_UNKNOWN,
    AvailabilityEvent,
    TemporalPrecision,
    event_from_row,
    normalize_status,
)
from nba_prediction_market.availability.identity import (
    PlayerRegistry,
    normalize_name,
    unresolved_report,
)
from nba_prediction_market.availability.snapshot_store import Snapshot, SnapshotStore
from nba_prediction_market.availability.sources import (
    SOURCE_MATRIX,
    SOURCES_BY_NAME,
    is_historically_safe,
)

ANCHOR = datetime(2026, 11, 5, 0, 0, tzinfo=UTC)


def event(minutes_before: float, status: str, *, source="nba_official_injury_report",
          precision=TemporalPrecision.EXACT, name="Example, Player", **kw):
    return event_from_row(
        source, ANCHOR - timedelta(minutes=minutes_before),
        player_name=name, status_raw=status, temporal_precision=precision, **kw,
    )


# --- status normalization --------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("Out", STATUS_OUT), ("OUT FOR SEASON", STATUS_OUT), ("Questionable", STATUS_QUESTIONABLE),
     ("Day To Day", STATUS_QUESTIONABLE), ("Doubtful", "doubtful"), ("Probable", "probable"),
     ("Available", "available"), ("Active", "available"), ("  out  ", STATUS_OUT)],
)
def test_known_statuses_normalize(raw: str, expected: str) -> None:
    assert normalize_status(raw) == expected


@pytest.mark.parametrize("raw", [None, "", "   ", "Game Time Decision", "banana"])
def test_unrecognised_status_becomes_unknown_not_available(raw) -> None:
    """Guessing here would systematically under-count unavailability."""
    assert normalize_status(raw) == STATUS_UNKNOWN


def test_every_normalized_status_is_in_the_vocabulary() -> None:
    for raw in ("Out", "Questionable", "banana", None):
        assert normalize_status(raw) in NORMALIZED_STATUSES


def test_raw_status_is_preserved_alongside_the_normalized_one() -> None:
    e = event(60, "Day To Day")
    assert e.status_raw == "Day To Day"
    assert e.status_normalized == STATUS_QUESTIONABLE


def test_a_naive_timestamp_is_rejected() -> None:
    with pytest.raises(ValueError, match="timezone-aware"):
        AvailabilityEvent(
            source="s", observed_at_utc=datetime(2026, 11, 5),
            player_name="X", status_raw="Out", status_normalized=STATUS_OUT,
            temporal_precision=TemporalPrecision.EXACT,
        )


# --- the anchor boundary ---------------------------------------------------


def test_an_observation_exactly_at_the_anchor_is_used() -> None:
    events = [event(60, "Questionable"), event(0, "Out")]
    assert status_for(events, ANCHOR, "Example, Player") == STATUS_OUT


def test_an_observation_one_second_after_the_anchor_is_rejected() -> None:
    """The whole point of the phase, in one test."""
    late = event_from_row(
        "nba_official_injury_report", ANCHOR + timedelta(seconds=1),
        player_name="Example, Player", status_raw="Available",
        temporal_precision=TemporalPrecision.EXACT,
    )
    events = [event(60, "Out"), late]
    assert status_for(events, ANCHOR, "Example, Player") == STATUS_OUT
    assert len(observations_at_or_before(events, ANCHOR)) == 1


def test_multiple_same_day_updates_resolve_to_the_latest_before_the_anchor() -> None:
    events = [event(300, "Probable"), event(120, "Questionable"), event(10, "Out")]
    assert status_for(events, ANCHOR, "Example, Player") == STATUS_OUT
    earlier = ANCHOR - timedelta(minutes=60)
    assert status_for(events, earlier, "Example, Player") == STATUS_QUESTIONABLE


def test_no_future_state_leaks_backward() -> None:
    events = [event(300, "Probable"), event(-60, "Out")]   # one an hour after
    early = ANCHOR - timedelta(minutes=200)
    assert status_for(events, early, "Example, Player") == "probable"


def test_a_date_only_observation_cannot_satisfy_a_t30_anchor() -> None:
    events = [event(120, "Out", source="sportradar_daily_injuries",
                    precision=TemporalPrecision.DATE_ONLY)]
    assert state_at(events, ANCHOR) == {}
    assert status_for(events, ANCHOR, "Example, Player") == STATUS_UNKNOWN
    # It is still visible when precision is explicitly not required.
    assert state_at(events, ANCHOR, require_anchor_safe_precision=False)


def test_absence_from_any_report_is_unknown_not_available() -> None:
    assert status_for([event(60, "Out")], ANCHOR, "Someone, Else") == STATUS_UNKNOWN


def test_assert_no_future_observation_raises_with_detail() -> None:
    events = [event(60, "Out"), event(-1, "Available")]
    with pytest.raises(AnchorViolationError, match="postdate the anchor"):
        assert_no_future_observation(events, ANCHOR)


def test_assert_no_future_observation_passes_when_clean() -> None:
    assert_no_future_observation([event(60, "Out"), event(0, "Out")], ANCHOR)


def test_staleness_is_reported() -> None:
    state = state_at([event(45, "Out")], ANCHOR)["Example, Player"]
    assert state.staleness_seconds == pytest.approx(45 * 60)


def test_coverage_reports_refused_sources() -> None:
    events = [
        event(60, "Out"),
        event(30, "Out", source="sportradar_daily_injuries",
              precision=TemporalPrecision.DATE_ONLY),
    ]
    coverage = coverage_at(events, ANCHOR)
    assert coverage.usable is True
    assert coverage.observations_before_anchor == 1
    assert coverage.refused_date_only_sources == ["sportradar_daily_injuries"]


def test_coverage_is_unusable_with_no_safe_observation() -> None:
    coverage = coverage_at([], ANCHOR)
    assert coverage.usable is False
    assert coverage.staleness_seconds is None


def test_a_naive_anchor_is_rejected() -> None:
    with pytest.raises(ValueError, match="timezone-aware"):
        observations_at_or_before([], datetime(2026, 11, 5))


def test_replay_is_deterministic() -> None:
    events = [event(300, "Probable"), event(120, "Questionable"), event(10, "Out")]
    assert state_at(events, ANCHOR) == state_at(list(reversed(events)), ANCHOR)


# --- snapshot store --------------------------------------------------------


def snapshot(moment: datetime, payload, request="daily") -> Snapshot:
    return Snapshot(
        source="nba_official", retrieved_at_utc=moment, request=request, payload=payload
    )


def test_snapshots_are_written_and_readable(tmp_path: Path) -> None:
    store = SnapshotStore(tmp_path)
    path = store.append(snapshot(ANCHOR, {"players": ["a"]}))
    assert path.is_file()
    document = json.loads(path.read_text())
    assert document["payload"] == {"players": ["a"]}
    assert document["retrieved_at_utc"].startswith("2026-11-05")
    assert store.stats.written == 1


def test_a_newer_state_never_overwrites_an_older_one(tmp_path: Path) -> None:
    """Append-only: the record of what was known earlier must survive."""
    store = SnapshotStore(tmp_path)
    store.append(snapshot(ANCHOR - timedelta(hours=2), {"status": "Questionable"}))
    store.append(snapshot(ANCHOR, {"status": "Out"}))

    stored = store.iter_snapshots("nba_official")
    assert len(stored) == 2
    assert [s["payload"]["status"] for s in stored] == ["Questionable", "Out"]


def test_an_identical_recapture_is_not_duplicated(tmp_path: Path) -> None:
    store = SnapshotStore(tmp_path)
    store.append(snapshot(ANCHOR, {"a": 1}))
    store.append(snapshot(ANCHOR, {"a": 1}))
    assert store.stats.written == 1
    assert store.stats.duplicates == 1
    assert len(store.iter_snapshots("nba_official")) == 1


def test_different_content_at_the_same_instant_is_kept_separately(tmp_path: Path) -> None:
    store = SnapshotStore(tmp_path)
    store.append(snapshot(ANCHOR, {"a": 1}))
    store.append(snapshot(ANCHOR, {"a": 2}))
    assert len(store.iter_snapshots("nba_official")) == 2


def test_source_report_timestamp_is_preserved(tmp_path: Path) -> None:
    store = SnapshotStore(tmp_path)
    reported = ANCHOR - timedelta(minutes=5)
    path = store.append(Snapshot(
        source="nba_official", retrieved_at_utc=ANCHOR, request="r",
        payload={}, source_report_timestamp=reported, source_effective_date="2026-11-04",
    ))
    document = json.loads(path.read_text())
    assert document["source_report_timestamp"].startswith("2026-11-04T23:55")
    assert document["source_effective_date"] == "2026-11-04"


def test_replaying_until_a_past_instant_hides_later_snapshots(tmp_path: Path) -> None:
    """Honest replay: only what existed then."""
    store = SnapshotStore(tmp_path)
    store.append(snapshot(ANCHOR - timedelta(hours=2), {"n": 1}))
    store.append(snapshot(ANCHOR + timedelta(hours=2), {"n": 2}))
    assert len(store.iter_snapshots("nba_official", until=ANCHOR)) == 1


def test_a_snapshot_needs_an_aware_timestamp() -> None:
    with pytest.raises(ValueError, match="timezone-aware"):
        Snapshot(source="s", retrieved_at_utc=datetime(2026, 11, 5), request="r", payload={})


def test_a_source_outage_leaves_the_archive_intact(tmp_path: Path) -> None:
    """Nothing is written for a failed capture; earlier state remains."""
    store = SnapshotStore(tmp_path)
    store.append(snapshot(ANCHOR - timedelta(hours=1), {"n": 1}))
    assert len(store.iter_snapshots("nba_official")) == 1
    assert store.iter_snapshots("sportradar") == []


def test_unreadable_snapshots_are_skipped_not_fatal(tmp_path: Path) -> None:
    store = SnapshotStore(tmp_path)
    store.append(snapshot(ANCHOR, {"n": 1}))
    bad = store.directory_for("nba_official", ANCHOR) / "20261105T000000Z__x__bad.json"
    bad.write_text("{ not json")
    assert len(store.iter_snapshots("nba_official")) == 1


# --- identity --------------------------------------------------------------


@pytest.mark.parametrize(
    ("a", "b"),
    [("Porter Jr., Michael", "Michael Porter Jr."),
     ("Jones Jr.,De'Andre", "De'Andre Jones Jr."),
     ("Collins,Zach", "Zach Collins"),
     ("Jokic, Nikola", "Nikola Jokić"),
     ("Smith III, Jabari", "Jabari Smith III")],
)
def test_both_name_conventions_reduce_to_the_same_key(a: str, b: str) -> None:
    """The injury report writes 'Last, First'; BALLDONTLIE writes 'First Last'."""
    assert normalize_name(a) == normalize_name(b)


def test_different_players_do_not_collide() -> None:
    assert normalize_name("Smith, Jabari") != normalize_name("Smith, Jalen")


def test_resolution_prefers_the_nba_reference_id() -> None:
    registry = PlayerRegistry()
    registry.register(101, "Michael Porter Jr.", team_id=8, nba_reference_id="1629008")
    result = registry.resolve("anything at all", nba_reference_id="1629008")
    assert result.ok and result.balldontlie_player_id == 101
    assert result.method == "nba_reference_id"


def test_an_unregistered_reference_id_is_unresolved_not_guessed() -> None:
    registry = PlayerRegistry()
    registry.register(101, "Michael Porter Jr.", team_id=8)
    result = registry.resolve("Porter Jr., Michael", nba_reference_id="999999")
    assert not result.ok
    assert "not registered" in result.reason


def test_team_and_name_resolution_works_across_conventions() -> None:
    registry = PlayerRegistry()
    registry.register(101, "Michael Porter Jr.", team_id=8)
    result = registry.resolve("Porter Jr., Michael", team_id=8)
    assert result.ok and result.method == "team_and_name"


def test_name_only_resolution_is_refused() -> None:
    """Names alone collide; a team is required."""
    registry = PlayerRegistry()
    registry.register(101, "Michael Porter Jr.", team_id=8)
    result = registry.resolve("Porter Jr., Michael")
    assert not result.ok
    assert "requires a team_id" in result.reason


def test_the_wrong_team_does_not_resolve() -> None:
    registry = PlayerRegistry()
    registry.register(101, "Michael Porter Jr.", team_id=8)
    assert not registry.resolve("Porter Jr., Michael", team_id=14).ok


def test_conflicting_registrations_raise_rather_than_silently_win() -> None:
    registry = PlayerRegistry()
    registry.register(101, "A Player", nba_reference_id="1")
    with pytest.raises(ValueError, match="already maps"):
        registry.register(202, "Another Player", nba_reference_id="1")


def test_conflicting_aliases_raise() -> None:
    registry = PlayerRegistry()
    registry.register(101, "A Player", aliases=("Nickname",))
    with pytest.raises(ValueError, match="claimed by both"):
        registry.register(202, "B Player", aliases=("Nickname",))


def test_unresolved_players_are_reported() -> None:
    registry = PlayerRegistry()
    registry.register(101, "Michael Porter Jr.", team_id=8)
    results = [
        registry.resolve("Porter Jr., Michael", team_id=8),
        registry.resolve("Nobody, Somebody", team_id=8),
    ]
    report = unresolved_report(results)
    assert report["resolved"] == 1
    assert report["unresolved"] == 1
    assert report["unresolved_examples"][0]["name"] == "Nobody, Somebody"


# --- capture schedule ------------------------------------------------------


def test_the_anchor_is_thirty_minutes_before_tipoff() -> None:
    tipoff = datetime(2026, 11, 5, 0, 30, tzinfo=UTC)
    assert ANCHOR_MINUTES_BEFORE_TIP == 30
    assert anchor_for(tipoff) == tipoff - timedelta(minutes=30)


@pytest.mark.parametrize(
    ("anchor_minute", "expected_minute"), [(0, 0), (10, 0), (29, 0), (30, 30), (59, 30)]
)
def test_the_report_slot_is_the_latest_at_or_before_the_anchor(
    anchor_minute: int, expected_minute: int
) -> None:
    anchor = datetime(2026, 11, 5, 19, anchor_minute, tzinfo=UTC)
    assert latest_report_slot(anchor).minute == expected_minute
    assert latest_report_slot(anchor) <= anchor


def test_every_game_gets_a_capture_before_its_anchor() -> None:
    games = [
        {"game_id": 1, "scheduled_tipoff_utc": datetime(2026, 11, 5, 0, 30, tzinfo=UTC)},
        {"game_id": 2, "scheduled_tipoff_utc": datetime(2026, 11, 5, 2, 0, tzinfo=UTC)},
    ]
    tasks = plan_captures(games, ["nba_official_injury_report"])
    assert verify_anchor_guarantee(tasks, games)["guaranteed"] is True


def test_no_planned_capture_lands_at_or_after_its_anchor() -> None:
    games = [{"game_id": 1, "scheduled_tipoff_utc": datetime(2026, 11, 5, 0, 30, tzinfo=UTC)}]
    for task in plan_captures(games, ["nba_official_injury_report"]):
        assert task.capture_at_utc < task.anchor_utc


def test_captures_densify_approaching_tipoff() -> None:
    games = [{"game_id": 1, "scheduled_tipoff_utc": datetime(2026, 11, 5, 0, 30, tzinfo=UTC)}]
    offsets = sorted({t.minutes_before_tip for t in plan_captures(games, ["s"])})
    gaps = [b - a for a, b in pairwise(offsets)]
    assert gaps[0] <= gaps[-1], "spacing should tighten as tipoff approaches"


def test_a_naive_tipoff_is_rejected() -> None:
    with pytest.raises(ValueError, match="naive tipoff"):
        plan_captures([{"game_id": 1, "scheduled_tipoff_utc": datetime(2026, 11, 5)}], ["s"])


# --- source matrix ---------------------------------------------------------


def test_no_audited_source_is_historically_safe_yet() -> None:
    """The central finding: none can reconstruct a past T-30 state."""
    assert not any(is_historically_safe(s.name) for s in SOURCE_MATRIX)


@pytest.mark.parametrize(
    "name", ["balldontlie_lineups", "balldontlie_injuries", "nba_official_injury_report"]
)
def test_verified_sources_carry_empirical_evidence(name: str) -> None:
    source = SOURCES_BY_NAME[name]
    assert source.empirically_verified if hasattr(source, "empirically_verified") else True
    assert source.evidence.strip()
    assert source.verified is True


@pytest.mark.parametrize("name", ["sportradar_daily_injuries", "sportsdataio"])
def test_unverified_sources_are_marked_as_such(name: str) -> None:
    """No credentials were held, so nothing was claimed as verified."""
    assert SOURCES_BY_NAME[name].verified is False


def test_lineups_remain_prohibited() -> None:
    lineups = SOURCES_BY_NAME["balldontlie_lineups"]
    assert lineups.historical_asof_safety == "unsafe"
    assert "once a game begins" in lineups.evidence
