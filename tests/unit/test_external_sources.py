"""Third-party archives: schema, provenance, and as-of classification."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from nba_prediction_market.availability.external_sources import (
    DATE_ONLY,
    EXTERNAL_SOURCES,
    PROFESSOR_PETE,
    STATSURGE,
    T30_SAFE,
    legacy_slot_timestamp,
    professor_pete_events,
    statsurge_events,
)

PP_ROW = {
    "snapshot_date": "2025-01-15", "snapshot_time": "05PM",
    "game_date": "01/15/2025", "game_time": "07:00", "matchup": "MIN@DEN",
    "team": "Denver Nuggets", "player": "Jokic, Nikola",
    "status": "Questionable", "reason": "Injury/Illness - Right Wrist",
}
SS_ROW = {
    "PLAYER": "Jokic, Nikola", "STATUS": "Questionable",
    "REASON": "Injury/Illness - Right Wrist", "TEAM": "Denver Nuggets",
    "GAME": "MIN@DEN", "DATE": "01/15/2025",
}


class TestLegacySlotTimestamps:
    def test_a_slot_resolves_to_half_past_its_hour(self):
        ts = legacy_slot_timestamp("2025-01-15", "05PM")
        assert (ts.hour, ts.minute) == (17, 30)

    def test_morning_and_noon_slots_resolve_correctly(self):
        assert legacy_slot_timestamp("2025-01-15", "09AM").hour == 9
        assert legacy_slot_timestamp("2025-01-15", "12PM").hour == 12
        assert legacy_slot_timestamp("2025-01-15", "12AM").hour == 0

    def test_reading_the_slot_as_the_hour_would_understate_it(self):
        # This is the failure mode the +:30 rule prevents: treating the report
        # as half an hour earlier than published lets it satisfy an anchor it
        # actually postdates.
        ts = legacy_slot_timestamp("2025-01-15", "05PM")
        naive = ts.replace(minute=0)
        assert ts > naive

    def test_an_unparseable_slot_is_refused(self):
        with pytest.raises(ValueError, match="unrecognised slot"):
            legacy_slot_timestamp("2025-01-15", "05XM")


class TestProfessorPete:
    def test_rows_carry_an_exact_recoverable_timestamp(self):
        event = professor_pete_events([PP_ROW])[0]
        assert event.timestamp_precision == "exact_timestamp"
        assert event.asof_class == T30_SAFE
        assert event.observed_at_utc == datetime(2025, 1, 15, 22, 30, tzinfo=UTC)

    def test_the_snapshot_is_identified_not_inferred_from_row_order(self):
        event = professor_pete_events([PP_ROW])[0]
        assert event.source_report_id == "2025-01-15_05PM"

    def test_matchup_and_game_date_are_normalized(self):
        event = professor_pete_events([PP_ROW])[0]
        assert (event.away_team, event.home_team) == ("MIN", "DEN")
        assert event.game_date == "2025-01-15"

    def test_status_is_normalized_with_the_raw_kept(self):
        event = professor_pete_events([PP_ROW])[0]
        assert event.status_normalized == "questionable"
        assert event.status_raw == "Questionable"

    def test_team_level_not_yet_submitted_rows_are_not_player_events(self):
        row = dict(PP_ROW, player="", status="NOT YET SUBMITTED", reason="")
        assert professor_pete_events([row]) == []

    def test_an_observation_before_an_anchor_is_usable(self):
        event = professor_pete_events([PP_ROW])[0]
        assert event.usable_for_anchor(datetime(2025, 1, 15, 23, 30, tzinfo=UTC))

    def test_an_observation_after_an_anchor_is_refused(self):
        event = professor_pete_events([PP_ROW])[0]
        assert not event.usable_for_anchor(datetime(2025, 1, 15, 22, 29, tzinfo=UTC))


class TestStatSurge:
    def test_no_timestamp_is_invented(self):
        # The file has no time field. Synthesising one is what would let a
        # date-only record masquerade as an anchor observation.
        event = statsurge_events([SS_ROW])[0]
        assert event.observed_at_utc is None
        assert event.timestamp_precision == "date_only"

    def test_it_is_classified_date_only_not_t30(self):
        assert statsurge_events([SS_ROW])[0].asof_class == DATE_ONLY
        assert STATSURGE.asof_class == DATE_ONLY

    def test_a_date_only_record_can_never_answer_an_anchor(self):
        event = statsurge_events([SS_ROW])[0]
        for hour in range(0, 24, 3):
            anchor = datetime(2025, 1, 16, hour, 0, tzinfo=UTC)
            assert not event.usable_for_anchor(anchor)

    def test_fields_are_normalized_without_losing_the_raw_values(self):
        event = statsurge_events([SS_ROW])[0]
        assert event.game_date == "2025-01-15"
        assert (event.away_team, event.home_team) == ("MIN", "DEN")
        assert event.player_name_raw == "Jokic, Nikola"
        assert event.status_normalized == "questionable"
        assert event.reason_raw == "Injury/Illness - Right Wrist"


class TestProvenance:
    def test_no_external_source_is_marked_redistributable(self):
        assert all(not s.redistributable for s in EXTERNAL_SOURCES)

    def test_every_source_records_its_licence_position(self):
        for source in EXTERNAL_SOURCES:
            assert source.license_status
            assert source.to_dict()["license_status"] == source.license_status

    def test_professor_pete_is_t30_safe_but_flagged_as_stale(self):
        assert PROFESSOR_PETE.asof_class == T30_SAFE
        assert "staler" in PROFESSOR_PETE.notes

    def test_the_statsurge_two_pm_claim_is_recorded_as_unverifiable(self):
        assert "cannot be checked" in STATSURGE.notes


class TestNormalizedShape:
    def test_both_sources_emit_the_same_fields(self):
        pp = professor_pete_events([PP_ROW])[0].to_dict()
        ss = statsurge_events([SS_ROW])[0].to_dict()
        assert set(pp) == set(ss)
        for required in ("source", "source_report_id", "observed_at_utc",
                         "timestamp_precision", "asof_class", "player_name_raw",
                         "status_raw", "status_normalized", "reason_raw"):
            assert required in pp

    def test_source_identity_is_preserved(self):
        assert professor_pete_events([PP_ROW])[0].source == PROFESSOR_PETE.name
        assert statsurge_events([SS_ROW])[0].source == STATSURGE.name


class TestDeterministicReplay:
    """The same input must always produce the same normalized output."""

    def test_parsing_the_same_rows_twice_is_identical(self):
        first = [e.to_dict() for e in professor_pete_events([PP_ROW, PP_ROW])]
        second = [e.to_dict() for e in professor_pete_events([PP_ROW, PP_ROW])]
        assert first == second

    def test_row_order_does_not_change_any_events_timestamp(self):
        # Timestamps come from the snapshot fields, never from position.
        other = dict(PP_ROW, snapshot_time="08PM", player="Murray, Jamal")
        forward = {e.player_name_raw: e.observed_at_utc
                   for e in professor_pete_events([PP_ROW, other])}
        backward = {e.player_name_raw: e.observed_at_utc
                    for e in professor_pete_events([other, PP_ROW])}
        assert forward == backward

    def test_statsurge_replay_is_stable(self):
        first = [e.to_dict() for e in statsurge_events([SS_ROW])]
        second = [e.to_dict() for e in statsurge_events([SS_ROW])]
        assert first == second
