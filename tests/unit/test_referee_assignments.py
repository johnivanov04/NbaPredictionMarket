"""Prospective assignment capture: as-of semantics and change handling."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from nba_prediction_market.referees.assignments import (
    CHANGE_DISAPPEARED,
    CHANGE_NEW,
    CHANGE_REASSIGNED,
    POSITIONS,
    AssignmentLedger,
    parse_assignments,
)

MORNING = datetime(2026, 10, 20, 13, 5, tzinfo=UTC)   # ~09:05 ET
AFTERNOON = datetime(2026, 10, 20, 18, 0, tzinfo=UTC)
ANCHOR_T6 = datetime(2026, 10, 20, 17, 0, tzinfo=UTC)

TABLE = """
<table><thead><tr><th>Game</th><th>Crew Chief</th><th>Referee</th>
<th>Umpire</th><th>Alternate</th></tr></thead><tbody>
<tr><td>BOS@DET</td><td>Scott Foster</td><td>Kevin Cutler</td>
<td>Suyash Mehta</td><td>Brent Barnaky</td></tr>
<tr><td>OKC@SAS</td><td>Tony Brothers</td><td>James Williams</td>
<td>Ashley Moyer-Gleich</td><td></td></tr>
</tbody></table>
"""


class TestParsing:
    def test_rows_and_positions_are_read(self):
        rows = parse_assignments(TABLE, observed_at_utc=MORNING)
        assert [r.matchup for r in rows] == ["BOS@DET", "OKC@SAS"]
        assert rows[0].crew_chief == "Scott Foster"
        assert rows[0].umpire == "Suyash Mehta"
        assert rows[0].alternate == "Brent Barnaky"

    def test_header_row_is_not_an_assignment(self):
        assert len(parse_assignments(TABLE, observed_at_utc=MORNING)) == 2

    def test_a_missing_alternate_is_none_not_empty_string(self):
        rows = parse_assignments(TABLE, observed_at_utc=MORNING)
        assert rows[1].alternate is None

    def test_alternate_is_excluded_from_the_working_crew(self):
        """The alternate does not officiate unless promoted."""
        rows = parse_assignments(TABLE, observed_at_utc=MORNING)
        assert rows[0].assigned_names == (
            "Scott Foster", "Kevin Cutler", "Suyash Mehta"
        )

    def test_an_empty_table_is_a_normal_state(self):
        """Offseason, and every morning before the league posts."""
        html = "<table><thead><tr><th>Game</th></tr></thead><tbody></tbody></table>"
        assert parse_assignments(html, observed_at_utc=MORNING) == []

    def test_every_position_is_captured(self):
        rows = parse_assignments(TABLE, observed_at_utc=MORNING)
        assert set(rows[0].officials) == set(POSITIONS)

    def test_observation_time_is_stamped_on_every_row(self):
        rows = parse_assignments(TABLE, observed_at_utc=MORNING)
        assert all(r.first_observed_at_utc == MORNING for r in rows)


class TestAsOfSemantics:
    def test_a_crew_is_invisible_before_it_was_observed(self):
        ledger = AssignmentLedger()
        ledger.observe(parse_assignments(TABLE, observed_at_utc=AFTERNOON),
                       now=AFTERNOON)
        assert ledger.known_at("BOS@DET", ANCHOR_T6) is None
        assert ledger.was_known_at("BOS@DET", ANCHOR_T6) is False

    def test_a_crew_observed_before_the_cutoff_is_visible(self):
        ledger = AssignmentLedger()
        ledger.observe(parse_assignments(TABLE, observed_at_utc=MORNING),
                       now=MORNING)
        assert ledger.known_at("BOS@DET", ANCHOR_T6).crew_chief == "Scott Foster"

    def test_a_late_discovery_is_never_backfilled_into_an_earlier_anchor(self):
        """The property that stops hindsight leaking into a prediction."""
        ledger = AssignmentLedger()
        ledger.observe(parse_assignments(TABLE, observed_at_utc=MORNING),
                       now=MORNING)
        late = TABLE.replace("Scott Foster", "Marc Davis")
        ledger.observe(parse_assignments(late, observed_at_utc=AFTERNOON),
                       now=AFTERNOON)
        # At T-6h only the morning crew existed.
        assert ledger.known_at("BOS@DET", ANCHOR_T6).crew_chief == "Scott Foster"
        # Later, the reassignment is the current truth.
        assert ledger.known_at("BOS@DET", AFTERNOON).crew_chief == "Marc Davis"


class TestChangeHandling:
    def test_first_observation_is_reported_as_new(self):
        ledger = AssignmentLedger()
        changes = ledger.observe(
            parse_assignments(TABLE, observed_at_utc=MORNING), now=MORNING
        )
        assert {c.change for c in changes} == {CHANGE_NEW}

    def test_reobserving_the_same_crew_reports_nothing(self):
        ledger = AssignmentLedger()
        ledger.observe(parse_assignments(TABLE, observed_at_utc=MORNING),
                       now=MORNING)
        again = ledger.observe(
            parse_assignments(TABLE, observed_at_utc=AFTERNOON), now=AFTERNOON
        )
        assert again == []

    def test_first_observed_time_survives_reobservation(self):
        ledger = AssignmentLedger()
        ledger.observe(parse_assignments(TABLE, observed_at_utc=MORNING),
                       now=MORNING)
        ledger.observe(parse_assignments(TABLE, observed_at_utc=AFTERNOON),
                       now=AFTERNOON)
        assert ledger.known_at("BOS@DET", AFTERNOON).first_observed_at_utc == (
            MORNING
        )

    def test_a_reassignment_preserves_both_states(self):
        ledger = AssignmentLedger()
        ledger.observe(parse_assignments(TABLE, observed_at_utc=MORNING),
                       now=MORNING)
        changed = TABLE.replace("Suyash Mehta", "Josh Tiven")
        changes = ledger.observe(
            parse_assignments(changed, observed_at_utc=AFTERNOON), now=AFTERNOON
        )
        assert [c.change for c in changes] == [CHANGE_REASSIGNED]
        assert len(ledger.states["BOS@DET"]) == 2
        assert ledger.states["BOS@DET"][0].umpire == "Suyash Mehta"
        assert ledger.states["BOS@DET"][1].umpire == "Josh Tiven"

    def test_a_vanished_matchup_is_reported_not_deleted(self):
        ledger = AssignmentLedger()
        ledger.observe(parse_assignments(TABLE, observed_at_utc=MORNING),
                       now=MORNING)
        one_game = TABLE.replace(
            "<tr><td>OKC@SAS</td><td>Tony Brothers</td><td>James Williams</td>\n"
            "<td>Ashley Moyer-Gleich</td><td></td></tr>", ""
        )
        changes = ledger.observe(
            parse_assignments(one_game, observed_at_utc=AFTERNOON), now=AFTERNOON
        )
        assert [c.change for c in changes] == [CHANGE_DISAPPEARED]
        assert ledger.states["OKC@SAS"], "history must survive disappearance"

    def test_summary_counts_reassignments(self):
        ledger = AssignmentLedger()
        ledger.observe(parse_assignments(TABLE, observed_at_utc=MORNING),
                       now=MORNING)
        ledger.observe(
            parse_assignments(TABLE.replace("Kevin Cutler", "Marc Davis"),
                              observed_at_utc=AFTERNOON),
            now=AFTERNOON,
        )
        assert ledger.summary() == {
            "matchups": 2, "reassigned": 1, "total_states": 3
        }

    def test_an_anchor_between_two_states_sees_the_earlier_one(self):
        ledger = AssignmentLedger()
        ledger.observe(parse_assignments(TABLE, observed_at_utc=MORNING),
                       now=MORNING)
        mid = MORNING + timedelta(hours=1)
        ledger.observe(
            parse_assignments(TABLE.replace("Kevin Cutler", "Marc Davis"),
                              observed_at_utc=mid + timedelta(hours=1)),
            now=mid + timedelta(hours=1),
        )
        assert ledger.known_at("BOS@DET", mid).referee == "Kevin Cutler"
