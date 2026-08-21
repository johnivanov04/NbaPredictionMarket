"""Coverage accounting over the salvaged report archive."""

from __future__ import annotations

from datetime import date

from nba_prediction_market.availability.archive_inventory import (
    SLOTS_PER_DAY,
    build_inventory,
)
from nba_prediction_market.availability.nba_official import slots_for_date

DAY_A = date(2025, 12, 24)
DAY_B = date(2025, 12, 25)
DAY_C = date(2025, 12, 26)
DAY_D = date(2025, 12, 27)


def _full(day: date):
    return slots_for_date(day)


def _partial(day: date, count: int):
    return slots_for_date(day)[:count]


class TestEmptyArchive:
    def test_an_empty_archive_reports_nothing_rather_than_guessing(self):
        inventory = build_inventory([])
        assert inventory.total_reports == 0
        assert inventory.earliest_report is None
        assert inventory.latest_report is None
        assert inventory.coverage_fraction == 0.0
        assert inventory.gaps == ()


class TestCompleteCoverage:
    def test_two_complete_days_report_full_coverage(self):
        inventory = build_inventory(_full(DAY_A) + _full(DAY_B))
        assert inventory.total_reports == 2 * SLOTS_PER_DAY
        assert inventory.earliest_report == DAY_A
        assert inventory.latest_report == DAY_B
        assert inventory.complete_days == 2
        assert inventory.partial_days == 0
        assert inventory.empty_days == 0
        assert inventory.coverage_fraction == 1.0
        assert inventory.gaps == ()

    def test_a_day_has_forty_eight_slots(self):
        assert SLOTS_PER_DAY == 48
        assert len(slots_for_date(DAY_A)) == 48


class TestPartialCoverage:
    def test_a_partial_day_names_the_slots_it_is_missing(self):
        inventory = build_inventory(_full(DAY_A) + _partial(DAY_B, 10))
        assert inventory.partial_days == 1
        detail = inventory.partial_day_details[0]
        assert detail.report_date == DAY_B
        assert detail.archived == 10
        assert len(detail.missing_slots) == SLOTS_PER_DAY - 10
        # The named slots are real filenames, not placeholders.
        assert all(name.startswith("Injury-Report_2025-12-25_") for name in detail.missing_slots)

    def test_coverage_fraction_counts_slots_not_days(self):
        inventory = build_inventory(_full(DAY_A) + _partial(DAY_B, 24))
        assert inventory.coverage_fraction == (SLOTS_PER_DAY + 24) / (2 * SLOTS_PER_DAY)


class TestGaps:
    def test_a_missing_day_inside_the_span_is_reported_as_a_gap(self):
        inventory = build_inventory(_full(DAY_A) + _full(DAY_C))
        assert inventory.empty_days == 1
        assert [g.to_dict() for g in inventory.gaps] == [
            {"start": DAY_B.isoformat(), "end": DAY_B.isoformat(), "days": 1}
        ]

    def test_consecutive_missing_days_collapse_into_one_run(self):
        inventory = build_inventory(_full(DAY_A) + _full(DAY_D))
        assert len(inventory.gaps) == 1
        gap = inventory.gaps[0]
        assert (gap.start, gap.end, gap.days) == (DAY_B, DAY_C, 2)

    def test_the_span_never_extends_past_the_outermost_archived_day(self):
        inventory = build_inventory(_full(DAY_B) + _full(DAY_C))
        assert inventory.days_observed == 2
        assert inventory.empty_days == 0


class TestBlockedRunDetection:
    def test_a_near_empty_day_between_complete_days_is_flagged_as_suspect(self):
        # A rate-limited stretch looks identical to non-publication in the
        # sidecars, so it has to surface as a suspicion rather than a fact.
        inventory = build_inventory(_full(DAY_A) + _partial(DAY_B, 2) + _full(DAY_C))
        assert inventory.suspect_blocked_days == (DAY_B,)

    def test_a_merely_thin_day_is_not_flagged(self):
        inventory = build_inventory(_full(DAY_A) + _partial(DAY_B, 30) + _full(DAY_C))
        assert inventory.suspect_blocked_days == ()

    def test_a_thin_day_at_the_edge_of_the_span_is_not_flagged(self):
        # The first archived day is thin because retention ran out there, not
        # because a run was blocked.
        inventory = build_inventory(_partial(DAY_A, 2) + _full(DAY_B) + _full(DAY_C))
        assert inventory.suspect_blocked_days == ()


class TestSerialisation:
    def test_the_report_is_json_safe_and_carries_the_403_caveat(self):
        payload = build_inventory(_full(DAY_A) + _partial(DAY_B, 10)).to_dict()
        assert payload["earliest_report_date"] == DAY_A.isoformat()
        assert payload["total_reports_archived"] == SLOTS_PER_DAY + 10
        assert "rate limited" in payload["caveat"]
        import json

        json.dumps(payload)
