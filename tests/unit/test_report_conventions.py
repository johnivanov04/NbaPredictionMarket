"""Both published filename conventions, and the cutover between them."""

from __future__ import annotations

from datetime import UTC, date, datetime

import pytest

from nba_prediction_market.availability.nba_official import (
    CONVENTION_CUTOVER_ET,
    EARLIEST_AVAILABLE_REPORT_DATE,
    EASTERN,
    LEGACY_SLOT_MINUTE,
    latest_slot_at_or_before,
    slot_from_filename,
    slots_for_date,
)

LEGACY_DAY = date(2025, 1, 15)
CUTOVER_DAY = date(2025, 12, 22)
MODERN_DAY = date(2026, 1, 15)


class TestFilenameConventions:
    def test_a_legacy_slot_carries_only_the_hour(self):
        slot = next(s for s in slots_for_date(LEGACY_DAY) if s.hour_24 == 17)
        assert slot.filename == "Injury-Report_2025-01-15_05PM.pdf"
        assert slot.is_legacy

    def test_a_modern_slot_carries_hour_and_minute(self):
        slot = next(
            s for s in slots_for_date(MODERN_DAY) if s.hour_24 == 17 and s.minute == 30
        )
        assert slot.filename == "Injury-Report_2026-01-15_05_30PM.pdf"
        assert not slot.is_legacy

    def test_a_legacy_report_is_stamped_at_half_past(self):
        # The filename says 05PM but the report inside is the 5:30 one. Reading
        # it as 5:00 would place the observation earlier than it happened.
        slot = slot_from_filename("Injury-Report_2025-01-15_05PM.pdf")
        assert slot.minute == LEGACY_SLOT_MINUTE
        assert slot.report_timestamp_et.hour == 17
        assert slot.report_timestamp_et.minute == 30

    @pytest.mark.parametrize(
        "filename",
        ["Injury-Report_2025-01-15_05PM.pdf", "Injury-Report_2026-01-24_05_30PM.pdf"],
    )
    def test_filenames_round_trip(self, filename):
        assert slot_from_filename(filename).filename == filename

    def test_a_filename_that_is_not_a_report_is_refused(self):
        assert slot_from_filename("something-else.pdf") is None
        assert slot_from_filename("Injury-Report_2025-01-15_05XM.pdf") is None


class TestSlotGrids:
    def test_legacy_dates_publish_hourly(self):
        slots = slots_for_date(LEGACY_DAY)
        assert len(slots) == 24
        assert {s.minute for s in slots} == {LEGACY_SLOT_MINUTE}

    def test_modern_dates_publish_half_hourly(self):
        assert len(slots_for_date(MODERN_DAY)) == 48

    def test_the_cutover_day_carries_both_grids(self):
        slots = slots_for_date(CUTOVER_DAY)
        legacy = [s for s in slots if s.is_legacy]
        modern = [s for s in slots if not s.is_legacy]
        # Measured: the 08:30 report is legacy-named, the 09:00 one is not.
        assert len(legacy) == 9
        assert len(modern) == 30
        assert len(slots) == 39
        assert legacy[-1].report_timestamp_et.strftime("%H:%M") == "08:30"
        assert modern[0].report_timestamp_et.strftime("%H:%M") == "09:00"

    def test_slots_are_chronological(self):
        for day in (LEGACY_DAY, CUTOVER_DAY, MODERN_DAY):
            stamps = [s.report_timestamp_utc for s in slots_for_date(day)]
            assert stamps == sorted(stamps)

    def test_the_cutover_instant_is_the_measured_one(self):
        assert datetime(2025, 12, 22, 9, 0, tzinfo=EASTERN) == CONVENTION_CUTOVER_ET
        assert date(2018, 12, 17) == EARLIEST_AVAILABLE_REPORT_DATE


class TestAnchorSelection:
    def test_a_legacy_anchor_never_selects_a_later_report(self):
        # 19:00 must resolve to 18:30, not 19:30 -- 19:30 is published after.
        anchor = datetime(2025, 1, 15, 19, 0, tzinfo=EASTERN)
        slot = latest_slot_at_or_before(anchor)
        assert slot.report_timestamp_et.strftime("%H:%M") == "18:30"
        assert slot.report_timestamp_et <= anchor

    def test_a_legacy_anchor_past_the_half_hour_uses_that_hour(self):
        anchor = datetime(2025, 1, 15, 19, 45, tzinfo=EASTERN)
        assert latest_slot_at_or_before(anchor).report_timestamp_et.strftime("%H:%M") == "19:30"

    def test_a_legacy_anchor_just_after_midnight_crosses_the_day(self):
        anchor = datetime(2025, 1, 16, 0, 15, tzinfo=EASTERN)
        slot = latest_slot_at_or_before(anchor)
        assert slot.report_date == date(2025, 1, 15)
        assert slot.report_timestamp_et.strftime("%H:%M") == "23:30"

    def test_a_modern_anchor_lands_on_the_half_hour_grid(self):
        anchor = datetime(2026, 1, 15, 19, 0, tzinfo=EASTERN)
        assert latest_slot_at_or_before(anchor).report_timestamp_et.strftime("%H:%M") == "19:00"

    @pytest.mark.parametrize("minutes", list(range(0, 60, 7)))
    def test_no_selected_slot_ever_postdates_its_anchor(self, minutes):
        for day in (datetime(2025, 3, 4, 20, minutes, tzinfo=UTC),
                    datetime(2026, 3, 4, 20, minutes, tzinfo=UTC)):
            assert latest_slot_at_or_before(day).report_timestamp_utc <= day

    def test_a_naive_anchor_is_rejected(self):
        with pytest.raises(ValueError, match="timezone-aware"):
            latest_slot_at_or_before(datetime(2025, 1, 15, 19, 0))
