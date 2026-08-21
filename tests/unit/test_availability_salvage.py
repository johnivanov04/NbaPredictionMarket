"""Salvage pipeline: teams that had not filed, and the T-30 anchor rule."""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta

import pandas as pd
import pytest

from nba_prediction_market.availability.nba_report_parser import (
    NotSubmitted,
    ParsedReport,
)
from nba_prediction_market.pipelines.build_availability_salvage import (
    build_t30_states,
    not_submitted_index,
)

# 7:30 PM ET on 2026-01-15. Reports date games by their Eastern date, so the
# tipoff has to be picked so the UTC instant and the ET date agree.
TIP = datetime(2026, 1, 16, 0, 30, tzinfo=UTC)
ANCHOR = TIP - timedelta(minutes=30)
FILENAME = "Injury-Report_2026-01-14_07_00PM.pdf"


GAME_DATE = "01/15/2026"
MATCHUP = "MIA@LAC"


def _report(
    *teams: str,
    filename: str = FILENAME,
    game_date: str = GAME_DATE,
    matchup: str = MATCHUP,
) -> ParsedReport:
    return ParsedReport(
        source_filename=filename,
        report_timestamp_et=datetime(2026, 1, 14, 19, 0),
        report_date=date(2026, 1, 14),
        teams_not_submitted=[
            NotSubmitted(game_date=game_date, matchup=matchup, team=team)
            for team in teams
        ],
    )


KEY = (FILENAME, "2026-01-15", "MIA", "LAC")


class TestNotSubmittedIndex:
    def test_team_names_resolve_to_codes(self):
        assert not_submitted_index([_report("LA Clippers")]) == {(*KEY, "LAC")}

    def test_several_teams_in_one_report_are_all_indexed(self):
        index = not_submitted_index([_report("Miami Heat", "Phoenix Suns")])
        assert index == {(*KEY, "MIA"), (*KEY, "PHX")}

    def test_the_index_is_keyed_by_report_so_a_later_filing_is_not_retroactive(self):
        # The 7pm report is missing Miami; the 8pm one is not. Keying by
        # filename is what stops the later filing from erasing the earlier gap.
        index = not_submitted_index([
            _report("Miami Heat"),
            _report(filename="Injury-Report_2026-01-14_08_00PM.pdf"),
        ])
        assert index == {(*KEY, "MIA")}

    def test_an_unresolvable_team_name_is_dropped_not_guessed(self):
        assert not_submitted_index([_report("Springfield Isotopes")]) == set()

    def test_a_pending_filing_for_another_date_is_a_different_key(self):
        # One report lists several dates. A team pending for tomorrow must not
        # taint tonight's game.
        index = not_submitted_index([_report("LA Clippers", game_date="01/16/2026")])
        assert index == {(FILENAME, "2026-01-16", "MIA", "LAC", "LAC")}

    def test_a_pending_row_without_game_context_is_dropped(self):
        assert not_submitted_index([_report("LA Clippers", matchup="")]) == set()

    def test_a_report_with_nothing_pending_contributes_nothing(self):
        assert not_submitted_index([_report()]) == set()


def _games() -> pd.DataFrame:
    return pd.DataFrame([{
        "nba_game_id": 101,
        "game_datetime_utc": TIP,
        "home_team": "LAC",
        "away_team": "MIA",
    }])


def _events(report_ts: datetime, filename: str = FILENAME) -> pd.DataFrame:
    return pd.DataFrame([{
        "report_timestamp_utc": report_ts,
        "source_filename": filename,
        "game_date": "01/15/2026",
        "away_team": "MIA",
        "home_team": "LAC",
        "team_code": "MIA",
        "player_name": "Butler, Jimmy",
        "balldontlie_player_id": 42,
        "status_normalized": "out",
    }])


MATCHED = {("2026-01-15", "MIA", "LAC"): 101}


class TestT30NotSubmitted:
    def test_a_pending_team_downgrades_coverage_quality(self):
        t30 = build_t30_states(
            _events(ANCHOR - timedelta(minutes=5)), _games(), MATCHED,
            {(*KEY, "LAC")},
        )
        row = t30.iloc[0]
        assert row["coverage_quality"] == "team_not_yet_submitted"
        assert row["teams_not_submitted"] == "LAC"
        assert not row["both_teams_submitted"]
        # The state is still recorded -- the flag qualifies it, not deletes it.
        assert row["t30_state_available"]

    def test_both_teams_filed_is_ordinary_coverage(self):
        t30 = build_t30_states(
            _events(ANCHOR - timedelta(minutes=5)), _games(), MATCHED, set()
        )
        row = t30.iloc[0]
        assert row["coverage_quality"] == "ok"
        assert row["teams_not_submitted"] == ""
        assert row["both_teams_submitted"]

    def test_a_pending_team_from_a_different_report_does_not_apply(self):
        t30 = build_t30_states(
            _events(ANCHOR - timedelta(minutes=5)), _games(), MATCHED,
            {("some-other-report.pdf", "2026-01-15", "MIA", "LAC", "LAC")},
        )
        assert t30.iloc[0]["both_teams_submitted"]

    def test_a_pending_team_from_another_game_does_not_apply(self):
        t30 = build_t30_states(
            _events(ANCHOR - timedelta(minutes=5)), _games(), MATCHED,
            {(*KEY, "BOS")},
        )
        assert t30.iloc[0]["both_teams_submitted"]


class TestAnchorRule:
    def test_a_report_published_after_the_anchor_is_refused(self):
        t30 = build_t30_states(
            _events(ANCHOR + timedelta(seconds=1)), _games(), MATCHED, set()
        )
        row = t30.iloc[0]
        assert row["coverage_quality"] == "first_report_after_anchor"
        assert not row["t30_state_available"]
        assert row["selected_report_timestamp_utc"] is None

    def test_a_report_exactly_at_the_anchor_is_accepted(self):
        t30 = build_t30_states(_events(ANCHOR), _games(), MATCHED, set())
        assert t30.iloc[0]["t30_state_available"]
        assert t30.iloc[0]["report_age_minutes"] == 0.0

    def test_a_game_with_no_surviving_report_is_never_backfilled(self):
        t30 = build_t30_states(
            pd.DataFrame(columns=_events(ANCHOR).columns), _games(), MATCHED, set()
        )
        row = t30.iloc[0]
        assert row["coverage_quality"] == "no_surviving_report"
        assert not row["t30_state_available"]
        # Unknown, not "a team failed to file".
        assert pd.isna(row["both_teams_submitted"])
        assert row["teams_not_submitted"] is None

    def test_the_latest_eligible_report_wins(self):
        early = _events(ANCHOR - timedelta(hours=6), "early.pdf")
        late = _events(ANCHOR - timedelta(minutes=10), "late.pdf")
        t30 = build_t30_states(
            pd.concat([early, late], ignore_index=True), _games(), MATCHED, set()
        )
        row = t30.iloc[0]
        assert row["selected_report_filename"] == "late.pdf"
        assert row["report_age_minutes"] == pytest.approx(10.0)

    def test_a_stale_but_valid_report_is_flagged_not_discarded(self):
        t30 = build_t30_states(
            _events(ANCHOR - timedelta(hours=5)), _games(), MATCHED, set()
        )
        row = t30.iloc[0]
        assert row["coverage_quality"] == "stale_report"
        assert row["t30_state_available"]
        assert row["report_age_minutes"] == pytest.approx(300.0)
