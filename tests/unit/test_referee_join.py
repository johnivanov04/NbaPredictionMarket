"""The deterministic join from Basketball-Reference pages to trusted game ids."""

from __future__ import annotations

from datetime import UTC, datetime

import pandas as pd
import pytest

from nba_prediction_market.pipelines.build_referee_assignments import (
    EMPTY_CREW,
    MAPPED,
    NO_BOXSCORE,
    NO_OFFICIALS_BLOCK,
    assemble,
)

TIP = datetime(2023, 10, 24, 23, 0, tzinfo=UTC)   # 19:00 ET


def games(*rows):
    frame = pd.DataFrame(rows)
    frame["join_key"] = (
        frame["game_date_et"] + "|" + frame["away_team"] + "|" + frame["home_team"]
    )
    return frame


def game(gid=1, *, date="2023-10-24", away="LAL", home="DEN", season=2023):
    return {
        "nba_game_id": gid, "season": season, "game_datetime_utc": TIP,
        "game_date_et": date, "home_team": home, "away_team": away,
    }


def source(*, date="2023-10-24", away="LAL", home="DEN", season=2023,
           quality=MAPPED, officials=None, path="/boxscores/202310240DEN.html"):
    if officials is None:
        officials = [
            {"referee_slug": "a99r", "referee_name": "A One"},
            {"referee_slug": "b99r", "referee_name": "B Two"},
            {"referee_slug": "c99r", "referee_name": "C Three"},
        ]
    return {
        "season": season, "boxscore_path": path, "game_date_et": date,
        "home_team": home, "away_team": away,
        "mapping_quality": quality, "officials": officials,
        "crew_size": len(officials),
    }


class TestJoin:
    def test_a_matching_game_carries_its_crew(self):
        frame, report = assemble(None, [source()], games(game()))
        assert report["matched_games"] == 1
        assert list(frame.loc[0, "referee_slugs"]) == ["a99r", "b99r", "c99r"]
        assert frame.loc[0, "source"] == "basketball_reference"

    def test_the_key_is_date_plus_both_team_codes(self):
        """Same date and home team, different visitor: not the same game."""
        _, report = assemble(None, [source(away="BOS")], games(game(away="LAL")))
        assert report["matched_games"] == 0
        assert report["unmatched_games"] == 1

    def test_a_different_date_does_not_match(self):
        _, report = assemble(
            None, [source(date="2023-10-25")], games(game(date="2023-10-24"))
        )
        assert report["unmatched_games"] == 1

    def test_two_source_rows_for_one_key_are_ambiguous_not_arbitrary(self):
        """Attaching the wrong crew is the one error that corrupts silently."""
        frame, report = assemble(
            None, [source(), source(path="/boxscores/other.html")], games(game())
        )
        assert report["ambiguous_games"] == 1
        assert report["matched_games"] == 0
        assert frame.empty

    def test_unmatched_games_are_reported_with_examples(self):
        _, report = assemble(None, [], games(game(1), game(2, home="BOS")))
        assert report["unmatched_games"] == 2
        # With no index supplied, nothing is retryable, so both are absent.
        assert set(report["absent_examples"]) == {"1", "2"}

    def test_expected_games_counts_the_trusted_frame_not_the_source(self):
        _, report = assemble(None, [source()], games(game(1), game(2, home="BOS")))
        assert report["expected_games"] == 2
        assert report["source_rows"] == 1

    def test_extra_source_rows_are_simply_unused(self):
        """The index carries playoff games the regular-season frame lacks."""
        _, report = assemble(
            None, [source(), source(date="2024-05-01")], games(game())
        )
        assert report["matched_games"] == 1
        assert report["expected_games"] == 1


class TestQualityPropagation:
    def test_a_page_that_never_arrived_is_marked(self):
        frame, _ = assemble(
            None, [source(quality=NO_BOXSCORE, officials=[])], games(game())
        )
        assert frame.loc[0, "mapping_quality"] == NO_BOXSCORE
        assert frame.loc[0, "crew_size"] == 0

    def test_a_missing_officials_block_is_distinct_from_an_empty_crew(self):
        no_block, _ = assemble(
            None, [source(quality=NO_OFFICIALS_BLOCK, officials=[])], games(game())
        )
        empty, _ = assemble(
            None, [source(quality=EMPTY_CREW, officials=[])], games(game())
        )
        assert no_block.loc[0, "mapping_quality"] == NO_OFFICIALS_BLOCK
        assert empty.loc[0, "mapping_quality"] == EMPTY_CREW

    def test_a_two_official_crew_is_kept_not_padded(self):
        two = [
            {"referee_slug": "a99r", "referee_name": "A One"},
            {"referee_slug": "b99r", "referee_name": "B Two"},
        ]
        frame, _ = assemble(None, [source(officials=two)], games(game()))
        assert frame.loc[0, "crew_size"] == 2
        assert len(frame.loc[0, "referee_slugs"]) == 2

    def test_missing_officials_are_never_filled_in(self):
        frame, _ = assemble(
            None, [source(quality=EMPTY_CREW, officials=[])], games(game())
        )
        assert list(frame.loc[0, "referee_slugs"]) == []
        assert list(frame.loc[0, "referee_names"]) == []

    def test_quality_and_crew_size_are_summarised(self):
        _, report = assemble(None, [source()], games(game()))
        assert report["by_quality"] == {MAPPED: 1}
        assert report["crew_size_distribution"] == {3: 1}

    def test_per_season_coverage_is_reported(self):
        _, report = assemble(
            None,
            [source(), source(date="2022-11-01", season=2022,
                              path="/boxscores/202211010DEN.html")],
            games(game(1), game(2, date="2022-11-01", season=2022)),
        )
        assert report["by_season"]["2023"] == {"games": 1, "with_crew": 1}
        assert report["by_season"]["2022"] == {"games": 1, "with_crew": 1}


class TestIdentityAudit:
    def test_identity_and_support_are_reported_together(self):
        _, report = assemble(None, [source()], games(game()))
        assert report["identity"]["distinct_referees"] == 3
        assert report["team_referee_support"]["cells"] == 6

    def test_a_shared_name_across_two_slugs_is_surfaced(self):
        officials = [
            {"referee_slug": "will01r", "referee_name": "James Williams"},
            {"referee_slug": "will02r", "referee_name": "James Williams"},
            {"referee_slug": "c99r", "referee_name": "C Three"},
        ]
        _, report = assemble(None, [source(officials=officials)], games(game()))
        assert report["identity"]["normalized_names_shared_by_multiple_ids"] == 1

    def test_a_renamed_official_is_surfaced_not_split(self):
        rows = [
            source(),
            source(date="2023-11-01", path="/boxscores/202311010DEN.html",
                   officials=[{"referee_slug": "a99r", "referee_name": "A. One"}]),
        ]
        _, report = assemble(
            None, rows, games(game(1), game(2, date="2023-11-01"))
        )
        assert report["identity"]["referees_with_multiple_display_names"] == 1
        assert report["identity"]["distinct_referees"] == 3


@pytest.mark.parametrize("bbref,canonical", [
    ("PHO", "PHX"), ("BRK", "BKN"), ("CHO", "CHA"), ("BOS", "BOS"),
])
def test_join_uses_canonical_codes(bbref, canonical):
    """The index is written with canonical codes, so the key lines up."""
    from nba_prediction_market.referees.bbref_source import canonical_team

    assert canonical_team(bbref) == canonical


class TestRetryableVersusAbsent:
    """A handful of timeouts must not masquerade as missing history."""

    def _index(self, *entries):
        return [
            {"season": 2023, "boxscore_path": "/boxscores/x.html",
             "game_date_et": d, "home_team": h, "away_team": a}
            for d, a, h in entries
        ]

    def test_a_listed_but_unretrieved_game_is_retryable(self):
        _, report = assemble(
            None, [], games(game()),
            index=self._index(("2023-10-24", "LAL", "DEN")),
        )
        assert report["unretrieved_but_listed"] == 1
        assert report["absent_from_source"] == 0
        assert report["repair_command"] is not None

    def test_a_game_the_source_never_lists_is_absent(self):
        _, report = assemble(None, [], games(game()), index=[])
        assert report["absent_from_source"] == 1
        assert report["unretrieved_but_listed"] == 0
        assert report["repair_command"] is None

    def test_the_two_categories_partition_the_unmatched(self):
        _, report = assemble(
            None, [], games(game(1), game(2, home="BOS")),
            index=self._index(("2023-10-24", "LAL", "DEN")),
        )
        assert report["unmatched_games"] == 2
        assert (report["unretrieved_but_listed"]
                + report["absent_from_source"]) == report["unmatched_games"]

    def test_a_matched_game_is_in_neither_category(self):
        _, report = assemble(
            None, [source()], games(game()),
            index=self._index(("2023-10-24", "LAL", "DEN")),
        )
        assert report["matched_games"] == 1
        assert report["unretrieved_but_listed"] == 0
        assert report["absent_from_source"] == 0

    def test_no_repair_is_offered_when_nothing_is_retryable(self):
        _, report = assemble(
            None, [source()], games(game()),
            index=self._index(("2023-10-24", "LAL", "DEN")),
        )
        assert report["repair_command"] is None
