"""Normalization of BALLDONTLIE games, season verification, and de-duplication."""

from __future__ import annotations

from datetime import UTC, date, datetime

import pytest
from tests.conftest import bdl_game

from nba_prediction_market.ingestion.game_phase import (
    PHASE_NBA_CUP_CHAMPIONSHIP,
    PHASE_PLAY_IN,
    PHASE_PLAYOFFS,
    PHASE_REGULAR_SEASON,
)
from nba_prediction_market.ingestion.nba_games import (
    NBA_GAME_COLUMNS,
    SeasonVerificationError,
    build_games_frame,
    normalize_game,
    parse_utc_datetime,
    verify_season,
)


def test_normalizes_a_completed_game() -> None:
    row = normalize_game(bdl_game())

    assert row["source"] == "balldontlie"
    assert row["source_game_id"] == 18446819
    assert row["season"] == 2025
    assert row["season_label"] == "2025-26"
    assert row["game_date"] == date(2025, 10, 21)
    assert row["tipoff_utc"] == datetime(2025, 10, 21, 23, 30, tzinfo=UTC)
    assert row["is_final"] is True
    assert row["period"] == 6
    assert row["postseason"] is False
    assert row["home_team_code"] == "OKC"
    assert row["visitor_team_code"] == "HOU"
    assert row["home_score"] == 125
    assert row["visitor_score"] == 124
    assert row["home_win"] is True
    assert row["matchup_key"] == "2025-10-21|HOU|OKC"


def test_matchup_key_is_orientation_independent() -> None:
    """Swapping home/away yields the same key, so the join never depends on it."""
    home_first = normalize_game(bdl_game())
    swapped = normalize_game(
        bdl_game(
            home_team=bdl_game()["visitor_team"],
            visitor_team=bdl_game()["home_team"],
        )
    )
    assert home_first["matchup_key"] == swapped["matchup_key"]


def test_home_win_is_false_when_the_visitor_wins() -> None:
    row = normalize_game(bdl_game(home_team_score=100, visitor_team_score=110))
    assert row["home_win"] is False


@pytest.mark.parametrize(
    "overrides",
    [
        # Not finished -- a leading team is not a winner.
        {"status": "3rd Qtr", "status_state": "in_progress", "home_team_score": 80,
         "visitor_team_score": 70},
        # Scheduled, no scores at all.
        {"status": "2025-10-21T23:30:00Z", "status_state": "scheduled",
         "home_team_score": None, "visitor_team_score": None},
        # Final but level: no winner is established, so none is invented.
        {"home_team_score": 110, "visitor_team_score": 110},
        # Final but a score is missing.
        {"home_team_score": 110, "visitor_team_score": None},
    ],
)
def test_home_win_is_never_inferred(overrides: dict) -> None:
    assert normalize_game(bdl_game(**overrides))["home_win"] is None


def test_is_final_accepts_either_status_field() -> None:
    assert normalize_game(bdl_game(status_state=None, status="Final"))["is_final"] is True
    assert normalize_game(bdl_game(status_state="final", status=None))["is_final"] is True
    assert normalize_game(bdl_game(status_state="scheduled", status="7:30 PM"))["is_final"] is False


def test_postseason_games_are_kept_and_flagged() -> None:
    row = normalize_game(bdl_game(postseason=True, date="2026-06-13"))
    assert row["postseason"] is True
    assert row["game_date"] == date(2026, 6, 13)


def test_missing_datetime_leaves_tipoff_null_without_failing() -> None:
    row = normalize_game(bdl_game(datetime=None))
    assert row["tipoff_utc"] is None
    assert row["game_date"] == date(2025, 10, 21)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("2025-10-21T23:30:00.000Z", datetime(2025, 10, 21, 23, 30, tzinfo=UTC)),
        ("2025-10-21T23:30:00Z", datetime(2025, 10, 21, 23, 30, tzinfo=UTC)),
        ("2025-10-21T23:30:00+00:00", datetime(2025, 10, 21, 23, 30, tzinfo=UTC)),
        # Offsets are converted to UTC rather than kept local.
        ("2025-10-21T19:30:00-04:00", datetime(2025, 10, 21, 23, 30, tzinfo=UTC)),
        # Naive input is assumed to already be UTC.
        ("2025-10-21T23:30:00", datetime(2025, 10, 21, 23, 30, tzinfo=UTC)),
        (None, None),
        ("", None),
        ("not a timestamp", None),
    ],
)
def test_parse_utc_datetime(raw: str | None, expected: datetime | None) -> None:
    assert parse_utc_datetime(raw) == expected


def test_build_frame_has_the_documented_schema(raw_games: list[dict]) -> None:
    frame = build_games_frame(raw_games, 2025)
    assert list(frame.columns) == NBA_GAME_COLUMNS
    assert len(frame) == 2


def test_build_frame_deduplicates_repeated_game_ids(raw_games: list[dict]) -> None:
    """Overlapping pages must not double-count a game."""
    frame = build_games_frame([*raw_games, raw_games[0], raw_games[1]], 2025)
    assert len(frame) == 2
    assert frame["source_game_id"].is_unique


def test_build_frame_sorts_by_date_then_id() -> None:
    games = [
        bdl_game(id=3, date="2025-12-25"),
        bdl_game(id=1, date="2025-10-21"),
        bdl_game(id=2, date="2025-10-21"),
    ]
    frame = build_games_frame(games, 2025)
    assert list(frame["source_game_id"]) == [1, 2, 3]


def test_build_frame_rejects_an_unmappable_team() -> None:
    """An unknown franchise must stop the run, not be silently dropped or guessed."""
    unknown = bdl_game()
    unknown["home_team"] = {"id": 99, "abbreviation": "SEA", "full_name": "Seattle SuperSonics"}
    with pytest.raises(ValueError, match="not in the canonical map"):
        build_games_frame([unknown], 2025)


def test_build_frame_rejects_games_without_an_id() -> None:
    with pytest.raises(ValueError, match="without an id"):
        build_games_frame([bdl_game(id=None)], 2025)


# --- season verification ---------------------------------------------------


def test_verify_season_accepts_a_real_2025_26_schedule() -> None:
    games = [
        normalize_game(bdl_game(id=1, date="2025-10-21")),
        normalize_game(bdl_game(id=2, date="2026-06-13")),
    ]
    result = verify_season(games, 2025)
    assert result == {
        "season": 2025,
        "season_label": "2025-26",
        "game_count": 2,
        "first_game_date": "2025-10-21",
        "last_game_date": "2026-06-13",
        "verified": True,
    }


def test_verify_season_rejects_a_mislabelled_season() -> None:
    games = [normalize_game(bdl_game(season=2024))]
    with pytest.raises(SeasonVerificationError, match="expected every game to carry season=2025"):
        verify_season(games, 2025)


def test_verify_season_rejects_dates_outside_the_window() -> None:
    """Season 2025 must be the 2025-26 campaign, not 2024-25."""
    games = [
        normalize_game(bdl_game(id=1, date="2025-01-15")),
        normalize_game(bdl_game(id=2, date="2025-04-10")),
    ]
    with pytest.raises(SeasonVerificationError, match="outside the 2025-26 window"):
        verify_season(games, 2025)


def test_verify_season_rejects_a_schedule_that_does_not_start_in_autumn() -> None:
    games = [normalize_game(bdl_game(id=1, date="2026-01-05"))]
    with pytest.raises(SeasonVerificationError, match="start of 2025-26"):
        verify_season(games, 2025)


def test_verify_season_rejects_an_empty_pull() -> None:
    with pytest.raises(SeasonVerificationError, match="No parseable game dates"):
        verify_season([], 2025)


# --- game phase on normalized games ---------------------------------------


def test_normalized_game_carries_an_explicit_phase() -> None:
    row = normalize_game(bdl_game())
    assert row["game_phase"] == PHASE_REGULAR_SEASON
    assert row["ist_stage"] is None


def test_play_in_games_are_not_labelled_regular_season() -> None:
    """The 2026 play-in games carry postseason=False and no ist_stage."""
    row = normalize_game(bdl_game(id=21681576, date="2026-04-14", postseason=False))
    assert row["postseason"] is False
    assert row["game_phase"] == PHASE_PLAY_IN


def test_last_regular_season_day_is_still_regular_season() -> None:
    row = normalize_game(bdl_game(date="2026-04-12"))
    assert row["game_phase"] == PHASE_REGULAR_SEASON


def test_playoff_games_are_labelled_playoffs() -> None:
    row = normalize_game(bdl_game(date="2026-04-18", postseason=True))
    assert row["game_phase"] == PHASE_PLAYOFFS


def test_nba_cup_final_is_labelled_and_kept() -> None:
    row = normalize_game(bdl_game(date="2025-12-16", ist_stage="Championship"))
    assert row["ist_stage"] == "Championship"
    assert row["game_phase"] == PHASE_NBA_CUP_CHAMPIONSHIP
    assert row["postseason"] is False


def test_nba_cup_group_games_stay_regular_season() -> None:
    row = normalize_game(bdl_game(date="2025-11-28", ist_stage="East Group A"))
    assert row["game_phase"] == PHASE_REGULAR_SEASON


def test_all_phases_survive_into_the_frame() -> None:
    """Play-in and the cup final are preserved, never dropped."""
    games = [
        bdl_game(id=1, date="2025-10-21"),
        bdl_game(id=2, date="2025-12-16", ist_stage="Championship"),
        bdl_game(id=3, date="2026-04-14"),
        bdl_game(id=4, date="2026-04-18", postseason=True),
    ]
    frame = build_games_frame(games, 2025)

    assert len(frame) == 4
    assert list(frame["game_phase"]) == [
        PHASE_REGULAR_SEASON, PHASE_NBA_CUP_CHAMPIONSHIP, PHASE_PLAY_IN, PHASE_PLAYOFFS
    ]
    assert "ist_stage" in frame.columns
