"""Per-season structure declarations and their invariants.

The bug class guarded against: assuming every season is 30x82/2 = 1,230 games.
Three seasons in the historical range are not.
"""

from __future__ import annotations

from datetime import date

import pytest

from nba_prediction_market.ingestion.season_metadata import (
    FIRST_NBA_CUP_SEASON,
    FIRST_PLAY_IN_SEASON,
    HISTORICAL_SEASONS,
    NBA_TEAM_COUNT,
    SEASON_METADATA,
    STANDARD_GAMES_PER_TEAM,
    STANDARD_REGULAR_SEASON_GAMES,
    STRUCTURE_INTERRUPTED,
    STRUCTURE_SHORTENED,
    STRUCTURE_STANDARD,
    SeasonInfo,
    UnknownSeasonError,
    require_season_info,
    season_has_nba_cup,
    season_has_play_in,
    season_info,
)


def test_historical_range_is_twenty_seasons() -> None:
    assert tuple(range(2006, 2026)) == HISTORICAL_SEASONS
    assert len(HISTORICAL_SEASONS) == 20


@pytest.mark.parametrize("season", HISTORICAL_SEASONS)
def test_every_historical_season_is_declared(season: int) -> None:
    info = season_info(season)
    assert info is not None, f"season {season} has no declared metadata"
    assert info.season == season
    assert info.structure in {STRUCTURE_STANDARD, STRUCTURE_SHORTENED, STRUCTURE_INTERRUPTED}


@pytest.mark.parametrize("season", HISTORICAL_SEASONS)
def test_declared_windows_are_coherent(season: int) -> None:
    info = SEASON_METADATA[season]
    assert info.regular_season_start < info.regular_season_end
    if info.has_play_in:
        assert info.regular_season_end < info.play_in_start <= info.play_in_end
        if info.playoffs_start:
            assert info.play_in_end < info.playoffs_start


@pytest.mark.parametrize("season", HISTORICAL_SEASONS)
def test_season_labels_are_correct(season: int) -> None:
    assert SEASON_METADATA[season].label == f"{season}-{(season + 1) % 100:02d}"


def test_standard_seasons_expect_1230_games() -> None:
    standard = [s for s in HISTORICAL_SEASONS if SEASON_METADATA[s].structure == STRUCTURE_STANDARD]
    assert len(standard) == 16
    for season in standard:
        info = SEASON_METADATA[season]
        assert info.expected_regular_season_games == STANDARD_REGULAR_SEASON_GAMES == 1230
        assert info.expected_games_per_team == STANDARD_GAMES_PER_TEAM == 82


# --- the three non-standard seasons ----------------------------------------


def test_2011_lockout_season_is_66_games() -> None:
    info = SEASON_METADATA[2011]
    assert info.structure == STRUCTURE_SHORTENED
    assert info.unusual_reason == "lockout"
    assert info.expected_games_per_team == 66
    assert info.expected_regular_season_games == 990
    assert info.regular_season_start == date(2011, 12, 25)
    assert not info.has_play_in, "no play-in existed in 2011-12"


def test_2019_covid_season_asserts_no_uniform_structure() -> None:
    """22 bubble teams played 8 seeding games; teams finished on different counts."""
    info = SEASON_METADATA[2019]
    assert info.structure == STRUCTURE_INTERRUPTED
    assert info.unusual_reason == "covid_suspension_and_bubble_restart"
    assert info.expected_regular_season_games is None
    assert info.expected_games_per_team is None
    assert info.regular_season_end == date(2020, 8, 14)
    assert info.has_play_in, "2019-20 had a one-off West 8/9 play-in"


def test_2020_shortened_season_is_72_games() -> None:
    info = SEASON_METADATA[2020]
    assert info.structure == STRUCTURE_SHORTENED
    assert info.unusual_reason == "covid_shortened_schedule"
    assert info.expected_games_per_team == 72
    assert info.expected_regular_season_games == 1080
    assert info.has_play_in


def test_non_standard_seasons_are_exactly_the_four_known_ones() -> None:
    unusual = {s for s in HISTORICAL_SEASONS if SEASON_METADATA[s].structure != STRUCTURE_STANDARD}
    assert unusual == {2011, 2012, 2019, 2020}


def test_2012_season_lost_one_game_to_a_cancellation() -> None:
    """BOS v IND (2013-04-16) was cancelled after the Boston Marathon bombing."""
    info = SEASON_METADATA[2012]
    assert info.structure == STRUCTURE_INTERRUPTED
    assert info.unusual_reason == "cancelled_game_boston_marathon_bombing"
    assert info.expected_regular_season_games == 1229
    assert info.expected_games_per_team == 82
    assert info.known_team_game_count_exceptions == {"BOS": 81, "IND": 81}


def test_the_cancelled_game_exception_is_honoured_by_the_audit() -> None:
    """BOS and IND at 81 must pass; a third team at 81 must not."""
    from nba_prediction_market.ingestion.game_phase import verify_regular_season

    codes = [f"T{i:02d}" for i in range(28)] + ["BOS", "IND"]
    need = {c: (81 if c in {"BOS", "IND"} else 82) for c in codes}
    games = []
    # Greedy pairing of the two teams with the most games still needed.
    while True:
        remaining = sorted((n, c) for c, n in need.items() if n > 0)
        if len(remaining) < 2:
            break
        home, away = remaining[-1][1], remaining[-2][1]
        games.append({"game_phase": "regular_season",
                      "home_team_code": home, "visitor_team_code": away})
        need[home] -= 1
        need[away] -= 1

    assert len(games) == 1229
    audit = verify_regular_season(games, 2012)
    assert audit["regular_season_games"] == 1229
    assert audit["teams_with_unexpected_game_count"] == {}
    assert audit["games_per_team_expected"] == 82


def test_standard_season_count_excludes_2012() -> None:
    standard = [s for s in HISTORICAL_SEASONS if SEASON_METADATA[s].structure == STRUCTURE_STANDARD]
    assert 2012 not in standard


@pytest.mark.parametrize("season", [2011, 2012, 2019, 2020])
def test_non_standard_seasons_state_a_reason_and_notes(season: int) -> None:
    info = SEASON_METADATA[season]
    assert info.unusual_reason
    assert info.notes, "an unusual season must document its evidence"


# --- era gating ------------------------------------------------------------


@pytest.mark.parametrize("season", range(2006, 2019))
def test_no_play_in_before_2019(season: int) -> None:
    assert not season_has_play_in(season)
    assert SEASON_METADATA[season].play_in_start is None


@pytest.mark.parametrize("season", range(2019, 2026))
def test_play_in_declared_from_2019(season: int) -> None:
    assert season_has_play_in(season)
    assert FIRST_PLAY_IN_SEASON == 2019


@pytest.mark.parametrize("season", range(2006, 2023))
def test_no_nba_cup_before_2023(season: int) -> None:
    assert not season_has_nba_cup(season)


@pytest.mark.parametrize("season", [2023, 2024, 2025])
def test_nba_cup_from_2023(season: int) -> None:
    assert season_has_nba_cup(season)
    assert FIRST_NBA_CUP_SEASON == 2023


# --- guardrails ------------------------------------------------------------


def test_unknown_season_raises_an_actionable_error() -> None:
    assert season_info(1999) is None
    with pytest.raises(UnknownSeasonError, match="SEASON_METADATA"):
        require_season_info(1999)


def test_inverted_regular_season_window_is_rejected() -> None:
    with pytest.raises(ValueError, match="inverted"):
        SeasonInfo(
            season=2025, structure=STRUCTURE_STANDARD,
            regular_season_start=date(2026, 4, 12), regular_season_end=date(2025, 10, 21),
        )


def test_half_declared_play_in_window_is_rejected() -> None:
    with pytest.raises(ValueError, match="half-declared"):
        SeasonInfo(
            season=2025, structure=STRUCTURE_STANDARD,
            regular_season_start=date(2025, 10, 21), regular_season_end=date(2026, 4, 12),
            play_in_start=date(2026, 4, 14),
        )


def test_unusual_season_without_a_reason_is_rejected() -> None:
    with pytest.raises(ValueError, match="must state a reason"):
        SeasonInfo(
            season=2011, structure=STRUCTURE_SHORTENED,
            regular_season_start=date(2011, 12, 25), regular_season_end=date(2012, 4, 26),
        )


def test_standard_game_arithmetic() -> None:
    assert STANDARD_REGULAR_SEASON_GAMES == NBA_TEAM_COUNT * STANDARD_GAMES_PER_TEAM // 2 == 1230


@pytest.mark.parametrize("season", HISTORICAL_SEASONS)
def test_metadata_serialises_for_the_report(season: int) -> None:
    d = SEASON_METADATA[season].to_dict()
    assert d["season"] == season
    assert isinstance(d["regular_season_start"], str)
    assert set(d) >= {
        "season", "structure", "regular_season_start", "regular_season_end",
        "expected_regular_season_games", "expected_games_per_team", "unusual_reason", "notes",
    }


# --- NBA Cup final identification -----------------------------------------


@pytest.mark.parametrize(
    ("season", "expected"),
    [(2023, date(2023, 12, 9)), (2024, date(2024, 12, 17)), (2025, date(2025, 12, 16))],
)
def test_cup_final_dates_are_declared_for_every_cup_season(season, expected) -> None:
    """ist_stage is only populated for 2025-26, so the date is the fallback."""
    assert SEASON_METADATA[season].nba_cup_final_date == expected


@pytest.mark.parametrize("season", range(2006, 2023))
def test_no_cup_final_declared_before_the_cup_existed(season: int) -> None:
    assert SEASON_METADATA[season].nba_cup_final_date is None


def test_a_cup_final_before_2023_is_rejected() -> None:
    with pytest.raises(ValueError, match="did not exist before"):
        SeasonInfo(
            season=2010, structure=STRUCTURE_STANDARD,
            regular_season_start=date(2010, 10, 26), regular_season_end=date(2011, 4, 13),
            nba_cup_final_date=date(2010, 12, 9),
        )


def test_a_cup_final_outside_the_season_window_is_rejected() -> None:
    with pytest.raises(ValueError, match="outside the season window"):
        SeasonInfo(
            season=2023, structure=STRUCTURE_STANDARD,
            regular_season_start=date(2023, 10, 24), regular_season_end=date(2024, 4, 14),
            nba_cup_final_date=date(2024, 7, 1),
        )
