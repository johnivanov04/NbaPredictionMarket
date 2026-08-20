"""Game-phase classification.

The bug this guards against: ``postseason == False`` was treated as "regular
season", which admitted the six 2026 Play-In games (and would have admitted the
NBA Cup final) into the primary modelling set.
"""

from __future__ import annotations

from datetime import date

import pytest

from nba_prediction_market.ingestion.game_phase import (
    EXPECTED_REGULAR_SEASON_GAMES,
    GAME_PHASES,
    PHASE_NBA_CUP_CHAMPIONSHIP,
    PHASE_PLAY_IN,
    PHASE_PLAYOFFS,
    PHASE_REGULAR_SEASON,
    PHASE_UNCLASSIFIED,
    REGULAR_SEASON_GAMES_PER_TEAM,
    SEASON_PHASE_BOUNDARIES,
    SeasonPhaseBoundaries,
    classify_game_phase,
    phase_boundaries,
    verify_regular_season,
)


def classify(day: str, *, postseason=False, ist_stage=None, season=2025) -> str:
    return classify_game_phase(
        game_date=date.fromisoformat(day),
        postseason=postseason,
        ist_stage=ist_stage,
        season=season,
    )


# --- the required boundary cases -------------------------------------------


def test_2026_04_12_is_the_last_regular_season_day() -> None:
    """The 2025-26 regular season ended 2026-04-12."""
    assert classify("2026-04-12") == PHASE_REGULAR_SEASON


@pytest.mark.parametrize("day", ["2026-04-14", "2026-04-15", "2026-04-16", "2026-04-17"])
def test_play_in_window_classifies_as_play_in(day: str) -> None:
    assert classify(day) == PHASE_PLAY_IN


@pytest.mark.parametrize("day", ["2026-04-18", "2026-05-01", "2026-06-13"])
def test_postseason_games_classify_as_playoffs(day: str) -> None:
    assert classify(day, postseason=True) == PHASE_PLAYOFFS


def test_playoff_flag_wins_regardless_of_date() -> None:
    """The API's own flag is the most authoritative playoff signal."""
    assert classify("2026-04-15", postseason=True) == PHASE_PLAYOFFS


def test_the_day_between_season_end_and_play_in_is_not_regular_season() -> None:
    """2026-04-13 sits in the gap; it must not fall through to regular season."""
    assert classify("2026-04-13") == PHASE_UNCLASSIFIED


@pytest.mark.parametrize("day", ["2025-10-21", "2025-12-25", "2026-01-15", "2026-04-10"])
def test_ordinary_in_season_dates_are_regular_season(day: str) -> None:
    assert classify(day) == PHASE_REGULAR_SEASON


# --- NBA Cup ---------------------------------------------------------------


def test_nba_cup_final_is_not_a_regular_season_game() -> None:
    """SAS at NYK, 2025-12-16: postseason=False but excluded from the 82."""
    assert classify("2025-12-16", ist_stage="Championship") == PHASE_NBA_CUP_CHAMPIONSHIP


@pytest.mark.parametrize(
    "stage",
    ["East Group A", "West Group C", "East Quarterfinal", "West Semifinal"],
)
def test_other_nba_cup_games_do_count_as_regular_season(stage: str) -> None:
    """Only the final is excluded; group and knockout games count."""
    assert classify("2025-11-28", ist_stage=stage) == PHASE_REGULAR_SEASON


def test_championship_detection_tolerates_whitespace() -> None:
    assert classify("2025-12-16", ist_stage=" Championship ") == PHASE_NBA_CUP_CHAMPIONSHIP


def test_championship_flag_beats_the_play_in_window() -> None:
    assert classify("2026-04-15", ist_stage="Championship") == PHASE_NBA_CUP_CHAMPIONSHIP


# --- never silently regular season -----------------------------------------


def test_undeclared_season_is_unclassified_not_regular_season() -> None:
    """A new season must be declared deliberately, not guessed."""
    assert 2030 not in SEASON_PHASE_BOUNDARIES
    assert classify("2031-01-15", season=2030) == PHASE_UNCLASSIFIED
    assert phase_boundaries(2030) is None


@pytest.mark.parametrize(("day", "season"), [(None, 2025), ("2026-01-15", None)])
def test_missing_inputs_are_unclassified(day, season) -> None:
    result = classify_game_phase(
        game_date=date.fromisoformat(day) if day else None,
        postseason=False, ist_stage=None, season=season,
    )
    assert result == PHASE_UNCLASSIFIED


def test_a_date_after_the_playoffs_start_is_not_regular_season() -> None:
    assert classify("2026-05-20") == PHASE_UNCLASSIFIED


def test_every_phase_is_a_known_value() -> None:
    for day in ("2025-10-21", "2026-04-15", "2026-04-13", "2026-04-20"):
        assert classify(day) in GAME_PHASES
    assert classify("2026-04-20", postseason=True) in GAME_PHASES


# --- boundary table integrity ----------------------------------------------


def test_2025_boundaries_match_the_published_calendar() -> None:
    b = phase_boundaries(2025)
    assert b.regular_season_end == date(2026, 4, 12)
    assert b.play_in_start == date(2026, 4, 14)
    assert b.play_in_end == date(2026, 4, 17)
    assert b.playoffs_start == date(2026, 4, 18)
    assert b.source, "every declared season needs a stated provenance"


@pytest.mark.parametrize("season", sorted(SEASON_PHASE_BOUNDARIES))
def test_declared_boundaries_are_internally_ordered(season: int) -> None:
    b = SEASON_PHASE_BOUNDARIES[season]
    assert b.regular_season_end < b.play_in_start <= b.play_in_end < b.playoffs_start


@pytest.mark.parametrize(
    ("end", "pi_start", "pi_end", "po_start"),
    [
        ((2026, 4, 14), (2026, 4, 14), (2026, 4, 17), (2026, 4, 18)),  # overlap
        ((2026, 4, 12), (2026, 4, 17), (2026, 4, 14), (2026, 4, 18)),  # inverted
        ((2026, 4, 12), (2026, 4, 14), (2026, 4, 18), (2026, 4, 18)),  # overlap
    ],
)
def test_incoherent_boundaries_are_rejected_at_construction(
    end, pi_start, pi_end, po_start
) -> None:
    with pytest.raises(ValueError):
        SeasonPhaseBoundaries(
            regular_season_end=date(*end), play_in_start=date(*pi_start),
            play_in_end=date(*pi_end), playoffs_start=date(*po_start), source="test",
        )


# --- the 82-game auditor ---------------------------------------------------


def synthetic_regular_season() -> list[dict]:
    """A structurally valid season: 30 teams, 82 games each, 1230 games."""
    teams = [f"T{i:02d}" for i in range(30)]
    games, counts = [], dict.fromkeys(teams, 0)
    for home in teams:
        for away in teams:
            if home == away:
                continue
            if counts[home] >= 82 or counts[away] >= 82:
                continue
            games.append({"game_phase": PHASE_REGULAR_SEASON,
                          "home_team_code": home, "visitor_team_code": away})
            counts[home] += 1
            counts[away] += 1
    # Top up any team short of 82 by pairing the remaining shortfalls.
    while True:
        short = [t for t in teams if counts[t] < 82]
        if len(short) < 2:
            break
        a, b = short[0], short[1]
        games.append({"game_phase": PHASE_REGULAR_SEASON,
                      "home_team_code": a, "visitor_team_code": b})
        counts[a] += 1
        counts[b] += 1
    return games


def test_expected_game_count_is_the_league_arithmetic() -> None:
    assert EXPECTED_REGULAR_SEASON_GAMES == 30 * REGULAR_SEASON_GAMES_PER_TEAM // 2 == 1230


def test_audit_passes_on_a_structurally_valid_season() -> None:
    audit = verify_regular_season(synthetic_regular_season(), 2025)
    assert audit["verified"] is True
    assert audit["regular_season_games"] == 1230
    assert audit["teams"] == 30
    assert audit["teams_with_unexpected_game_count"] == {}


def test_audit_fails_when_extra_games_slip_in() -> None:
    """This is what a wrong play-in boundary would look like."""
    games = synthetic_regular_season()
    games.append({"game_phase": PHASE_REGULAR_SEASON,
                  "home_team_code": "T00", "visitor_team_code": "T01"})
    audit = verify_regular_season(games, 2025)

    assert audit["verified"] is False
    assert audit["regular_season_games"] == 1231
    assert audit["teams_with_unexpected_game_count"] == {"T00": 83, "T01": 83}


def test_audit_ignores_non_regular_season_phases() -> None:
    games = [
        *synthetic_regular_season(),
        {"game_phase": PHASE_PLAY_IN, "home_team_code": "T00", "visitor_team_code": "T01"},
        {"game_phase": PHASE_PLAYOFFS, "home_team_code": "T00", "visitor_team_code": "T02"},
        {"game_phase": PHASE_NBA_CUP_CHAMPIONSHIP,
         "home_team_code": "T03", "visitor_team_code": "T04"},
    ]
    audit = verify_regular_season(games, 2025)

    assert audit["verified"] is True
    assert audit["regular_season_games"] == 1230
    assert audit["phase_counts"][PHASE_PLAY_IN] == 1
    assert audit["phase_counts"][PHASE_NBA_CUP_CHAMPIONSHIP] == 1


def test_audit_reports_undeclared_boundaries() -> None:
    audit = verify_regular_season([], 2030)
    assert audit["boundaries_declared"] is False
    assert audit["verified"] is False
