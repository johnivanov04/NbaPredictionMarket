"""Game-phase classification.

The bug this guards against: ``postseason == False`` was treated as "regular
season", which admitted the six 2026 Play-In games (and would have admitted the
NBA Cup final) into the primary modelling set.
"""

from __future__ import annotations

from datetime import date, timedelta

import pytest

from nba_prediction_market.ingestion.game_phase import (
    EXPECTED_REGULAR_SEASON_GAMES,
    GAME_PHASES,
    PHASE_NBA_CUP_CHAMPIONSHIP,
    PHASE_OTHER_SPECIAL,
    PHASE_PLAY_IN,
    PHASE_PLAYOFFS,
    PHASE_REGULAR_SEASON,
    PHASE_UNCLASSIFIED,
    REGULAR_SEASON_GAMES_PER_TEAM,
    classify_game_phase,
    verify_regular_season,
)
from nba_prediction_market.ingestion.season_metadata import (
    SEASON_METADATA,
    SeasonInfo,
    season_info,
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


def test_playoff_flag_decides_outside_the_play_in_window() -> None:
    """The flag is authoritative everywhere except play-in dates, where it is
    demonstrably inconsistent upstream (see the precedence tests below)."""
    assert classify("2026-04-20", postseason=True) == PHASE_PLAYOFFS
    assert classify("2026-01-15", postseason=True) == PHASE_PLAYOFFS


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


def test_cup_final_and_play_in_windows_cannot_overlap() -> None:
    """The Cup final is in December and the play-in in April, so the two rules
    never compete on a real game. SeasonInfo enforces the Cup final sits inside
    the regular-season window, which ends before the play-in begins."""
    info = season_info(2025)
    assert info.nba_cup_final_date < info.regular_season_end < info.play_in_start
    assert classify("2025-12-16", ist_stage="Championship") == PHASE_NBA_CUP_CHAMPIONSHIP
    assert classify("2025-12-16") == PHASE_NBA_CUP_CHAMPIONSHIP


# --- never silently regular season -----------------------------------------


def test_undeclared_season_is_unclassified_not_regular_season() -> None:
    """A new season must be declared deliberately, not guessed."""
    assert 2030 not in SEASON_METADATA
    assert classify("2031-01-15", season=2030) == PHASE_UNCLASSIFIED
    assert season_info(2030) is None


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
    b = season_info(2025)
    assert b.regular_season_end == date(2026, 4, 12)
    assert b.play_in_start == date(2026, 4, 14)
    assert b.play_in_end == date(2026, 4, 17)
    assert b.playoffs_start == date(2026, 4, 18)
    assert b.notes, "every declared season needs stated provenance"


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
        SeasonInfo(
            season=2025, structure="standard",
            regular_season_start=date(2025, 10, 21), regular_season_end=date(*end),
            play_in_start=date(*pi_start), play_in_end=date(*pi_end),
            playoffs_start=date(*po_start),
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
    assert audit["checks"]["metadata_declared"] is False


# --- era awareness: modern rules must not be projected onto old seasons ----


@pytest.mark.parametrize("season", range(2006, 2019))
def test_no_game_can_be_classified_play_in_before_2019(season: int) -> None:
    """Play-in did not exist; mid-April games are ordinary regular-season games."""
    info = season_info(season)
    result = classify_game_phase(
        game_date=info.regular_season_end, postseason=False, ist_stage=None, season=season
    )
    assert result == PHASE_REGULAR_SEASON
    # A date after the regular season is unclassified, never play_in.
    after = classify_game_phase(
        game_date=date(info.regular_season_end.year, info.regular_season_end.month,
                       info.regular_season_end.day) + timedelta(days=2),
        postseason=False, ist_stage=None, season=season,
    )
    assert after != PHASE_PLAY_IN
    assert after == PHASE_UNCLASSIFIED


@pytest.mark.parametrize("season", [2019, 2020, 2021, 2022, 2023, 2024, 2025])
def test_play_in_window_classifies_only_for_seasons_that_had_one(season: int) -> None:
    info = season_info(season)
    assert classify_game_phase(
        game_date=info.play_in_start, postseason=False, ist_stage=None, season=season
    ) == PHASE_PLAY_IN


def test_2011_lockout_season_start_is_regular_season() -> None:
    """The lockout season opened on Christmas Day 2011."""
    assert classify_game_phase(
        game_date=date(2011, 12, 25), postseason=False, ist_stage=None, season=2011
    ) == PHASE_REGULAR_SEASON


def test_2019_bubble_seeding_games_are_regular_season() -> None:
    """August 2020 seeding games counted toward the regular season."""
    assert classify_game_phase(
        game_date=date(2020, 8, 10), postseason=False, ist_stage=None, season=2019
    ) == PHASE_REGULAR_SEASON


def test_2019_one_off_play_in_is_classified() -> None:
    assert classify_game_phase(
        game_date=date(2020, 8, 15), postseason=False, ist_stage=None, season=2019
    ) == PHASE_PLAY_IN


def test_2020_season_starting_in_december_is_regular_season() -> None:
    assert classify_game_phase(
        game_date=date(2020, 12, 22), postseason=False, ist_stage=None, season=2020
    ) == PHASE_REGULAR_SEASON


def test_a_2006_date_in_a_2025_play_in_window_is_not_play_in() -> None:
    """Windows are per season, not global."""
    assert classify_game_phase(
        game_date=date(2007, 4, 15), postseason=False, ist_stage=None, season=2006
    ) == PHASE_REGULAR_SEASON


# --- non-NBA opponents -----------------------------------------------------


@pytest.mark.parametrize("bad_id", [2844, 5193, 37, 216597])
def test_a_game_against_a_non_franchise_is_other_special(bad_id: int) -> None:
    """Exhibitions vs international/defunct clubs are not NBA games."""
    assert classify_game_phase(
        game_date=date(2025, 10, 13), postseason=False, ist_stage=None, season=2025,
        home_team_source_id=18, visitor_team_source_id=bad_id,
    ) == PHASE_OTHER_SPECIAL
    assert classify_game_phase(
        game_date=date(2025, 10, 13), postseason=False, ist_stage=None, season=2025,
        home_team_source_id=bad_id, visitor_team_source_id=18,
    ) == PHASE_OTHER_SPECIAL


def test_two_real_franchises_are_unaffected_by_the_id_check() -> None:
    assert classify_game_phase(
        game_date=date(2026, 1, 15), postseason=False, ist_stage=None, season=2025,
        home_team_source_id=21, visitor_team_source_id=11,
    ) == PHASE_REGULAR_SEASON


def test_team_ids_are_optional() -> None:
    """Existing callers that omit ids keep working."""
    assert classify_game_phase(
        game_date=date(2026, 1, 15), postseason=False, ist_stage=None, season=2025
    ) == PHASE_REGULAR_SEASON


# --- shortened-season validation ------------------------------------------


def _season_games(n: int, teams: int = 30) -> list[dict]:
    """n games spread evenly so each team gets the same count."""
    codes = [f"T{i:02d}" for i in range(teams)]
    games, counts = [], dict.fromkeys(codes, 0)
    i = 0
    while len(games) < n:
        home, away = codes[i % teams], codes[(i + 1 + (i // teams)) % teams]
        if home != away:
            games.append({"game_phase": PHASE_REGULAR_SEASON,
                          "home_team_code": home, "visitor_team_code": away})
            counts[home] += 1
            counts[away] += 1
        i += 1
    return games


def test_shortened_season_is_validated_against_its_own_total() -> None:
    """990, not 1230, is correct for the lockout season."""
    audit = verify_regular_season(_season_games(990), 2011)
    assert audit["expected_regular_season_games"] == 990
    assert audit["games_per_team_expected"] == 66
    assert audit["regular_season_games"] == 990
    assert audit["checks"]["total_matches_expected"] is True
    assert audit["structure"] == "shortened"


def test_a_1230_game_lockout_season_would_fail_validation() -> None:
    """The old blanket invariant would have passed this; the season-aware one must not."""
    audit = verify_regular_season(_season_games(1230), 2011)
    assert audit["checks"]["total_matches_expected"] is False
    assert audit["verified"] is False


def test_interrupted_season_skips_the_uniform_checks_but_still_audits() -> None:
    """2019-20 has no uniform structure, so no total is asserted -- but teams are."""
    audit = verify_regular_season(_season_games(1059), 2019)
    assert audit["expected_regular_season_games"] is None
    assert audit["games_per_team_expected"] is None
    assert audit["checks"]["total_matches_expected"] is True
    assert audit["checks"]["team_count_matches"] is True
    assert audit["teams"] == 30
    assert audit["games_per_team_min"] is not None
    assert audit["verified"] is True


def test_wrong_team_count_fails_even_for_an_irregular_season() -> None:
    audit = verify_regular_season(_season_games(500, teams=28), 2019)
    assert audit["checks"]["team_count_matches"] is False
    assert audit["verified"] is False


# --- Cup final identification without ist_stage ----------------------------


@pytest.mark.parametrize(
    ("season", "day"),
    [(2023, "2023-12-09"), (2024, "2024-12-17"), (2025, "2025-12-16")],
)
def test_cup_final_is_classified_from_the_declared_date_alone(season, day) -> None:
    """BALLDONTLIE leaves ist_stage null for 2023-24 and 2024-25."""
    assert classify(day, ist_stage=None, season=season) == PHASE_NBA_CUP_CHAMPIONSHIP


def test_the_declared_date_and_ist_stage_agree_for_2025() -> None:
    """2025-26 is the cross-check: both routes must reach the same answer."""
    by_field = classify("2025-12-16", ist_stage="Championship", season=2025)
    by_date = classify("2025-12-16", ist_stage=None, season=2025)
    assert by_field == by_date == PHASE_NBA_CUP_CHAMPIONSHIP


def test_a_normal_game_the_day_before_a_cup_final_is_regular_season() -> None:
    assert classify("2023-12-08", season=2023) == PHASE_REGULAR_SEASON
    assert classify("2023-12-11", season=2023) == PHASE_REGULAR_SEASON


def test_the_cup_final_date_does_not_leak_into_other_seasons() -> None:
    """A December 9 game in a pre-Cup season is an ordinary game."""
    assert classify("2006-12-09", season=2006) == PHASE_REGULAR_SEASON


# --- the postseason flag is unreliable for play-in games -------------------


@pytest.mark.parametrize(
    ("season", "day"),
    [
        (2019, "2020-08-15"),   # flagged postseason=True upstream
        (2020, "2021-05-18"),   # season contains BOTH flag values upstream
        (2021, "2022-04-12"),   # flagged postseason=True upstream
        (2022, "2023-04-11"),   # flagged postseason=False upstream
        (2025, "2026-04-14"),
    ],
)
def test_play_in_window_wins_over_the_postseason_flag(season: int, day: str) -> None:
    """BALLDONTLIE flags play-in games inconsistently; the window is exact."""
    assert classify(day, postseason=True, season=season) == PHASE_PLAY_IN
    assert classify(day, postseason=False, season=season) == PHASE_PLAY_IN


def test_a_real_playoff_game_after_the_window_is_still_playoffs() -> None:
    assert classify("2022-04-16", postseason=True, season=2021) == PHASE_PLAYOFFS
    assert classify("2026-04-18", postseason=True, season=2025) == PHASE_PLAYOFFS


def test_the_flag_still_decides_outside_a_play_in_window() -> None:
    """Only play-in dates are overridden; everything else trusts the flag."""
    assert classify("2011-04-20", postseason=True, season=2010) == PHASE_PLAYOFFS
    assert classify("2007-05-01", postseason=True, season=2006) == PHASE_PLAYOFFS


def test_seasons_without_a_play_in_never_override_the_flag() -> None:
    for season in (2006, 2011, 2018):
        info = season_info(season)
        assert info.play_in_start is None
        assert classify(str(info.regular_season_end), postseason=True,
                        season=season) == PHASE_PLAYOFFS
