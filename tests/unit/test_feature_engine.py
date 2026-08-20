"""Sequential feature engine: leakage safety above all else."""

from __future__ import annotations

import copy
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from nba_prediction_market.features.feature_engine import (
    FEATURE_COLUMNS,
    build_features,
)

BASE = datetime(2021, 10, 19, 23, 30, tzinfo=UTC)


def game(
    gid: int, day_offset: float, home: int, away: int, *, season: int = 2021,
    home_score: int = 110, away_score: int = 100, home_win: bool | None = None,
) -> dict[str, Any]:
    if home_win is None:
        home_win = home_score > away_score
    return {
        "nba_game_id": gid,
        "season": season,
        "game_datetime_utc": BASE + timedelta(days=day_offset),
        "home_franchise_id": home,
        "away_franchise_id": away,
        "home_team": f"T{home:02d}",
        "away_team": f"T{away:02d}",
        "home_score": home_score,
        "away_score": away_score,
        "home_win": home_win,
    }


def rows_by_id(games: list[dict]) -> dict[int, dict]:
    return {r["nba_game_id"]: r for r in build_features(games)}


def features_only(row: dict) -> dict:
    """Everything except the target, which legitimately changes with the result."""
    return {k: v for k, v in row.items() if k != "home_win"}


# --- schema and first game -------------------------------------------------


def test_output_has_the_documented_schema() -> None:
    rows = build_features([game(1, 0, 1, 2)])
    assert list(rows[0]) == list(FEATURE_COLUMNS)


def test_first_game_of_a_season_has_no_prior_state() -> None:
    row = build_features([game(1, 0, 1, 2)])[0]
    assert row["home_games_played"] == 0
    assert row["away_games_played"] == 0
    assert row["home_win_pct_prior"] is None
    assert row["home_last5_win_pct"] is None
    assert row["home_last10_point_diff"] is None
    assert row["win_pct_diff"] is None


def test_first_game_rest_is_null_and_not_a_back_to_back() -> None:
    """Never an offseason length -- rest is simply unknown."""
    row = build_features([game(1, 0, 1, 2)])[0]
    assert row["home_rest_days"] is None
    assert row["away_rest_days"] is None
    assert row["rest_days_diff"] is None
    assert row["home_back_to_back"] is False
    assert row["away_back_to_back"] is False


def test_second_game_sees_exactly_one_prior_game() -> None:
    rows = rows_by_id([game(1, 0, 1, 2, home_score=110, away_score=100), game(2, 2, 1, 3)])
    second = rows[2]
    assert second["home_games_played"] == 1
    assert second["home_win_pct_prior"] == 1.0
    assert second["home_last5_win_pct"] == 1.0
    assert second["home_last5_point_diff"] == 10.0
    assert second["away_games_played"] == 0, "team 3 has not played yet"
    assert second["away_win_pct_prior"] is None


# --- the core leakage properties -------------------------------------------


def test_a_games_own_result_never_reaches_its_own_features() -> None:
    """Requirement 1: flipping the current result must not change its row."""
    games = [game(1, 0, 1, 2, home_score=110, away_score=100), game(2, 2, 1, 3)]
    flipped = copy.deepcopy(games)
    flipped[1]["home_score"], flipped[1]["away_score"] = 100, 110
    flipped[1]["home_win"] = False

    assert features_only(rows_by_id(games)[2]) == features_only(rows_by_id(flipped)[2])


def test_a_future_result_never_reaches_an_earlier_game() -> None:
    """Requirement 2: later games cannot inform earlier ones."""
    games = [game(i, i * 2, 1, 2 + i) for i in range(1, 6)]
    flipped = copy.deepcopy(games)
    for g in flipped[2:]:
        g["home_score"], g["away_score"] = 90, 120
        g["home_win"] = False

    original, changed = rows_by_id(games), rows_by_id(flipped)
    for gid in (1, 2, 3):
        assert features_only(original[gid]) == features_only(changed[gid]), (
            f"game {gid} changed when a later game changed"
        )


def test_current_season_stats_use_only_earlier_played_games() -> None:
    games = [
        game(1, 0, 1, 2, home_score=110, away_score=100),
        game(2, 2, 1, 3, home_score=90, away_score=120),
        game(3, 4, 1, 4),
    ]
    third = rows_by_id(games)[3]
    assert third["home_games_played"] == 2
    assert third["home_win_pct_prior"] == 0.5
    assert third["home_last5_point_diff"] == pytest.approx((10 + -30) / 2)


def test_rolling_window_keeps_only_the_last_n_games() -> None:
    games = [game(i, i * 2, 1, 20 + i, home_score=110, away_score=100) for i in range(1, 7)]
    # Make the earliest game a loss; it must fall out of the 5-game window.
    games[0]["home_score"], games[0]["away_score"], games[0]["home_win"] = 90, 120, False
    games.append(game(99, 20, 1, 30))

    row = rows_by_id(games)[99]
    assert row["home_games_played"] == 6
    assert row["home_win_pct_prior"] == pytest.approx(5 / 6)
    assert row["home_last5_win_pct"] == 1.0, "the early loss is outside the 5-game window"
    assert row["home_last10_win_pct"] == pytest.approx(5 / 6)


# --- point differential perspective ---------------------------------------


def test_point_differential_is_signed_from_each_team_perspective() -> None:
    games = [game(1, 0, 1, 2, home_score=120, away_score=100), game(2, 2, 1, 3), game(3, 2.5, 2, 4)]
    rows = rows_by_id(games)
    assert rows[2]["home_last5_point_diff"] == 20.0, "team 1 won by 20"
    assert rows[3]["home_last5_point_diff"] == -20.0, "team 2 lost by 20"


# --- season boundaries -----------------------------------------------------


def test_current_season_state_resets_at_a_season_boundary() -> None:
    """Requirement 8: form does not survive the offseason."""
    games = [
        game(1, 0, 1, 2, season=2021),
        game(2, 2, 1, 3, season=2021),
        {**game(3, 200, 1, 4, season=2022)},
    ]
    row = rows_by_id(games)[3]
    assert row["home_games_played"] == 0
    assert row["home_win_pct_prior"] is None
    assert row["home_last5_win_pct"] is None
    assert row["home_last5_point_diff"] is None


def test_rest_does_not_carry_across_the_offseason() -> None:
    """Requirement 9: a season opener must not show ~150 days of rest."""
    games = [game(1, 0, 1, 2, season=2021), game(2, 200, 1, 3, season=2022)]
    row = rows_by_id(games)[2]
    assert row["home_rest_days"] is None
    assert row["home_back_to_back"] is False


# --- rest and back-to-backs ------------------------------------------------


def test_rest_is_measured_from_the_actual_previous_tipoff() -> None:
    games = [game(1, 0, 1, 2), game(2, 3, 1, 3)]
    assert rows_by_id(games)[2]["home_rest_days"] == pytest.approx(3.0)


def test_back_to_back_flags_a_next_day_game() -> None:
    games = [game(1, 0, 1, 2), game(2, 1, 1, 3)]
    row = rows_by_id(games)[2]
    assert row["home_rest_days"] == pytest.approx(1.0)
    assert row["home_back_to_back"] is True
    assert row["away_back_to_back"] is False, "team 3 is playing its first game"


def test_two_days_off_is_not_a_back_to_back() -> None:
    games = [game(1, 0, 1, 2), game(2, 2, 1, 3)]
    assert rows_by_id(games)[2]["home_back_to_back"] is False


def test_rest_diff_is_home_minus_away() -> None:
    games = [game(1, 0, 1, 2), game(2, 0.1, 3, 4), game(3, 4, 1, 3)]
    row = rows_by_id(games)[3]
    assert row["home_rest_days"] == pytest.approx(4.0)
    assert row["away_rest_days"] == pytest.approx(3.9)
    assert row["rest_days_diff"] == pytest.approx(0.1)


# --- chronology ------------------------------------------------------------


def test_a_postponed_game_updates_state_at_its_actual_tipoff() -> None:
    """Requirement 3: the scheduled date is irrelevant to ordering."""
    postponed = game(2, 100, 1, 3, home_score=90, away_score=120, home_win=False)
    postponed["date"] = "2021-10-20"  # scheduled long before it was played
    games = [game(1, 0, 1, 2), game(3, 50, 1, 4), postponed, game(4, 150, 1, 5)]
    games.sort(key=lambda g: g["game_datetime_utc"])

    rows = rows_by_id(games)
    assert rows[3]["home_games_played"] == 1, "the postponed game had not been played yet"
    assert rows[4]["home_games_played"] == 3, "by now it has"


def test_unsorted_input_is_refused() -> None:
    games = [game(2, 5, 1, 2), game(1, 0, 1, 3)]
    with pytest.raises(ValueError, match="must be sorted"):
        build_features(games)


def test_a_game_without_a_timestamp_is_refused() -> None:
    bad = game(1, 0, 1, 2)
    bad["game_datetime_utc"] = None
    with pytest.raises(ValueError, match="cannot be sequenced"):
        build_features([bad])


def test_feature_generation_is_deterministic() -> None:
    """Requirement 13."""
    games = [game(i, i * 1.5, 1 + (i % 4), 5 + (i % 3)) for i in range(1, 40)]
    assert build_features(games) == build_features(copy.deepcopy(games))
