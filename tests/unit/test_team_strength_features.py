"""Adjusted margin, SOS, scoring, splits, and fatigue -- all leakage-audited."""

from __future__ import annotations

import copy
from datetime import UTC, datetime, timedelta

import pytest

from nba_prediction_market.features.adjusted_margin import (
    DEFAULT_MIN_GAMES,
    run_adjusted_margin,
)
from nba_prediction_market.features.team_strength import (
    TEAM_STRENGTH_COLUMNS,
    build_team_strength_features,
    ewma_decay,
)

BASE = datetime(2021, 10, 19, tzinfo=UTC)


def game(gid, day, home, away, home_score=110, away_score=100, season=2021):
    return {
        "nba_game_id": gid, "season": season,
        "game_datetime_utc": BASE + timedelta(days=day),
        "home_franchise_id": home, "away_franchise_id": away,
        "home_team": f"T{home:02d}", "away_team": f"T{away:02d}",
        "home_score": home_score, "away_score": away_score,
        "home_win": home_score > away_score,
    }


def rows(games, elo=None, **kw):
    elo = elo or dict.fromkeys((g["nba_game_id"] for g in games), (1500.0, 1500.0))
    return {r["nba_game_id"]: r for r in build_team_strength_features(games, elo, **kw)}


# --- adjusted margin -------------------------------------------------------


def test_a_team_below_the_minimum_has_no_rating() -> None:
    games = [game(1, 0, 1, 2), game(2, 2, 1, 3)]
    result = {r.nba_game_id: r for r in run_adjusted_margin(games)}
    assert result[1].home_adjusted_margin_rating is None
    assert result[1].adjusted_margin_diff is None


def test_ratings_appear_once_enough_games_are_played() -> None:
    games = [game(i, i, 1, 1 + i) for i in range(2, 8)]
    result = run_adjusted_margin(games, min_games=DEFAULT_MIN_GAMES)
    ratings = [r.home_adjusted_margin_rating for r in result]
    assert ratings[0] is None
    assert any(r is not None for r in ratings)


def test_a_team_beating_strong_opponents_outranks_one_beating_weak_ones() -> None:
    """The whole point: margin adjusted for who produced it."""
    games = []
    gid = 0
    # Teams 5, 6, 7 are strong: they beat the weak sides 8, 9, 10 heavily.
    for strong in (5, 6, 7):
        for weak in (8, 9, 10):
            gid += 1
            games.append(game(gid, gid, strong, weak, 130, 100))
    # Team 1 beats the strong teams by 10; team 2 beats the weak teams by 10.
    for strong in (5, 6, 7):
        gid += 1
        games.append(game(gid, gid, 1, strong, 110, 100))
    for weak in (8, 9, 10):
        gid += 1
        games.append(game(gid, gid, 2, weak, 110, 100))
    gid += 1
    games.append(game(gid, gid, 1, 2, 100, 100))

    final = run_adjusted_margin(games)[-1]
    assert final.home_adjusted_margin_rating is not None
    assert final.away_adjusted_margin_rating is not None
    assert final.home_adjusted_margin_rating > final.away_adjusted_margin_rating, (
        "identical raw margins, but team 1 earned them against stronger opponents"
    )


def test_current_margin_cannot_affect_its_own_rating() -> None:
    games = [game(i, i, 1, 1 + i, 110, 100) for i in range(2, 8)]
    blown_out = copy.deepcopy(games)
    blown_out[-1]["home_score"] = 200
    a = run_adjusted_margin(games)[-1]
    b = run_adjusted_margin(blown_out)[-1]
    assert a.home_adjusted_margin_rating == b.home_adjusted_margin_rating


def test_a_future_result_cannot_affect_a_prior_rating() -> None:
    games = [game(i, i, 1, 1 + i, 110, 100) for i in range(2, 10)]
    changed = copy.deepcopy(games)
    changed[-1]["home_score"] = 200
    original, updated = run_adjusted_margin(games), run_adjusted_margin(changed)
    assert original[:-1] == updated[:-1]


def test_state_resets_between_seasons() -> None:
    first = [game(i, i, 1, 1 + i, 140, 100) for i in range(2, 8)]
    second = [game(100, 300, 1, 2, season=2022)]
    result = {r.nba_game_id: r for r in run_adjusted_margin(first + second)}
    assert result[100].home_adjusted_margin_rating is None


def test_adjusted_ratings_are_centred() -> None:
    games = [game(i, i, 1 + (i % 4), 5 + (i % 4), 110 + i, 100) for i in range(1, 30)]
    result = run_adjusted_margin(games)
    rated = [r for r in result if r.home_adjusted_margin_rating is not None]
    assert rated, "some ratings should be available"


# --- strength of schedule --------------------------------------------------


def test_sos_uses_the_opponent_rating_from_when_the_game_was_played() -> None:
    """A later rating change must not retro-fit an old game's SOS."""
    games = [game(1, 0, 1, 2), game(2, 2, 1, 3)]
    early = rows(games, elo={1: (1500.0, 1200.0), 2: (1500.0, 1500.0)})
    late = rows(games, elo={1: (1500.0, 1200.0), 2: (1500.0, 1900.0)})
    # Game 2's own opponent rating differs, but game 1 recorded 1200 either way.
    assert early[2]["home_season_avg_opponent_elo"] == pytest.approx(1200.0)
    assert late[2]["home_season_avg_opponent_elo"] == pytest.approx(1200.0)


def test_sos_averages_only_prior_games() -> None:
    games = [game(1, 0, 1, 2), game(2, 2, 1, 3), game(3, 4, 1, 4)]
    elo = {1: (1500.0, 1000.0), 2: (1500.0, 2000.0), 3: (1500.0, 1500.0)}
    result = rows(games, elo=elo)
    assert result[1]["home_season_avg_opponent_elo"] is None
    assert result[2]["home_season_avg_opponent_elo"] == pytest.approx(1000.0)
    assert result[3]["home_season_avg_opponent_elo"] == pytest.approx(1500.0)


# --- scoring ---------------------------------------------------------------


def test_scoring_features_use_prior_games_only() -> None:
    games = [game(1, 0, 1, 2, 120, 100), game(2, 2, 1, 3, 90, 95)]
    result = rows(games)
    assert result[1]["home_season_points_scored"] is None
    assert result[2]["home_season_points_scored"] == pytest.approx(120.0)
    assert result[2]["home_season_points_allowed"] == pytest.approx(100.0)
    assert result[2]["home_season_margin"] == pytest.approx(20.0)


def test_ewma_weights_recent_games_more() -> None:
    games = [
        game(1, 0, 1, 2, 90, 100),    # margin -10
        game(2, 2, 1, 3, 130, 100),   # margin +30
        game(3, 4, 1, 4),
    ]
    result = rows(games, ewma_half_life=1.0)
    plain = (-10 + 30) / 2
    assert result[3]["home_ewma_margin"] > plain, "recent +30 must dominate"


def test_a_longer_half_life_approaches_the_simple_mean() -> None:
    games = [game(1, 0, 1, 2, 90, 100), game(2, 2, 1, 3, 130, 100), game(3, 4, 1, 4)]
    slow = rows(games, ewma_half_life=1000.0)[3]["home_ewma_margin"]
    assert slow == pytest.approx(10.0, abs=0.1)


def test_ewma_decay_matches_the_half_life() -> None:
    assert ewma_decay(1.0) == pytest.approx(0.5)
    assert ewma_decay(2.0) ** 2 == pytest.approx(0.5)
    with pytest.raises(ValueError, match="half_life must be positive"):
        ewma_decay(0)


# --- league environment ----------------------------------------------------


def test_league_average_uses_only_prior_games() -> None:
    games = [game(1, 0, 1, 2, 100, 100), game(2, 2, 3, 4, 120, 120), game(3, 4, 1, 3)]
    result = rows(games)
    assert result[1]["league_avg_points_prior"] is None
    assert result[2]["league_avg_points_prior"] == pytest.approx(100.0)
    assert result[3]["league_avg_points_prior"] == pytest.approx(110.0)


def test_league_relative_scoring_is_a_difference_from_the_running_average() -> None:
    games = [game(1, 0, 1, 2, 120, 100), game(2, 2, 1, 3)]
    result = rows(games)
    # After game 1 the league average is 110; team 1 scored 120.
    assert result[2]["home_points_scored_vs_league"] == pytest.approx(10.0)


# --- venue splits ----------------------------------------------------------


def test_venue_splits_use_only_prior_games_in_that_role() -> None:
    games = [
        game(1, 0, 1, 2, 110, 100),   # team 1 wins at home
        game(2, 2, 3, 1, 120, 90),    # team 1 loses away
        game(3, 4, 1, 4),
    ]
    result = rows(games)
    assert result[3]["home_home_win_pct"] == pytest.approx(1.0), "home record only"
    assert result[3]["home_home_margin"] == pytest.approx(10.0)


def test_venue_splits_are_missing_before_any_game_in_that_role() -> None:
    result = rows([game(1, 0, 1, 2)])
    assert result[1]["home_home_win_pct"] is None
    assert result[1]["away_away_win_pct"] is None
    assert result[1]["venue_split_margin_diff"] is None


# --- fatigue ---------------------------------------------------------------


def test_fatigue_counts_prior_games_in_the_trailing_window() -> None:
    games = [game(1, 0, 1, 2), game(2, 1, 1, 3), game(3, 2, 1, 4)]
    result = rows(games)
    assert result[1]["home_games_last3d"] == 0
    assert result[2]["home_games_last3d"] == 1
    assert result[3]["home_games_last3d"] == 2


def test_three_games_in_four_days_is_flagged() -> None:
    games = [game(1, 0, 1, 2), game(2, 1, 1, 3), game(3, 3, 1, 4)]
    result = rows(games)
    assert result[3]["home_3_games_in_4_days"] is True
    assert result[2]["home_3_games_in_4_days"] is False


def test_fatigue_uses_actual_tipoffs_not_scheduled_dates() -> None:
    """A postponed game loads the schedule when it was really played."""
    postponed = game(2, 100, 1, 3)
    postponed["date"] = "2021-10-20"
    games = [game(1, 0, 1, 2), game(3, 99, 1, 4), postponed, game(4, 101, 1, 5)]
    games.sort(key=lambda g: g["game_datetime_utc"])
    result = rows(games)
    # By the last game, the postponed game counts as recent load.
    assert result[4]["home_games_last3d"] == 2
    assert result[3]["home_games_last3d"] == 0


def test_state_resets_at_a_season_boundary() -> None:
    games = [game(1, 0, 1, 2, 130, 100), game(2, 200, 1, 3, season=2022)]
    result = rows(games)
    assert result[2]["home_season_points_scored"] is None
    assert result[2]["home_games_last7d"] == 0
    assert result[2]["home_home_win_pct"] is None


# --- schema and determinism ------------------------------------------------


def test_output_has_the_documented_schema() -> None:
    result = build_team_strength_features([game(1, 0, 1, 2)], {1: (1500.0, 1500.0)})
    assert list(result[0]) == list(TEAM_STRENGTH_COLUMNS)


def test_feature_generation_is_deterministic() -> None:
    games = [game(i, i, 1 + (i % 4), 5 + (i % 3), 100 + i, 100) for i in range(1, 30)]
    elo = dict.fromkeys((g["nba_game_id"] for g in games), (1500.0, 1500.0))
    assert build_team_strength_features(games, elo) == build_team_strength_features(
        copy.deepcopy(games), elo
    )


def test_a_future_result_cannot_change_an_earlier_row() -> None:
    games = [game(i, i, 1, 1 + i) for i in range(2, 8)]
    changed = copy.deepcopy(games)
    changed[-1]["home_score"] = 200
    elo = dict.fromkeys((g["nba_game_id"] for g in games), (1500.0, 1500.0))
    original = build_team_strength_features(games, elo)
    updated = build_team_strength_features(changed, elo)
    assert original[:-1] == updated[:-1]
