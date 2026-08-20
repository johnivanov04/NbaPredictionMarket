"""Elo: ordering discipline, math, and history-window semantics."""

from __future__ import annotations

import copy
from datetime import UTC, datetime, timedelta

import pytest

from nba_prediction_market.features.elo import (
    DEFAULT_INITIAL_RATING,
    ELO_SCALE,
    HISTORY_ALL,
    EloConfig,
    EloState,
    expected_home_win,
    run_elo,
)

BASE = datetime(2021, 10, 19, tzinfo=UTC)


def game(gid, day, home, away, home_win=True, season=2021):
    return {
        "nba_game_id": gid, "season": season,
        "game_datetime_utc": BASE + timedelta(days=day),
        "home_franchise_id": home, "away_franchise_id": away, "home_win": home_win,
    }


# --- probability math ------------------------------------------------------


def test_equal_ratings_without_home_advantage_is_a_coin_flip() -> None:
    assert expected_home_win(1500, 1500, 0.0) == pytest.approx(0.5)


def test_home_advantage_raises_the_home_probability() -> None:
    assert expected_home_win(1500, 1500, 100.0) > 0.5
    assert expected_home_win(1500, 1500, 100.0) == pytest.approx(
        1 / (1 + 10 ** (-100 / ELO_SCALE))
    )


def test_a_400_point_edge_is_ten_to_one() -> None:
    assert expected_home_win(1900, 1500, 0.0) == pytest.approx(10 / 11)


def test_probability_is_monotonic_in_rating_difference() -> None:
    probs = [expected_home_win(1500 + d, 1500, 0.0) for d in (-200, -100, 0, 100, 200)]
    assert probs == sorted(probs)
    assert all(0.0 < p < 1.0 for p in probs)


# --- update ordering -------------------------------------------------------


def test_pregame_ratings_are_stored_before_the_update() -> None:
    """Requirement 6: the stored rating must not contain the current result."""
    config = EloConfig(k_factor=20, home_advantage=0, regression_factor=1.0)
    predictions = run_elo([game(1, 0, 1, 2), game(2, 2, 1, 3)], config)

    assert predictions[0].home_elo == DEFAULT_INITIAL_RATING
    assert predictions[0].away_elo == DEFAULT_INITIAL_RATING
    # Only the second game sees the first result.
    assert predictions[1].home_elo == pytest.approx(1510.0)


def test_a_win_moves_ratings_by_k_times_the_surprise() -> None:
    config = EloConfig(k_factor=32, home_advantage=0, regression_factor=1.0)
    predictions = run_elo([game(1, 0, 1, 2), game(2, 2, 1, 3)], config)
    # Expected 0.5, actual 1 -> delta = 32 * 0.5 = 16
    assert predictions[1].home_elo == pytest.approx(1516.0)


def test_the_loser_loses_exactly_what_the_winner_gains() -> None:
    config = EloConfig(k_factor=20, home_advantage=0, regression_factor=1.0)
    state = EloState()
    state.update(1, 2, True, k_factor=20, expected=0.5)
    assert state.rating(1) - DEFAULT_INITIAL_RATING == pytest.approx(
        DEFAULT_INITIAL_RATING - state.rating(2)
    )
    assert config.k_factor == 20


def test_a_larger_k_moves_ratings_further() -> None:
    small = run_elo(
        [game(1, 0, 1, 2), game(2, 2, 1, 3)],
        EloConfig(k_factor=10, home_advantage=0, regression_factor=1.0),
    )
    large = run_elo(
        [game(1, 0, 1, 2), game(2, 2, 1, 3)],
        EloConfig(k_factor=40, home_advantage=0, regression_factor=1.0),
    )
    assert large[1].home_elo > small[1].home_elo


def test_a_game_without_a_result_leaves_ratings_untouched() -> None:
    config = EloConfig(k_factor=20, home_advantage=0, regression_factor=1.0)
    predictions = run_elo(
        [game(1, 0, 1, 2, home_win=None), game(2, 2, 1, 3)], config
    )
    assert predictions[1].home_elo == DEFAULT_INITIAL_RATING


# --- offseason regression --------------------------------------------------


@pytest.mark.parametrize(
    ("factor", "expected"), [(0.0, 1500.0), (0.5, 1505.0), (1.0, 1510.0)]
)
def test_offseason_regression_pulls_toward_1500(factor, expected) -> None:
    config = EloConfig(k_factor=20, home_advantage=0, regression_factor=factor)
    predictions = run_elo(
        [game(1, 0, 1, 2), game(2, 200, 1, 3, season=2022)], config
    )
    assert predictions[1].home_elo == pytest.approx(expected)


def test_regression_applies_once_per_season_boundary_crossed() -> None:
    config = EloConfig(k_factor=20, home_advantage=0, regression_factor=0.5)
    predictions = run_elo(
        [game(1, 0, 1, 2), game(2, 800, 1, 3, season=2023)], config
    )
    # Two boundaries (2021->2022->2023): 1510 -> 1505 -> 1502.5
    assert predictions[1].home_elo == pytest.approx(1502.5)


# --- history windows -------------------------------------------------------


@pytest.mark.parametrize(
    ("history", "target", "earliest", "expected"),
    [
        (3, 2021, 2006, 2018),
        (5, 2021, 2006, 2016),
        (15, 2021, 2006, 2006),
        (HISTORY_ALL, 2021, 2006, 2006),
        (10, 2010, 2006, 2006),
    ],
)
def test_history_window_start_season(history, target, earliest, expected) -> None:
    config = EloConfig(
        k_factor=20, home_advantage=0, regression_factor=0.5, history=history
    )
    assert config.start_season(target, earliest) == expected


def test_a_finite_window_starts_everyone_at_1500() -> None:
    """Requirement: finite history initialises fresh, discarding older results."""
    config = EloConfig(k_factor=20, home_advantage=0, regression_factor=1.0)
    late_only = run_elo([game(9, 0, 1, 2, season=2021)], config)
    assert late_only[0].home_elo == DEFAULT_INITIAL_RATING


def test_predict_seasons_restricts_output_but_not_state() -> None:
    config = EloConfig(k_factor=20, home_advantage=0, regression_factor=1.0)
    games = [game(1, 0, 1, 2, season=2020), game(2, 400, 1, 3, season=2021)]
    predictions = run_elo(games, config, predict_seasons=[2021])

    assert [p.nba_game_id for p in predictions] == [2]
    # The 2020 game still moved the rating even though it was not emitted.
    assert predictions[0].home_elo > DEFAULT_INITIAL_RATING


def test_elo_diff_is_home_minus_away() -> None:
    config = EloConfig(k_factor=20, home_advantage=0, regression_factor=1.0)
    predictions = run_elo([game(1, 0, 1, 2), game(2, 2, 1, 3)], config)
    p = predictions[1]
    assert p.elo_diff == pytest.approx(p.home_elo - p.away_elo)


def test_elo_is_deterministic() -> None:
    config = EloConfig(k_factor=20, home_advantage=60, regression_factor=0.5)
    games = [game(i, i, 1 + (i % 5), 6 + (i % 4), home_win=bool(i % 3)) for i in range(1, 40)]
    first = run_elo(games, config)
    second = run_elo(copy.deepcopy(games), config)
    assert [(p.nba_game_id, p.home_elo, p.away_elo) for p in first] == [
        (p.nba_game_id, p.home_elo, p.away_elo) for p in second
    ]


# --- configuration validation ---------------------------------------------


@pytest.mark.parametrize("k", [0, -1])
def test_non_positive_k_is_rejected(k) -> None:
    with pytest.raises(ValueError, match="k_factor"):
        EloConfig(k_factor=k, home_advantage=0, regression_factor=0.5)


@pytest.mark.parametrize("factor", [-0.1, 1.1])
def test_regression_factor_outside_zero_one_is_rejected(factor) -> None:
    with pytest.raises(ValueError, match="regression_factor"):
        EloConfig(k_factor=20, home_advantage=0, regression_factor=factor)


@pytest.mark.parametrize("history", [0, -3, "recent"])
def test_invalid_history_is_rejected(history) -> None:
    with pytest.raises(ValueError, match="history"):
        EloConfig(k_factor=20, home_advantage=0, regression_factor=0.5, history=history)
