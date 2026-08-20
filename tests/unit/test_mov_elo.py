"""Margin-of-victory Elo: multiplier math and update ordering."""

from __future__ import annotations

import copy
from datetime import UTC, datetime, timedelta

import pytest

from nba_prediction_market.features.elo import DEFAULT_INITIAL_RATING
from nba_prediction_market.features.mov_elo import (
    MOV_MULTIPLIERS,
    MULTIPLIER_538,
    MULTIPLIER_LOG,
    MULTIPLIER_SQRT,
    MovEloConfig,
    margin_multiplier,
    run_mov_elo,
)

BASE = datetime(2021, 10, 19, tzinfo=UTC)


def game(gid, day, home, away, home_score=110, away_score=100, season=2021):
    return {
        "nba_game_id": gid, "season": season,
        "game_datetime_utc": BASE + timedelta(days=day),
        "home_franchise_id": home, "away_franchise_id": away,
        "home_score": home_score, "away_score": away_score,
        "home_win": home_score > away_score,
    }


def config(**kw):
    base = {"k_factor": 20.0, "home_advantage": 0.0, "regression_factor": 1.0}
    base.update(kw)
    return MovEloConfig(**base)


# --- multiplier behaviour --------------------------------------------------


@pytest.mark.parametrize("formulation", MOV_MULTIPLIERS)
def test_a_bigger_margin_gives_a_bigger_multiplier(formulation: str) -> None:
    small = margin_multiplier(formulation, 1, 0.0)
    large = margin_multiplier(formulation, 30, 0.0)
    assert 0 < small < large


@pytest.mark.parametrize("formulation", MOV_MULTIPLIERS)
def test_multipliers_are_sign_agnostic(formulation: str) -> None:
    assert margin_multiplier(formulation, 12, 0.0) == margin_multiplier(
        formulation, -12, 0.0
    )


@pytest.mark.parametrize("formulation", MOV_MULTIPLIERS)
def test_a_typical_margin_leaves_effective_k_near_the_binary_case(
    formulation: str,
) -> None:
    """Normalisation keeps the K grid comparable across variants."""
    assert 0.85 < margin_multiplier(formulation, 11, 0.0) < 1.15


@pytest.mark.parametrize("formulation", MOV_MULTIPLIERS)
def test_multipliers_grow_sublinearly(formulation: str) -> None:
    """A 30-point win must not count 30 times a 1-point win."""
    ratio = margin_multiplier(formulation, 30, 0.0) / margin_multiplier(formulation, 1, 0.0)
    assert ratio < 10


def test_the_538_variant_damps_a_favourite_running_up_the_score() -> None:
    """Autocorrelation guard: the same margin counts less for a big favourite."""
    underdog = margin_multiplier(MULTIPLIER_538, 20, -200.0)
    favourite = margin_multiplier(MULTIPLIER_538, 20, 200.0)
    assert favourite < underdog


def test_the_other_variants_ignore_the_rating_gap() -> None:
    for formulation in (MULTIPLIER_LOG, MULTIPLIER_SQRT):
        assert margin_multiplier(formulation, 20, -200.0) == margin_multiplier(
            formulation, 20, 200.0
        )


def test_an_unknown_formulation_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown margin formulation"):
        margin_multiplier("linear", 10, 0.0)


# --- ordering and leakage --------------------------------------------------


def test_pregame_ratings_precede_the_margin_update() -> None:
    """Requirement: the current margin cannot affect its own MOV Elo."""
    predictions = run_mov_elo([game(1, 0, 1, 2), game(2, 2, 1, 3)], config())
    assert predictions[0].home_mov_elo == DEFAULT_INITIAL_RATING
    assert predictions[0].away_mov_elo == DEFAULT_INITIAL_RATING


def test_changing_the_current_margin_does_not_change_its_own_row() -> None:
    games = [game(1, 0, 1, 2, 110, 100)]
    blowout = [game(1, 0, 1, 2, 140, 100)]
    a, b = run_mov_elo(games, config())[0], run_mov_elo(blowout, config())[0]
    assert (a.home_mov_elo, a.away_mov_elo, a.home_win_probability) == (
        b.home_mov_elo, b.away_mov_elo, b.home_win_probability
    )


def test_a_bigger_margin_moves_later_ratings_further() -> None:
    close = run_mov_elo([game(1, 0, 1, 2, 101, 100), game(2, 2, 1, 3)], config())
    blowout = run_mov_elo([game(1, 0, 1, 2, 140, 100), game(2, 2, 1, 3)], config())
    assert blowout[1].home_mov_elo > close[1].home_mov_elo


def test_changing_a_future_margin_leaves_earlier_rows_untouched() -> None:
    games = [game(i, i * 2, 1, 2 + i) for i in range(1, 5)]
    changed = copy.deepcopy(games)
    changed[3]["home_score"] = 160
    original, updated = run_mov_elo(games, config()), run_mov_elo(changed, config())
    for a, b in zip(original[:3], updated[:3], strict=True):
        assert a == b


def test_a_game_without_scores_leaves_ratings_untouched() -> None:
    incomplete = game(1, 0, 1, 2)
    incomplete["home_score"] = None
    predictions = run_mov_elo([incomplete, game(2, 2, 1, 3)], config())
    assert predictions[1].home_mov_elo == DEFAULT_INITIAL_RATING


def test_offseason_regression_applies() -> None:
    predictions = run_mov_elo(
        [game(1, 0, 1, 2), game(2, 400, 1, 3, season=2022)],
        config(regression_factor=0.0),
    )
    assert predictions[1].home_mov_elo == pytest.approx(DEFAULT_INITIAL_RATING)


def test_mov_elo_is_deterministic() -> None:
    games = [game(i, i, 1 + (i % 4), 5 + (i % 3), 100 + i, 100) for i in range(1, 30)]
    first = run_mov_elo(games, config(multiplier=MULTIPLIER_538))
    second = run_mov_elo(copy.deepcopy(games), config(multiplier=MULTIPLIER_538))
    assert first == second


def test_diff_is_home_minus_away() -> None:
    predictions = run_mov_elo([game(1, 0, 1, 2), game(2, 2, 1, 3)], config())
    p = predictions[1]
    assert p.mov_elo_diff == pytest.approx(p.home_mov_elo - p.away_mov_elo)


@pytest.mark.parametrize("multiplier", ["linear", "", None])
def test_an_invalid_multiplier_is_rejected_at_construction(multiplier) -> None:
    with pytest.raises(ValueError, match="multiplier"):
        config(multiplier=multiplier)
