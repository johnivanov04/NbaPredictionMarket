"""Margin-of-victory aware Elo.

Binary Elo treats a one-point win and a thirty-point win identically. This
variant scales the rating update by a function of the final margin, so a blowout
moves ratings further than a buzzer-beater.

The ordering discipline is unchanged and is the thing most worth guarding: the
pregame ratings are read and stored **before** the current game's margin touches
anything. A margin can only ever influence *later* games.

Three multiplier formulations are offered, all standard and all bounded so a
single extreme result cannot dominate:

``log``
    ``ln(|margin| + 1)``. Simple, strongly diminishing.
``fivethirtyeight``
    ``ln(|margin| + 1) * 2.2 / (0.001 * elo_diff_winner + 2.2)``. Additionally
    damps margins run up by an already-favoured team, which counteracts the
    autocorrelation that otherwise inflates strong teams' ratings.
``sqrt``
    ``sqrt(|margin|)``. A middle ground between linear and logarithmic.

Multipliers are normalised so that a typical NBA margin leaves the effective K
close to the binary case, which keeps the K grid comparable across variants.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Final

from nba_prediction_market.features.elo import (
    DEFAULT_INITIAL_RATING,
    HISTORY_ALL,
    EloState,
    expected_home_win,
)

MULTIPLIER_LOG: Final = "log"
MULTIPLIER_538: Final = "fivethirtyeight"
MULTIPLIER_SQRT: Final = "sqrt"
MOV_MULTIPLIERS: Final[tuple[str, ...]] = (MULTIPLIER_LOG, MULTIPLIER_538, MULTIPLIER_SQRT)

#: Divisors chosen so an average NBA margin (~11 points) yields a multiplier near
#: 1.0, keeping effective K comparable with the binary Elo grid.
_NORMALISERS: Final[dict[str, float]] = {
    MULTIPLIER_LOG: math.log(12.0),
    MULTIPLIER_538: math.log(12.0),
    MULTIPLIER_SQRT: math.sqrt(11.0),
}


def margin_multiplier(
    formulation: str, margin: float, winner_elo_diff: float
) -> float:
    """Scale factor applied to the Elo update for one game.

    ``winner_elo_diff`` is the winner's pregame rating minus the loser's, from the
    winner's perspective, used only by the ``fivethirtyeight`` damping term.
    """
    magnitude = abs(float(margin))
    if formulation == MULTIPLIER_LOG:
        raw = math.log(magnitude + 1.0)
    elif formulation == MULTIPLIER_SQRT:
        raw = math.sqrt(magnitude)
    elif formulation == MULTIPLIER_538:
        raw = math.log(magnitude + 1.0) * (2.2 / (0.001 * winner_elo_diff + 2.2))
    else:
        raise ValueError(f"unknown margin formulation {formulation!r}")
    return raw / _NORMALISERS[formulation]


@dataclass(frozen=True)
class MovEloConfig:
    """A margin-aware Elo specification."""

    k_factor: float
    home_advantage: float
    regression_factor: float
    multiplier: str = MULTIPLIER_538
    history: int | str = HISTORY_ALL
    initial_rating: float = DEFAULT_INITIAL_RATING

    def __post_init__(self) -> None:
        if self.k_factor <= 0:
            raise ValueError(f"k_factor must be positive, got {self.k_factor}")
        if not 0.0 <= self.regression_factor <= 1.0:
            raise ValueError("regression_factor must be within [0, 1]")
        if self.multiplier not in MOV_MULTIPLIERS:
            raise ValueError(
                f"multiplier must be one of {MOV_MULTIPLIERS}, got {self.multiplier!r}"
            )
        if isinstance(self.history, int) and self.history < 1:
            raise ValueError(f"history must be >= 1 season, got {self.history}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "k_factor": self.k_factor,
            "home_advantage": self.home_advantage,
            "regression_factor": self.regression_factor,
            "multiplier": self.multiplier,
            "history": self.history,
            "initial_rating": self.initial_rating,
        }

    def start_season(self, target_season: int, earliest_season: int) -> int:
        if self.history == HISTORY_ALL:
            return earliest_season
        return max(earliest_season, target_season - int(self.history))


@dataclass(frozen=True)
class MovEloPrediction:
    """One game's pregame MOV-Elo view, captured before the update."""

    nba_game_id: Any
    season: int
    home_mov_elo: float
    away_mov_elo: float
    mov_elo_diff: float
    home_win_probability: float


@dataclass
class _State:
    elo: EloState = field(default_factory=EloState)


def run_mov_elo(
    games: Sequence[Mapping[str, Any]],
    config: MovEloConfig,
    *,
    predict_seasons: Iterable[int] | None = None,
) -> list[MovEloPrediction]:
    """Run margin-aware Elo chronologically, one pregame row per game.

    ``games`` must already be ordered by actual tipoff. Ratings regress between
    seasons exactly as in the binary system.
    """
    wanted = None if predict_seasons is None else set(predict_seasons)
    state = EloState(initial_rating=config.initial_rating)
    out: list[MovEloPrediction] = []
    current_season: int | None = None

    for game in games:
        season = int(game["season"])
        if current_season is None:
            current_season = season
        elif season != current_season:
            for _ in range(season - current_season):
                state.regress_to_mean(config.regression_factor)
            current_season = season

        home_id, away_id = game["home_franchise_id"], game["away_franchise_id"]
        home_rating, away_rating = state.rating(home_id), state.rating(away_id)
        expected = expected_home_win(home_rating, away_rating, config.home_advantage)

        # Pregame view is fixed here; nothing below can reach it.
        if wanted is None or season in wanted:
            out.append(
                MovEloPrediction(
                    nba_game_id=game["nba_game_id"],
                    season=season,
                    home_mov_elo=home_rating,
                    away_mov_elo=away_rating,
                    mov_elo_diff=home_rating - away_rating,
                    home_win_probability=expected,
                )
            )

        home_win = game.get("home_win")
        if home_win is None:
            continue
        home_score, away_score = game.get("home_score"), game.get("away_score")
        if home_score is None or away_score is None:
            continue

        margin = float(home_score) - float(away_score)
        # Rating gap from the winner's point of view, including home court.
        if bool(home_win):
            winner_diff = (home_rating + config.home_advantage) - away_rating
        else:
            winner_diff = away_rating - (home_rating + config.home_advantage)
        multiplier = margin_multiplier(config.multiplier, margin, winner_diff)
        state.update(
            home_id,
            away_id,
            bool(home_win),
            k_factor=config.k_factor * multiplier,
            expected=expected,
        )

    return out
