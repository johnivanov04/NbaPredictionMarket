"""Sequential Elo ratings for NBA franchises.

Ordering discipline is the whole point: for every game the pregame ratings are
**read and stored first**, and only then updated with the result. A rating that
has already absorbed the current game can never reach that game's feature row.

State is keyed by canonical franchise id, so a relocation or rebrand carries the
rating forward rather than resetting it.

Between seasons each rating regresses toward the league mean::

    new = 1500 + regression_factor * (old - 1500)

``regression_factor = 0`` resets everyone to 1500; ``1`` carries ratings
unchanged. A finite history window is expressed by *starting* the run at a later
season with every team at 1500 -- see :func:`run_elo`.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Final

DEFAULT_INITIAL_RATING: Final = 1500.0
#: Standard Elo scale: a 400-point edge is a 10:1 expected result.
ELO_SCALE: Final = 400.0

#: Sentinel meaning "start from the earliest season available".
HISTORY_ALL: Final = "all_available"


@dataclass(frozen=True)
class EloConfig:
    """A complete Elo specification."""

    k_factor: float
    home_advantage: float
    regression_factor: float
    history: int | str = HISTORY_ALL
    initial_rating: float = DEFAULT_INITIAL_RATING

    def __post_init__(self) -> None:
        if self.k_factor <= 0:
            raise ValueError(f"k_factor must be positive, got {self.k_factor}")
        if not 0.0 <= self.regression_factor <= 1.0:
            raise ValueError(
                f"regression_factor must be within [0, 1], got {self.regression_factor}"
            )
        if isinstance(self.history, int) and self.history < 1:
            raise ValueError(f"history must be >= 1 season, got {self.history}")
        if isinstance(self.history, str) and self.history != HISTORY_ALL:
            raise ValueError(f"history must be an int or {HISTORY_ALL!r}, got {self.history!r}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "k_factor": self.k_factor,
            "home_advantage": self.home_advantage,
            "regression_factor": self.regression_factor,
            "history": self.history,
            "initial_rating": self.initial_rating,
        }

    def start_season(self, target_season: int, earliest_season: int) -> int:
        """First season the run initialises from, for predicting ``target_season``."""
        if self.history == HISTORY_ALL:
            return earliest_season
        return max(earliest_season, target_season - int(self.history))


def expected_home_win(
    home_rating: float, away_rating: float, home_advantage: float
) -> float:
    """Elo win probability for the home side, including home-court advantage."""
    diff = (home_rating + home_advantage) - away_rating
    return 1.0 / (1.0 + math.pow(10.0, -diff / ELO_SCALE))


@dataclass
class EloState:
    """Mutable ratings keyed by franchise id."""

    initial_rating: float = DEFAULT_INITIAL_RATING
    ratings: dict[Any, float] = field(default_factory=dict)

    def rating(self, franchise_id: Any) -> float:
        return self.ratings.get(franchise_id, self.initial_rating)

    def regress_to_mean(self, regression_factor: float) -> None:
        """Apply the offseason pull toward 1500."""
        for key, value in list(self.ratings.items()):
            self.ratings[key] = (
                self.initial_rating + regression_factor * (value - self.initial_rating)
            )

    def update(
        self,
        home_id: Any,
        away_id: Any,
        home_win: bool,
        *,
        k_factor: float,
        expected: float,
    ) -> None:
        """Apply the post-game rating change. Called only after the row is stored."""
        actual = 1.0 if home_win else 0.0
        delta = k_factor * (actual - expected)
        self.ratings[home_id] = self.rating(home_id) + delta
        self.ratings[away_id] = self.rating(away_id) - delta


@dataclass(frozen=True)
class EloPrediction:
    """One game's pregame Elo view, captured before the update."""

    nba_game_id: Any
    season: int
    home_elo: float
    away_elo: float
    elo_diff: float
    home_win_probability: float


def run_elo(
    games: Sequence[Mapping[str, Any]],
    config: EloConfig,
    *,
    predict_seasons: Iterable[int] | None = None,
) -> list[EloPrediction]:
    """Run Elo chronologically and return one pregame row per game.

    ``games`` must already be ordered by actual played tipoff; this function does
    not sort, so an unordered input is a caller error rather than something
    silently repaired. Ratings regress between seasons.

    ``predict_seasons`` restricts which games appear in the output; every game is
    still used to update state. Passing ``None`` returns every game.
    """
    wanted = None if predict_seasons is None else set(predict_seasons)
    state = EloState(initial_rating=config.initial_rating)
    out: list[EloPrediction] = []
    current_season: int | None = None

    for game in games:
        season = int(game["season"])
        if current_season is None:
            current_season = season
        elif season != current_season:
            # One regression per season boundary crossed.
            for _ in range(season - current_season):
                state.regress_to_mean(config.regression_factor)
            current_season = season

        home_id = game["home_franchise_id"]
        away_id = game["away_franchise_id"]
        home_rating = state.rating(home_id)
        away_rating = state.rating(away_id)
        expected = expected_home_win(home_rating, away_rating, config.home_advantage)

        # Store the pregame view BEFORE any update touches these ratings.
        if wanted is None or season in wanted:
            out.append(
                EloPrediction(
                    nba_game_id=game["nba_game_id"],
                    season=season,
                    home_elo=home_rating,
                    away_elo=away_rating,
                    elo_diff=(home_rating - away_rating),
                    home_win_probability=expected,
                )
            )

        home_win = game.get("home_win")
        if home_win is None:
            # Nothing trustworthy to learn from; ratings stand.
            continue
        state.update(
            home_id, away_id, bool(home_win), k_factor=config.k_factor, expected=expected
        )

    return out
