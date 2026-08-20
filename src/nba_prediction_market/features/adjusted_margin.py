"""Online opponent-adjusted margin ratings (Massey/SRS style).

A team's raw point differential is confounded by who it played. This module
estimates, for every game, a rating that satisfies

    rating[t] = mean over t's prior games of ( margin + rating[opponent] )

subject to the ratings of teams that have played averaging zero. That is the
standard Simple Rating System identity: your strength is your average margin
adjusted for the strength of the opponents who produced it.

**The rating is recomputed from scratch before every game using only games
already played.** A full-season SRS attached retrospectively would be flatly
leaky -- it would encode results that had not happened yet -- so the solve is
re-run as the season unfolds and the state is reset at each season boundary.

The system is solved by fixed-point iteration on a 30x30 meeting-count matrix,
which is fast enough to redo per game and avoids any matrix inversion or
identifiability trouble.

Early season is genuinely sparse: with one or two games played the estimate is
mostly noise, so teams below ``min_games`` report ``None`` and the model
pipeline imputes them on training data only.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Final

import numpy as np

#: Fixed-point iterations per solve. The system converges quickly; 25 is well
#: past the point where further iterations change a rating meaningfully.
DEFAULT_ITERATIONS: Final = 25
#: Games a team must have played before its rating is reported at all.
DEFAULT_MIN_GAMES: Final = 3


@dataclass
class AdjustedMarginState:
    """Within-season accumulators for the SRS solve.

    Teams are mapped to dense indices so the meeting counts can live in a small
    square matrix.
    """

    min_games: int = DEFAULT_MIN_GAMES
    iterations: int = DEFAULT_ITERATIONS
    index: dict[Any, int] = field(default_factory=dict)
    counts: np.ndarray | None = None
    margin_sum: np.ndarray | None = None
    games: np.ndarray | None = None
    _ratings: np.ndarray | None = None
    _dirty: bool = True

    def _slot(self, team: Any) -> int:
        if team in self.index:
            return self.index[team]
        slot = len(self.index)
        self.index[team] = slot
        size = slot + 1
        self.counts = _grow_square(self.counts, size)
        self.margin_sum = _grow_vector(self.margin_sum, size)
        self.games = _grow_vector(self.games, size)
        self._dirty = True
        return slot

    def record(self, home: Any, away: Any, home_margin: float) -> None:
        """Fold one completed game into the accumulators."""
        h, a = self._slot(home), self._slot(away)
        self.counts[h, a] += 1.0
        self.counts[a, h] += 1.0
        self.margin_sum[h] += home_margin
        self.margin_sum[a] -= home_margin
        self.games[h] += 1.0
        self.games[a] += 1.0
        self._dirty = True

    def ratings(self) -> np.ndarray:
        """Solve the SRS fixed point over the games recorded so far."""
        if self.counts is None:
            return np.zeros(0)
        if not self._dirty and self._ratings is not None:
            return self._ratings
        played = self.games > 0
        rating = np.zeros(len(self.games))
        if played.any():
            safe_games = np.where(played, self.games, 1.0)
            for _ in range(self.iterations):
                opponent_sum = self.counts @ rating
                updated = np.where(played, (self.margin_sum + opponent_sum) / safe_games, 0.0)
                # SRS is identified only up to a constant; centre on the teams
                # that have actually played.
                updated = np.where(played, updated - updated[played].mean(), 0.0)
                rating = updated
        self._ratings = rating
        self._dirty = False
        return rating

    def rating_for(self, team: Any) -> float | None:
        """Current adjusted margin for ``team``, or ``None`` while too sparse."""
        slot = self.index.get(team)
        if slot is None or self.games is None or self.games[slot] < self.min_games:
            return None
        return float(self.ratings()[slot])


def _grow_square(matrix: np.ndarray | None, size: int) -> np.ndarray:
    grown = np.zeros((size, size))
    if matrix is not None:
        grown[: matrix.shape[0], : matrix.shape[1]] = matrix
    return grown


def _grow_vector(vector: np.ndarray | None, size: int) -> np.ndarray:
    grown = np.zeros(size)
    if vector is not None:
        grown[: vector.shape[0]] = vector
    return grown


@dataclass(frozen=True)
class AdjustedMarginPrediction:
    """One game's pregame adjusted-margin view."""

    nba_game_id: Any
    season: int
    home_adjusted_margin_rating: float | None
    away_adjusted_margin_rating: float | None
    adjusted_margin_diff: float | None


def run_adjusted_margin(
    games: Sequence[Mapping[str, Any]],
    *,
    min_games: int = DEFAULT_MIN_GAMES,
    iterations: int = DEFAULT_ITERATIONS,
) -> list[AdjustedMarginPrediction]:
    """Produce a pregame adjusted-margin rating for every game.

    State resets at each season boundary: strength of schedule is a within-season
    quantity here, and carrying it over would blur the two.
    """
    out: list[AdjustedMarginPrediction] = []
    state = AdjustedMarginState(min_games=min_games, iterations=iterations)
    current_season: int | None = None

    for game in games:
        season = int(game["season"])
        if season != current_season:
            state = AdjustedMarginState(min_games=min_games, iterations=iterations)
            current_season = season

        home_id, away_id = game["home_franchise_id"], game["away_franchise_id"]
        home_rating = state.rating_for(home_id)
        away_rating = state.rating_for(away_id)
        out.append(
            AdjustedMarginPrediction(
                nba_game_id=game["nba_game_id"],
                season=season,
                home_adjusted_margin_rating=home_rating,
                away_adjusted_margin_rating=away_rating,
                adjusted_margin_diff=(
                    None
                    if home_rating is None or away_rating is None
                    else home_rating - away_rating
                ),
            )
        )

        # --- only now may this game's margin enter the state ---
        home_score, away_score = game.get("home_score"), game.get("away_score")
        if home_score is None or away_score is None:
            continue
        state.record(home_id, away_id, float(home_score) - float(away_score))

    return out
