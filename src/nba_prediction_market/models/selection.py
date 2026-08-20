"""Development-time model selection.

Deliberately separate from holdout evaluation. Nothing in this module may read
season 2025 -- :func:`assert_no_holdout` enforces that on every candidate set, so
tuning against the holdout requires actively removing a guard rather than
forgetting one.
"""

from __future__ import annotations

import itertools
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Final

import numpy as np
import pandas as pd

from nba_prediction_market.features.elo import HISTORY_ALL, EloConfig, run_elo
from nba_prediction_market.models.logistic import (
    HISTORY_WEIGHTED,
    LogisticConfig,
    fit_logistic,
    training_seasons,
)
from nba_prediction_market.models.metrics import brier_score, log_loss

#: The season held out entirely from development.
HOLDOUT_SEASON: Final = 2025
#: Seasons used to compare candidate configurations.
DEVELOPMENT_VALIDATION_SEASONS: Final[tuple[int, ...]] = (2021, 2022, 2023, 2024)

ELO_K_GRID: Final[tuple[float, ...]] = (10.0, 20.0, 30.0, 40.0)
ELO_HOME_ADVANTAGE_GRID: Final[tuple[float, ...]] = (40.0, 60.0, 80.0, 100.0)
ELO_REGRESSION_GRID: Final[tuple[float, ...]] = (0.25, 0.50, 0.75)
ELO_HISTORY_GRID: Final[tuple[int | str, ...]] = (3, 5, 8, 10, 15, HISTORY_ALL)

LOGISTIC_HISTORY_GRID: Final[tuple[int | str, ...]] = (3, 5, 8, 10, 15, "all_available")
LOGISTIC_HALF_LIVES: Final[tuple[float, ...]] = (2.0, 3.0, 5.0, 8.0)
LOGISTIC_C_GRID: Final[tuple[float, ...]] = (0.01, 0.1, 1.0, 10.0)

#: Mean-Brier differences below this are treated as ties rather than as a real
#: advantage. Fold-to-fold Brier varies by ~0.01 across validation seasons, so a
#: difference three orders of magnitude smaller carries no information.
TIE_TOLERANCE: Final = 1e-4


class HoldoutLeakageError(RuntimeError):
    """Raised if a development stage is handed holdout data."""


def assert_no_holdout(seasons: Sequence[int], *, where: str) -> None:
    """Refuse to proceed if the holdout season appears during development."""
    offending = sorted({int(s) for s in seasons if int(s) >= HOLDOUT_SEASON})
    if offending:
        raise HoldoutLeakageError(
            f"{where} received holdout season(s) {offending}. Development must never "
            f"read season {HOLDOUT_SEASON}; freeze all choices first."
        )


@dataclass
class FoldResult:
    """One configuration evaluated on one validation season."""

    season: int
    brier_score: float
    log_loss: float
    n: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "season": self.season,
            "brier_score": self.brier_score,
            "log_loss": self.log_loss,
            "n": self.n,
        }


@dataclass
class CandidateResult:
    """A configuration's performance across every development fold."""

    label: str
    config: dict[str, Any]
    folds: list[FoldResult]

    @property
    def mean_brier(self) -> float:
        return float(np.mean([f.brier_score for f in self.folds]))

    @property
    def mean_log_loss(self) -> float:
        return float(np.mean([f.log_loss for f in self.folds]))

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "config": self.config,
            "mean_brier_score": self.mean_brier,
            "mean_log_loss": self.mean_log_loss,
            "folds": [f.to_dict() for f in self.folds],
        }


def rank_candidates(candidates: Sequence[CandidateResult]) -> list[CandidateResult]:
    """Best first: mean Brier primary, mean log loss as the tiebreak."""
    return sorted(candidates, key=lambda c: (c.mean_brier, c.mean_log_loss))


def tied_with_best(
    ranked: Sequence[CandidateResult], tolerance: float = TIE_TOLERANCE
) -> list[CandidateResult]:
    """Candidates statistically indistinguishable from the best one."""
    if not ranked:
        return []
    best = ranked[0].mean_brier
    return [c for c in ranked if c.mean_brier - best <= tolerance]


# --- Elo selection ---------------------------------------------------------


def elo_grid() -> list[EloConfig]:
    """Every Elo configuration under test."""
    return [
        EloConfig(k_factor=k, home_advantage=h, regression_factor=r, history=hist)
        for k, h, r, hist in itertools.product(
            ELO_K_GRID, ELO_HOME_ADVANTAGE_GRID, ELO_REGRESSION_GRID, ELO_HISTORY_GRID
        )
    ]


def records_by_season(games: pd.DataFrame) -> dict[int, list[dict[str, Any]]]:
    """Chronological records grouped by season, built once and reused.

    The Elo grid re-simulates hundreds of configurations; rebuilding records per
    configuration would dominate the runtime.
    """
    out: dict[int, list[dict[str, Any]]] = {}
    for season, sub in games.groupby("season", sort=True):
        out[int(season)] = sub.to_dict("records")
    return out


def truths_by_season(games: pd.DataFrame) -> dict[int, dict[Any, int]]:
    """``season -> {game_id: home_win}``, built once."""
    out: dict[int, dict[Any, int]] = {}
    for season, sub in games.groupby("season", sort=True):
        out[int(season)] = dict(
            zip(sub["nba_game_id"], sub["home_win"].astype(int), strict=True)
        )
    return out


def evaluate_elo_config(
    games: pd.DataFrame,
    config: EloConfig,
    validation_seasons: Sequence[int],
    earliest_season: int,
    *,
    by_season: dict[int, list[dict[str, Any]]] | None = None,
    truths: dict[int, dict[Any, int]] | None = None,
) -> CandidateResult:
    """Score one Elo configuration on each validation season independently.

    For season ``V`` the run starts at ``V - history`` with every team at 1500 and
    proceeds chronologically, so a prediction for a game in ``V`` uses only games
    played before it.
    """
    assert_no_holdout(validation_seasons, where="evaluate_elo_config")
    by_season = by_season if by_season is not None else records_by_season(games)
    truths = truths if truths is not None else truths_by_season(games)
    folds: list[FoldResult] = []
    for season in validation_seasons:
        start = config.start_season(season, earliest_season)
        records: list[dict[str, Any]] = []
        for s in range(start, season + 1):
            records.extend(by_season.get(s, []))
        predictions = run_elo(records, config, predict_seasons=[season])
        if not predictions:
            continue
        truth = truths[season]
        prob = np.array([p.home_win_probability for p in predictions])
        y = np.array([truth[p.nba_game_id] for p in predictions], dtype=float)
        folds.append(
            FoldResult(
                season=season,
                brier_score=brier_score(y, prob),
                log_loss=log_loss(y, prob),
                n=len(prob),
            )
        )
    return CandidateResult(
        label=(
            f"K={config.k_factor:g}_HCA={config.home_advantage:g}"
            f"_reg={config.regression_factor:g}_hist={config.history}"
        ),
        config=config.to_dict(),
        folds=folds,
    )


def select_elo(
    games: pd.DataFrame,
    *,
    validation_seasons: Sequence[int] = DEVELOPMENT_VALIDATION_SEASONS,
) -> tuple[EloConfig, list[CandidateResult]]:
    """Evaluate the full Elo grid and return the winner plus every result."""
    assert_no_holdout(games["season"].unique(), where="select_elo(games)")
    earliest = int(games["season"].min())
    by_season = records_by_season(games)
    truths = truths_by_season(games)
    results = [
        evaluate_elo_config(
            games, config, validation_seasons, earliest,
            by_season=by_season, truths=truths,
        )
        for config in elo_grid()
    ]
    ranked = rank_candidates(results)
    best = _preferred_elo(tied_with_best(ranked))
    winner = EloConfig(
        k_factor=best.config["k_factor"],
        home_advantage=best.config["home_advantage"],
        regression_factor=best.config["regression_factor"],
        history=best.config["history"],
    )
    return winner, ranked


def _preferred_elo(tied: Sequence[CandidateResult]) -> CandidateResult:
    """Choose among statistically tied Elo configurations.

    Offseason regression makes old seasons decay geometrically, so every history
    length scores identically to ~1e-8. Among ties we prefer ``all_available``:
    it is the only choice that leaves ``elo_diff`` defined for *every* logistic
    training example regardless of that model's own training window, and it needs
    no per-fold re-initialisation. This preference is declared here, before the
    holdout is touched, so it cannot be reverse-engineered from holdout results.
    """
    if not tied:
        raise ValueError("no candidates to choose from")
    full_history = [c for c in tied if c.config["history"] == HISTORY_ALL]
    pool = full_history or list(tied)
    return min(pool, key=lambda c: (c.mean_brier, c.mean_log_loss))


# --- logistic selection ----------------------------------------------------


def logistic_grid() -> list[LogisticConfig]:
    """Finite-window and recency-weighted candidates across the C grid."""
    configs = [
        LogisticConfig(training_history=h, c_value=c)
        for h, c in itertools.product(LOGISTIC_HISTORY_GRID, LOGISTIC_C_GRID)
    ]
    configs += [
        LogisticConfig(training_history=HISTORY_WEIGHTED, c_value=c, half_life=hl)
        for hl, c in itertools.product(LOGISTIC_HALF_LIVES, LOGISTIC_C_GRID)
    ]
    return configs


def evaluate_logistic_config(
    features: pd.DataFrame,
    config: LogisticConfig,
    validation_seasons: Sequence[int],
) -> CandidateResult:
    """Score one logistic configuration, refitting per fold on prior seasons only."""
    assert_no_holdout(validation_seasons, where="evaluate_logistic_config")
    available = sorted(features["season"].unique().tolist())
    folds: list[FoldResult] = []
    for season in validation_seasons:
        seasons = training_seasons(config, season, available)
        training = features[features["season"].isin(seasons)]
        validation = features[features["season"] == season]
        if training.empty or validation.empty:
            continue
        fitted = fit_logistic(training, config)
        prob = fitted.predict_proba(validation)
        y = validation["home_win"].astype(int).to_numpy()
        folds.append(
            FoldResult(
                season=season,
                brier_score=brier_score(y, prob),
                log_loss=log_loss(y, prob),
                n=len(prob),
            )
        )
    return CandidateResult(label=config.label, config=config.to_dict(), folds=folds)


def select_logistic(
    features: pd.DataFrame,
    *,
    validation_seasons: Sequence[int] = DEVELOPMENT_VALIDATION_SEASONS,
) -> tuple[LogisticConfig, list[CandidateResult]]:
    """Evaluate the logistic grid and return the winner plus every result."""
    assert_no_holdout(features["season"].unique(), where="select_logistic(features)")
    results = [
        evaluate_logistic_config(features, config, validation_seasons)
        for config in logistic_grid()
    ]
    ranked = rank_candidates(results)
    best = _preferred_logistic(tied_with_best(ranked))
    winner = LogisticConfig(
        training_history=best.config["training_history"],
        c_value=best.config["C"],
        half_life=best.config["half_life_seasons"],
    )
    return winner, ranked


def _preferred_logistic(tied: Sequence[CandidateResult]) -> CandidateResult:
    """Choose among statistically tied logistic configurations.

    History strategy is what this phase is actually researching, so ties are
    broken on it first and toward the simpler description: a fixed window beats
    recency weighting (one parameter instead of two), and a shorter window beats
    a longer one. Regularisation is then settled by the primary metric, since no
    simplicity argument distinguishes one ``C`` from another.
    """
    if not tied:
        raise ValueError("no candidates to choose from")
    finite = [c for c in tied if isinstance(c.config["training_history"], int)]
    pool = finite or list(tied)
    shortest = min(
        c.config["training_history"] for c in pool if isinstance(c.config["training_history"], int)
    ) if finite else None
    if shortest is not None:
        pool = [c for c in pool if c.config["training_history"] == shortest]
    return min(pool, key=lambda c: (c.mean_brier, c.mean_log_loss))
