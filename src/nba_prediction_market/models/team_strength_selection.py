"""Phase 3A2 development selection: ratings, bundles, and blends.

Everything here runs on pre-holdout seasons only. :func:`assert_no_holdout`
guards each entry point, exactly as in Phase 3A1.
"""

from __future__ import annotations

import itertools
from collections.abc import Sequence
from typing import Any, Final

import numpy as np
import pandas as pd

from nba_prediction_market.features.elo import HISTORY_ALL, EloConfig, run_elo
from nba_prediction_market.features.mov_elo import MOV_MULTIPLIERS, MovEloConfig, run_mov_elo
from nba_prediction_market.models.bundles import Bundle
from nba_prediction_market.models.logistic import LogisticConfig, fit_logistic, training_seasons
from nba_prediction_market.models.metrics import brier_score, log_loss, roc_auc
from nba_prediction_market.models.selection import (
    DEVELOPMENT_VALIDATION_SEASONS,
    CandidateResult,
    FoldResult,
    assert_no_holdout,
    rank_candidates,
    records_by_season,
    truths_by_season,
)

#: Phase 3A1 chose HCA=40 at the low edge of its grid, so the development grid is
#: extended downward. K and regression stay near the Phase 3A1 result.
REFINED_HCA_GRID: Final[tuple[float, ...]] = (0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0)
REFINED_K_GRID: Final[tuple[float, ...]] = (10.0, 20.0, 30.0)
REFINED_REGRESSION_GRID: Final[tuple[float, ...]] = (0.25, 0.50, 0.75)

#: Deliberately small: three defensible margin formulations, not a search.
MOV_K_GRID: Final[tuple[float, ...]] = (10.0, 20.0, 30.0)
MOV_REGRESSION_GRID: Final[tuple[float, ...]] = (0.50, 0.75)

EWMA_HALF_LIFE_GRID: Final[tuple[float, ...]] = (3.0, 5.0, 10.0)
BLEND_WEIGHT_GRID: Final[tuple[float, ...]] = (0.0, 0.25, 0.50, 0.75, 1.0)

#: Logistic configuration held fixed while bundles are compared, so the ablation
#: measures the features rather than a lucky hyperparameter pairing.
BUNDLE_PROBE_CONFIG: Final = LogisticConfig(training_history=5, c_value=1.0)


def _folds_from(
    frame: pd.DataFrame,
    validation_seasons: Sequence[int],
    predict: Any,
) -> list[FoldResult]:
    folds: list[FoldResult] = []
    for season in validation_seasons:
        validation = frame[frame["season"] == season]
        if validation.empty:
            continue
        prob = predict(season, validation)
        if prob is None:
            continue
        y = validation["home_win"].astype(int).to_numpy()
        folds.append(
            FoldResult(
                season=season,
                brier_score=brier_score(y, prob),
                log_loss=log_loss(y, prob),
                n=len(prob),
            )
        )
    return folds


# --- refined binary Elo ----------------------------------------------------


def refined_elo_grid() -> list[EloConfig]:
    return [
        EloConfig(k_factor=k, home_advantage=h, regression_factor=r, history=HISTORY_ALL)
        for k, h, r in itertools.product(
            REFINED_K_GRID, REFINED_HCA_GRID, REFINED_REGRESSION_GRID
        )
    ]


def select_refined_elo(
    games: pd.DataFrame,
    *,
    validation_seasons: Sequence[int] = DEVELOPMENT_VALIDATION_SEASONS,
) -> tuple[EloConfig, list[CandidateResult]]:
    """Re-search binary Elo with home-court advantage extended down to zero."""
    assert_no_holdout(games["season"].unique(), where="select_refined_elo")
    by_season = records_by_season(games)
    truths = truths_by_season(games)
    earliest = int(games["season"].min())

    results: list[CandidateResult] = []
    for config in refined_elo_grid():
        folds: list[FoldResult] = []
        for season in validation_seasons:
            records: list[dict[str, Any]] = []
            for s in range(config.start_season(season, earliest), season + 1):
                records.extend(by_season.get(s, []))
            predictions = run_elo(records, config, predict_seasons=[season])
            if not predictions:
                continue
            truth = truths[season]
            prob = np.array([p.home_win_probability for p in predictions])
            y = np.array([truth[p.nba_game_id] for p in predictions], dtype=float)
            folds.append(
                FoldResult(season, brier_score(y, prob), log_loss(y, prob), len(prob))
            )
        results.append(
            CandidateResult(
                label=f"K={config.k_factor:g}_HCA={config.home_advantage:g}"
                f"_reg={config.regression_factor:g}",
                config=config.to_dict(),
                folds=folds,
            )
        )
    ranked = rank_candidates(results)
    best = ranked[0].config
    return (
        EloConfig(
            k_factor=best["k_factor"],
            home_advantage=best["home_advantage"],
            regression_factor=best["regression_factor"],
            history=HISTORY_ALL,
        ),
        ranked,
    )


# --- margin-of-victory Elo -------------------------------------------------


def mov_elo_grid(hca_grid: Sequence[float]) -> list[MovEloConfig]:
    return [
        MovEloConfig(
            k_factor=k, home_advantage=h, regression_factor=r,
            multiplier=m, history=HISTORY_ALL,
        )
        for m, k, h, r in itertools.product(
            MOV_MULTIPLIERS, MOV_K_GRID, hca_grid, MOV_REGRESSION_GRID
        )
    ]


def select_mov_elo(
    games: pd.DataFrame,
    *,
    hca_grid: Sequence[float],
    validation_seasons: Sequence[int] = DEVELOPMENT_VALIDATION_SEASONS,
) -> tuple[MovEloConfig, list[CandidateResult]]:
    """Search the margin-aware Elo variants on development folds only."""
    assert_no_holdout(games["season"].unique(), where="select_mov_elo")
    by_season = records_by_season(games)
    truths = truths_by_season(games)
    earliest = int(games["season"].min())

    results: list[CandidateResult] = []
    for config in mov_elo_grid(hca_grid):
        folds: list[FoldResult] = []
        for season in validation_seasons:
            records: list[dict[str, Any]] = []
            for s in range(config.start_season(season, earliest), season + 1):
                records.extend(by_season.get(s, []))
            predictions = run_mov_elo(records, config, predict_seasons=[season])
            if not predictions:
                continue
            truth = truths[season]
            prob = np.array([p.home_win_probability for p in predictions])
            y = np.array([truth[p.nba_game_id] for p in predictions], dtype=float)
            folds.append(
                FoldResult(season, brier_score(y, prob), log_loss(y, prob), len(prob))
            )
        results.append(
            CandidateResult(
                label=f"{config.multiplier}_K={config.k_factor:g}"
                f"_HCA={config.home_advantage:g}_reg={config.regression_factor:g}",
                config=config.to_dict(),
                folds=folds,
            )
        )
    ranked = rank_candidates(results)
    best = ranked[0].config
    return (
        MovEloConfig(
            k_factor=best["k_factor"],
            home_advantage=best["home_advantage"],
            regression_factor=best["regression_factor"],
            multiplier=best["multiplier"],
            history=HISTORY_ALL,
        ),
        ranked,
    )


# --- bundle ablation -------------------------------------------------------


def evaluate_bundle(
    features: pd.DataFrame,
    bundle: Bundle,
    config: LogisticConfig,
    validation_seasons: Sequence[int] = DEVELOPMENT_VALIDATION_SEASONS,
) -> CandidateResult:
    """Score one bundle with a fixed logistic configuration."""
    assert_no_holdout(features["season"].unique(), where="evaluate_bundle")
    available = sorted(features["season"].unique().tolist())
    aucs: list[float] = []

    def predict(season: int, validation: pd.DataFrame) -> np.ndarray | None:
        seasons = training_seasons(config, season, available)
        training = features[features["season"].isin(seasons)]
        if training.empty:
            return None
        fitted = fit_logistic(training, config, bundle.features)
        prob = fitted.predict_proba(validation)
        auc = roc_auc(validation["home_win"].astype(int).to_numpy(), prob)
        if auc is not None:
            aucs.append(auc)
        return prob

    folds = _folds_from(features, validation_seasons, predict)
    return CandidateResult(
        label=f"bundle_{bundle.name}",
        config={
            **bundle.to_dict(),
            "logistic": config.to_dict(),
            "mean_roc_auc": float(np.mean(aucs)) if aucs else None,
        },
        folds=folds,
    )


def evaluate_logistic_on_bundle(
    features: pd.DataFrame,
    bundle: Bundle,
    config: LogisticConfig,
    validation_seasons: Sequence[int] = DEVELOPMENT_VALIDATION_SEASONS,
) -> CandidateResult:
    """Score one logistic configuration against a fixed bundle."""
    result = evaluate_bundle(features, bundle, config, validation_seasons)
    return CandidateResult(
        label=f"{bundle.name}_{config.label}", config=result.config, folds=result.folds
    )


# --- blending --------------------------------------------------------------


def blend(primary: np.ndarray, secondary: np.ndarray, weight: float) -> np.ndarray:
    """``weight`` is the share given to ``secondary``."""
    if not 0.0 <= weight <= 1.0:
        raise ValueError(f"blend weight must be within [0, 1], got {weight}")
    return (1.0 - weight) * np.asarray(primary) + weight * np.asarray(secondary)


def evaluate_blend(
    y_by_season: dict[int, np.ndarray],
    primary_by_season: dict[int, np.ndarray],
    secondary_by_season: dict[int, np.ndarray],
    weight: float,
    label: str,
) -> CandidateResult:
    """Score one blend weight across development folds."""
    assert_no_holdout(list(y_by_season), where="evaluate_blend")
    folds = []
    for season in sorted(y_by_season):
        y = y_by_season[season]
        prob = blend(primary_by_season[season], secondary_by_season[season], weight)
        folds.append(FoldResult(season, brier_score(y, prob), log_loss(y, prob), len(y)))
    return CandidateResult(
        label=f"{label}_w={weight:g}", config={"partner": label, "weight": weight}, folds=folds
    )
