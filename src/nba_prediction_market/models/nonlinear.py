"""Gradient-boosted model configurations for Phase 3A4.

Roughly 6,000 training games with a home-court base rate is a small problem, so
overfitting is the live danger and depth is the main way to invite it. Every
configuration here is therefore shallow (2-3), slow-learning, and regularised,
and the grid is a **compact hand-picked set rather than a Cartesian product** --
the full cross of the candidate ranges would be 128 fits per fold per feature
set, which is a search large enough to find something by chance.

A second family (sklearn's ``HistGradientBoostingClassifier``) is included for
one reason only: to tell "gradient boosting extracts interactions here" apart
from "this particular XGBoost configuration got lucky". It is not model
proliferation and it is discarded if it clearly underperforms.

Seeds are fixed and tree methods deterministic, so a rerun reproduces the
numbers exactly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Final

#: One seed for every fit in the phase.
RANDOM_SEED: Final = 20260824

#: Ranges the grid is drawn from, recorded so the report can state what was
#: considered as well as what was tried.
CANDIDATE_RANGES: Final[dict[str, tuple[Any, ...]]] = {
    "max_depth": (2, 3),
    "learning_rate": (0.02, 0.05),
    "n_estimators": (200, 400),
    "min_child_weight": (5, 20),
    "subsample": (0.8, 1.0),
    "colsample_bytree": (0.8, 1.0),
    "reg_lambda": (1, 10),
}


@dataclass(frozen=True)
class NonlinearConfig:
    """One fully specified gradient-boosted model."""

    name: str
    family: str
    params: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "family": self.family, "params": dict(self.params)}


def _xgb(name: str, **params: Any) -> NonlinearConfig:
    base = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "tree_method": "hist",
        "random_state": RANDOM_SEED,
        "n_jobs": 1,
        "verbosity": 0,
    }
    return NonlinearConfig(name, "xgboost", {**base, **params})


#: Sixteen configurations spanning the ranges above. Depth and learning rate
#: move together with tree count so that total capacity stays comparable, and
#: the regularisation knobs are varied one axis at a time rather than crossed.
XGBOOST_GRID: Final[tuple[NonlinearConfig, ...]] = (
    _xgb("xgb01", max_depth=2, learning_rate=0.02, n_estimators=400,
         min_child_weight=5, subsample=0.8, colsample_bytree=0.8, reg_lambda=1),
    _xgb("xgb02", max_depth=2, learning_rate=0.02, n_estimators=400,
         min_child_weight=20, subsample=0.8, colsample_bytree=0.8, reg_lambda=10),
    _xgb("xgb03", max_depth=2, learning_rate=0.05, n_estimators=200,
         min_child_weight=5, subsample=0.8, colsample_bytree=0.8, reg_lambda=1),
    _xgb("xgb04", max_depth=2, learning_rate=0.05, n_estimators=200,
         min_child_weight=20, subsample=0.8, colsample_bytree=0.8, reg_lambda=10),
    _xgb("xgb05", max_depth=2, learning_rate=0.05, n_estimators=400,
         min_child_weight=20, subsample=1.0, colsample_bytree=1.0, reg_lambda=10),
    _xgb("xgb06", max_depth=2, learning_rate=0.02, n_estimators=200,
         min_child_weight=20, subsample=1.0, colsample_bytree=0.8, reg_lambda=10),
    _xgb("xgb07", max_depth=3, learning_rate=0.02, n_estimators=400,
         min_child_weight=5, subsample=0.8, colsample_bytree=0.8, reg_lambda=1),
    _xgb("xgb08", max_depth=3, learning_rate=0.02, n_estimators=400,
         min_child_weight=20, subsample=0.8, colsample_bytree=0.8, reg_lambda=10),
    _xgb("xgb09", max_depth=3, learning_rate=0.05, n_estimators=200,
         min_child_weight=5, subsample=0.8, colsample_bytree=0.8, reg_lambda=1),
    _xgb("xgb10", max_depth=3, learning_rate=0.05, n_estimators=200,
         min_child_weight=20, subsample=0.8, colsample_bytree=0.8, reg_lambda=10),
    _xgb("xgb11", max_depth=3, learning_rate=0.05, n_estimators=400,
         min_child_weight=20, subsample=1.0, colsample_bytree=1.0, reg_lambda=10),
    _xgb("xgb12", max_depth=3, learning_rate=0.02, n_estimators=200,
         min_child_weight=20, subsample=1.0, colsample_bytree=0.8, reg_lambda=10),
    _xgb("xgb13", max_depth=2, learning_rate=0.05, n_estimators=400,
         min_child_weight=5, subsample=0.8, colsample_bytree=1.0, reg_lambda=1),
    _xgb("xgb14", max_depth=3, learning_rate=0.05, n_estimators=400,
         min_child_weight=5, subsample=0.8, colsample_bytree=1.0, reg_lambda=1),
    _xgb("xgb15", max_depth=2, learning_rate=0.02, n_estimators=200,
         min_child_weight=5, subsample=1.0, colsample_bytree=1.0, reg_lambda=10),
    _xgb("xgb16", max_depth=3, learning_rate=0.02, n_estimators=200,
         min_child_weight=5, subsample=1.0, colsample_bytree=1.0, reg_lambda=10),
)

#: A deliberately tiny second family. Four configurations, same shallow spirit.
HISTGB_GRID: Final[tuple[NonlinearConfig, ...]] = (
    NonlinearConfig("hgb01", "histgb", {
        "max_depth": 2, "learning_rate": 0.05, "max_iter": 200,
        "min_samples_leaf": 40, "l2_regularization": 1.0,
        "early_stopping": False, "random_state": RANDOM_SEED,
    }),
    NonlinearConfig("hgb02", "histgb", {
        "max_depth": 3, "learning_rate": 0.05, "max_iter": 200,
        "min_samples_leaf": 40, "l2_regularization": 10.0,
        "early_stopping": False, "random_state": RANDOM_SEED,
    }),
    NonlinearConfig("hgb03", "histgb", {
        "max_depth": 2, "learning_rate": 0.02, "max_iter": 400,
        "min_samples_leaf": 80, "l2_regularization": 10.0,
        "early_stopping": False, "random_state": RANDOM_SEED,
    }),
    NonlinearConfig("hgb04", "histgb", {
        "max_depth": 3, "learning_rate": 0.02, "max_iter": 400,
        "min_samples_leaf": 80, "l2_regularization": 1.0,
        "early_stopping": False, "random_state": RANDOM_SEED,
    }),
)

ALL_CONFIGS: Final[tuple[NonlinearConfig, ...]] = XGBOOST_GRID + HISTGB_GRID
CONFIGS_BY_NAME: Final[dict[str, NonlinearConfig]] = {
    c.name: c for c in ALL_CONFIGS
}


def build_estimator(config: NonlinearConfig) -> Any:
    """Instantiate the estimator for a configuration.

    Both families handle NaN natively, so no imputer is fitted here -- which
    also means no imputation statistic can cross the train/validation boundary.
    Trees are scale-invariant, so no scaler either.
    """
    if config.family == "xgboost":
        from xgboost import XGBClassifier

        return XGBClassifier(**config.params)
    if config.family == "histgb":
        from sklearn.ensemble import HistGradientBoostingClassifier

        return HistGradientBoostingClassifier(**config.params)
    raise ValueError(f"unknown model family {config.family!r}")
