"""Logistic-regression baseline and its training-history policy.

All preprocessing lives inside a scikit-learn ``Pipeline`` so imputation and
scaling are fitted on the training split only. Nothing is imputed from
dataset-wide statistics -- doing so would leak validation and holdout
information into training.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Final

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from nba_prediction_market.features.feature_spec import (
    MODEL_FEATURES,
    TARGET,
    validate_feature_matrix,
)

#: Sentinel for "train on every prior season".
HISTORY_ALL: Final = "all_available"
#: Sentinel for "train on every prior season, weighted by recency".
HISTORY_WEIGHTED: Final = "weighted_all"

RANDOM_STATE: Final = 20260820


@dataclass(frozen=True)
class LogisticConfig:
    """A complete logistic specification."""

    training_history: int | str
    c_value: float
    half_life: float | None = None

    def __post_init__(self) -> None:
        if isinstance(self.training_history, int) and self.training_history < 1:
            raise ValueError(f"training_history must be >= 1, got {self.training_history}")
        if isinstance(self.training_history, str) and self.training_history not in {
            HISTORY_ALL,
            HISTORY_WEIGHTED,
        }:
            raise ValueError(f"unknown training_history {self.training_history!r}")
        if self.training_history == HISTORY_WEIGHTED and not self.half_life:
            raise ValueError("weighted history requires a half_life")
        if self.half_life is not None and self.half_life <= 0:
            raise ValueError(f"half_life must be positive, got {self.half_life}")
        if self.c_value <= 0:
            raise ValueError(f"C must be positive, got {self.c_value}")

    @property
    def label(self) -> str:
        if self.training_history == HISTORY_WEIGHTED:
            return f"weighted(hl={self.half_life:g})_C={self.c_value:g}"
        return f"history={self.training_history}_C={self.c_value:g}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "training_history": self.training_history,
            "half_life_seasons": self.half_life,
            "C": self.c_value,
        }


def build_pipeline(c_value: float) -> Pipeline:
    """Imputer -> scaler -> logistic regression, all fitted per training split."""
    return Pipeline(
        steps=[
            # Median imputation handles early-season nulls; fitted on train only.
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    C=c_value,
                    solver="lbfgs",
                    max_iter=2000,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def training_seasons(
    config: LogisticConfig, validation_season: int, available: Sequence[int]
) -> list[int]:
    """Seasons whose games may be used as supervised training examples.

    Strictly earlier than ``validation_season``. A finite window keeps the most
    recent N; the weighted and all-available strategies keep everything prior.
    """
    prior = sorted(s for s in available if s < validation_season)
    if config.training_history in {HISTORY_ALL, HISTORY_WEIGHTED}:
        return prior
    return prior[-int(config.training_history) :]


def recency_weights(
    seasons: np.ndarray, most_recent_season: int, half_life: float
) -> np.ndarray:
    """Exponentially decaying sample weights by season age.

    ``weight = 0.5 ** (age_in_seasons / half_life)``, so the most recent training
    season has weight 1.0 and older seasons decay smoothly.
    """
    age = most_recent_season - np.asarray(seasons, dtype=float)
    return np.power(0.5, age / float(half_life))


def feature_matrix(frame: pd.DataFrame) -> pd.DataFrame:
    """The allowlisted feature columns, validated."""
    missing = [c for c in MODEL_FEATURES if c not in frame.columns]
    if missing:
        raise ValueError(f"feature frame is missing required columns: {missing}")
    matrix = frame.loc[:, list(MODEL_FEATURES)].astype(float)
    validate_feature_matrix(list(matrix.columns))
    return matrix


@dataclass
class FittedLogistic:
    """A trained pipeline plus what it was trained on."""

    pipeline: Pipeline
    config: LogisticConfig
    training_seasons: list[int]
    n_training_rows: int

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        return self.pipeline.predict_proba(feature_matrix(frame))[:, 1]


def fit_logistic(
    training: pd.DataFrame, config: LogisticConfig
) -> FittedLogistic:
    """Fit the pipeline on ``training`` only, applying recency weights if configured."""
    if training.empty:
        raise ValueError("no training rows supplied")
    x = feature_matrix(training)
    y = training[TARGET].astype(int).to_numpy()

    fit_params: dict[str, Any] = {}
    if config.training_history == HISTORY_WEIGHTED:
        weights = recency_weights(
            training["season"].to_numpy(), int(training["season"].max()), config.half_life
        )
        # Routed to the estimator step by name, so the weights reach the model
        # rather than being silently ignored by the pipeline.
        fit_params["model__sample_weight"] = weights

    pipeline = build_pipeline(config.c_value)
    pipeline.fit(x, y, **fit_params)
    return FittedLogistic(
        pipeline=pipeline,
        config=config,
        training_seasons=sorted(training["season"].unique().tolist()),
        n_training_rows=len(training),
    )
