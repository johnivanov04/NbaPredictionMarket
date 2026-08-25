"""Leakage-safe probability calibration for Phase 3A4.

Boosted trees often rank well while producing miscalibrated probabilities, and
Brier punishes miscalibration directly. Fixing that needs a calibrator -- which
is exactly the kind of component that leaks if fitted carelessly, because the
obvious thing to do is fit it on the predictions you are about to score.

The construction here never does that. For an outer validation season, the
calibrator is fitted on **chronological out-of-fold predictions generated
entirely inside that fold's training history**:

    training seasons  S1 S2 S3 S4        outer validation  S5
                      |  |  |  |
    inner folds:      fit(S1)      -> predict S2
                      fit(S1,S2)   -> predict S3
                      fit(S1..S3)  -> predict S4
                      ------------------------------------
                      those predictions and their outcomes fit the calibrator
                      -> frozen -> applied to S5

Each inner prediction is made by a model that never saw the season it scores,
so the calibrator sees honest out-of-sample probabilities. S5 contributes
nothing to its own calibration, and the holdout contributes to nothing.

The earliest training season cannot be scored out-of-fold (there is nothing
before it), so it supplies training examples only.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Final

import numpy as np
import pandas as pd

#: Below this many out-of-fold rows an isotonic fit is too unstable to trust.
MIN_ISOTONIC_ROWS: Final = 1500
#: Below this many, skip calibration altogether and keep raw probabilities.
MIN_CALIBRATION_ROWS: Final = 400

IDENTITY: Final = "none"
SIGMOID: Final = "sigmoid"
ISOTONIC: Final = "isotonic"
METHODS: Final[tuple[str, ...]] = (IDENTITY, SIGMOID, ISOTONIC)

_EPSILON: Final = 1e-6


@dataclass
class Calibrator:
    """A fitted probability transform, or a documented refusal to fit one."""

    method: str
    fitted: Any = None
    n_rows: int = 0
    reason: str | None = None

    def transform(self, probability: np.ndarray) -> np.ndarray:
        values = np.clip(np.asarray(probability, dtype=float), _EPSILON, 1 - _EPSILON)
        if self.method == IDENTITY or self.fitted is None:
            return values
        if self.method == SIGMOID:
            logit = np.log(values / (1.0 - values)).reshape(-1, 1)
            return self.fitted.predict_proba(logit)[:, 1]
        return np.clip(self.fitted.predict(values), _EPSILON, 1 - _EPSILON)

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "n_calibration_rows": self.n_rows,
            "fitted": self.fitted is not None,
            "reason": self.reason,
        }


def fit_calibrator(
    method: str, probability: Sequence[float], outcome: Sequence[int]
) -> Calibrator:
    """Fit one calibrator on out-of-fold predictions.

    Refuses rather than fits when support is thin: a calibrator estimated from
    a few hundred rows can easily be worse than leaving the probabilities
    alone, and an isotonic step function needs more support still.
    """
    values = np.asarray(probability, dtype=float)
    truth = np.asarray(outcome, dtype=int)
    n = int(values.size)

    if method == IDENTITY:
        return Calibrator(IDENTITY, None, n, "raw probabilities kept by design")
    if n < MIN_CALIBRATION_ROWS:
        return Calibrator(IDENTITY, None, n, f"only {n} rows; below the floor")
    if method == ISOTONIC and n < MIN_ISOTONIC_ROWS:
        return Calibrator(IDENTITY, None, n, f"only {n} rows for isotonic")
    if len(np.unique(truth)) < 2:
        return Calibrator(IDENTITY, None, n, "calibration set has one class")

    if method == SIGMOID:
        from sklearn.linear_model import LogisticRegression

        clipped = np.clip(values, _EPSILON, 1 - _EPSILON)
        logit = np.log(clipped / (1.0 - clipped)).reshape(-1, 1)
        model = LogisticRegression(C=1e10, solver="lbfgs")
        model.fit(logit, truth)
        return Calibrator(SIGMOID, model, n)

    if method == ISOTONIC:
        from sklearn.isotonic import IsotonicRegression

        model = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        model.fit(values, truth)
        return Calibrator(ISOTONIC, model, n)

    raise ValueError(f"unknown calibration method {method!r}")


def chronological_oof_predictions(
    frame: pd.DataFrame,
    training_seasons: Sequence[int],
    fit_predict: Any,
    *,
    target: str = "home_win",
) -> pd.DataFrame:
    """Out-of-fold predictions generated strictly inside the training history.

    Walks the training seasons forward: each is predicted by a model fitted only
    on the seasons before it. The first has no predecessor, so it contributes
    training rows but never a prediction -- the alternative would be to score it
    with a model that had seen it.

    ``fit_predict(train_frame, predict_frame) -> probabilities``.
    """
    seasons = sorted(training_seasons)
    rows: list[pd.DataFrame] = []
    for index, season in enumerate(seasons):
        if index == 0:
            continue
        inner_train = frame[frame["season"].isin(seasons[:index])]
        inner_valid = frame[frame["season"] == season]
        if inner_train.empty or inner_valid.empty:
            continue
        probability = fit_predict(inner_train, inner_valid)
        rows.append(
            pd.DataFrame({
                "season": season,
                "probability": np.asarray(probability, dtype=float),
                target: inner_valid[target].astype(int).to_numpy(),
            })
        )
    if not rows:
        return pd.DataFrame(columns=["season", "probability", target])
    return pd.concat(rows, ignore_index=True)
