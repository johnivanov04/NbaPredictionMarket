"""Probabilistic forecasting metrics, calibration, and paired bootstrap.

Stored predictions are never mutated. Log loss needs probabilities strictly
inside ``(0, 1)``, so a tiny epsilon is applied **to a copy** at computation
time only -- the raw values on disk keep whatever the model produced.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Final

import numpy as np

#: Numerical-stability clip for log loss only. Small enough not to change any
#: reported figure at the precision we report, large enough to avoid infinities.
LOG_LOSS_EPSILON: Final = 1e-15

#: Fixed calibration bins: [0.0,0.1), [0.1,0.2), ..., [0.9,1.0].
CALIBRATION_BIN_EDGES: Final = tuple(round(0.1 * i, 1) for i in range(11))


def _as_arrays(y_true: Sequence[Any], y_prob: Sequence[Any]) -> tuple[np.ndarray, np.ndarray]:
    truth = np.asarray(y_true, dtype=float)
    prob = np.asarray(y_prob, dtype=float)
    if truth.shape != prob.shape:
        raise ValueError(f"shape mismatch: {truth.shape} vs {prob.shape}")
    if truth.size == 0:
        raise ValueError("no predictions to score")
    return truth, prob


def brier_score(y_true: Sequence[Any], y_prob: Sequence[Any]) -> float:
    """Mean squared error of the probability. Lower is better."""
    truth, prob = _as_arrays(y_true, y_prob)
    return float(np.mean((prob - truth) ** 2))


def brier_losses(y_true: Sequence[Any], y_prob: Sequence[Any]) -> np.ndarray:
    """Per-game squared error, for paired comparisons."""
    truth, prob = _as_arrays(y_true, y_prob)
    return (prob - truth) ** 2


def log_losses(y_true: Sequence[Any], y_prob: Sequence[Any]) -> np.ndarray:
    """Per-game negative log likelihood, clipped only for numerical stability."""
    truth, prob = _as_arrays(y_true, y_prob)
    safe = np.clip(prob, LOG_LOSS_EPSILON, 1.0 - LOG_LOSS_EPSILON)
    return -(truth * np.log(safe) + (1.0 - truth) * np.log(1.0 - safe))


def log_loss(y_true: Sequence[Any], y_prob: Sequence[Any]) -> float:
    return float(np.mean(log_losses(y_true, y_prob)))


def accuracy(y_true: Sequence[Any], y_prob: Sequence[Any], threshold: float = 0.5) -> float:
    truth, prob = _as_arrays(y_true, y_prob)
    return float(np.mean((prob >= threshold).astype(float) == truth))


def roc_auc(y_true: Sequence[Any], y_prob: Sequence[Any]) -> float | None:
    """Rank-based AUC. ``None`` when one class is absent."""
    truth, prob = _as_arrays(y_true, y_prob)
    positives, negatives = truth == 1, truth == 0
    if not positives.any() or not negatives.any():
        return None
    order = np.argsort(prob, kind="mergesort")
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(prob) + 1)
    # Average ranks within ties so equal probabilities do not bias the estimate.
    _, inverse, counts = np.unique(prob, return_inverse=True, return_counts=True)
    sums = np.zeros(len(counts))
    np.add.at(sums, inverse, ranks)
    ranks = (sums / counts)[inverse]
    n_pos, n_neg = int(positives.sum()), int(negatives.sum())
    return float((ranks[positives].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def summary(y_true: Sequence[Any], y_prob: Sequence[Any]) -> dict[str, Any]:
    """Every headline metric for one predictor."""
    truth, prob = _as_arrays(y_true, y_prob)
    return {
        "n": int(truth.size),
        "brier_score": brier_score(truth, prob),
        "log_loss": log_loss(truth, prob),
        "accuracy": accuracy(truth, prob),
        "roc_auc": roc_auc(truth, prob),
        "mean_predicted_probability": float(np.mean(prob)),
        "actual_home_win_rate": float(np.mean(truth)),
    }


def calibration_table(
    y_true: Sequence[Any], y_prob: Sequence[Any]
) -> list[dict[str, Any]]:
    """Fixed-bin calibration. The final bin is closed so 1.0 is included."""
    truth, prob = _as_arrays(y_true, y_prob)
    rows: list[dict[str, Any]] = []
    for index in range(len(CALIBRATION_BIN_EDGES) - 1):
        low, high = CALIBRATION_BIN_EDGES[index], CALIBRATION_BIN_EDGES[index + 1]
        mask = (prob >= low) & (prob < high) if index < 9 else (prob >= low) & (prob <= high)
        count = int(mask.sum())
        mean_pred = float(np.mean(prob[mask])) if count else None
        actual = float(np.mean(truth[mask])) if count else None
        rows.append(
            {
                "bin": f"[{low:.1f},{high:.1f}{')' if index < 9 else ']'}",
                "bin_low": low,
                "bin_high": high,
                "count": count,
                "mean_prediction": mean_pred,
                "actual_home_win_rate": actual,
                "calibration_gap": None if count == 0 else actual - mean_pred,
                # Extreme bins are often sparse; flag rather than over-read them.
                "sparse": count > 0 and count < 30,
            }
        )
    return rows


def expected_calibration_error(y_true: Sequence[Any], y_prob: Sequence[Any]) -> float:
    """Count-weighted mean absolute calibration gap across the fixed bins."""
    truth, _ = _as_arrays(y_true, y_prob)
    total = 0.0
    for row in calibration_table(y_true, y_prob):
        if row["count"]:
            total += row["count"] * abs(row["calibration_gap"])
    return float(total / truth.size)


def paired_bootstrap(
    losses_model: np.ndarray,
    losses_benchmark: np.ndarray,
    *,
    n_resamples: int = 10_000,
    seed: int = 20260820,
    confidence: float = 0.95,
) -> dict[str, Any]:
    """Paired bootstrap CI for ``mean(model) - mean(benchmark)``.

    Sign convention: **negative means the model has lower (better) loss than the
    benchmark.** Resampling is paired -- the same game indices are drawn for both
    predictors, which is what makes the comparison valid on a shared game set.
    """
    model = np.asarray(losses_model, dtype=float)
    benchmark = np.asarray(losses_benchmark, dtype=float)
    if model.shape != benchmark.shape:
        raise ValueError(f"paired inputs must align: {model.shape} vs {benchmark.shape}")
    differences = model - benchmark
    rng = np.random.default_rng(seed)
    n = differences.size
    indices = rng.integers(0, n, size=(n_resamples, n))
    means = differences[indices].mean(axis=1)
    low = float(np.quantile(means, (1.0 - confidence) / 2.0))
    high = float(np.quantile(means, 1.0 - (1.0 - confidence) / 2.0))
    return {
        "mean_loss_difference": float(differences.mean()),
        "ci_low": low,
        "ci_high": high,
        "confidence": confidence,
        "n_resamples": n_resamples,
        "seed": seed,
        "n_games": int(n),
        "favours_model": high < 0.0,
        "favours_benchmark": low > 0.0,
        "inconclusive": low <= 0.0 <= high,
        "sign_convention": "model_minus_benchmark < 0 means the model has lower (better) loss",
    }
