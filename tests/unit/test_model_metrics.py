"""Metrics, calibration, and the paired bootstrap."""

from __future__ import annotations

import numpy as np
import pytest

from nba_prediction_market.models.metrics import (
    CALIBRATION_BIN_EDGES,
    LOG_LOSS_EPSILON,
    accuracy,
    brier_losses,
    brier_score,
    calibration_table,
    expected_calibration_error,
    log_loss,
    log_losses,
    paired_bootstrap,
    roc_auc,
    summary,
)

# --- Brier -----------------------------------------------------------------


def test_a_perfect_forecast_scores_zero() -> None:
    assert brier_score([1, 0, 1], [1.0, 0.0, 1.0]) == 0.0


def test_a_maximally_wrong_forecast_scores_one() -> None:
    assert brier_score([1, 0], [0.0, 1.0]) == 1.0


def test_always_predicting_a_half_scores_a_quarter() -> None:
    assert brier_score([1, 0, 1, 0], [0.5] * 4) == 0.25


def test_brier_is_the_mean_squared_error() -> None:
    assert brier_score([1, 0], [0.8, 0.3]) == pytest.approx((0.04 + 0.09) / 2)


def test_per_game_brier_losses_average_to_the_score() -> None:
    y, p = [1, 0, 1, 0], [0.7, 0.2, 0.9, 0.4]
    assert brier_losses(y, p).mean() == pytest.approx(brier_score(y, p))


# --- log loss --------------------------------------------------------------


def test_log_loss_of_a_confident_correct_forecast_is_near_zero() -> None:
    assert log_loss([1], [0.999]) < 0.002


def test_log_loss_at_a_half_is_ln_two() -> None:
    assert log_loss([1, 0], [0.5, 0.5]) == pytest.approx(np.log(2))


def test_a_confidently_wrong_forecast_is_finite_not_infinite() -> None:
    """Epsilon clipping exists only for numerical stability."""
    value = log_loss([1], [0.0])
    assert np.isfinite(value)
    assert value == pytest.approx(-np.log(LOG_LOSS_EPSILON))


def test_clipping_does_not_mutate_the_inputs() -> None:
    prob = np.array([0.0, 1.0, 0.5])
    original = prob.copy()
    log_losses([1, 0, 1], prob)
    assert np.array_equal(prob, original)


def test_epsilon_is_small_enough_to_be_invisible_at_reporting_precision() -> None:
    assert round(log_loss([1, 0], [0.7, 0.3]), 6) == round(
        float(-(np.log(0.7) + np.log(0.7)) / 2), 6
    )


# --- accuracy and AUC ------------------------------------------------------


def test_accuracy_uses_a_half_threshold() -> None:
    assert accuracy([1, 1, 0, 0], [0.6, 0.4, 0.4, 0.6]) == 0.5
    assert accuracy([1, 1, 0, 0], [0.9, 0.8, 0.2, 0.1]) == 1.0


def test_auc_of_a_perfect_ranking_is_one() -> None:
    assert roc_auc([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]) == pytest.approx(1.0)


def test_auc_of_a_constant_forecast_is_a_half() -> None:
    assert roc_auc([0, 1, 0, 1], [0.5] * 4) == pytest.approx(0.5)


def test_auc_is_none_when_one_class_is_absent() -> None:
    assert roc_auc([1, 1, 1], [0.2, 0.5, 0.9]) is None


def test_summary_reports_every_headline_metric() -> None:
    result = summary([1, 0, 1, 0], [0.7, 0.3, 0.6, 0.4])
    assert set(result) == {
        "n", "brier_score", "log_loss", "accuracy", "roc_auc",
        "mean_predicted_probability", "actual_home_win_rate",
    }
    assert result["n"] == 4
    assert result["actual_home_win_rate"] == 0.5
    assert result["mean_predicted_probability"] == pytest.approx(0.5)


def test_mismatched_lengths_are_rejected() -> None:
    with pytest.raises(ValueError, match="shape mismatch"):
        brier_score([1, 0], [0.5])


def test_empty_input_is_rejected() -> None:
    with pytest.raises(ValueError, match="no predictions"):
        brier_score([], [])


# --- calibration -----------------------------------------------------------


def test_bins_are_the_documented_deciles() -> None:
    assert CALIBRATION_BIN_EDGES == (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
    assert len(calibration_table([1, 0], [0.5, 0.5])) == 10


def test_a_prediction_of_one_lands_in_the_final_closed_bin() -> None:
    table = calibration_table([1], [1.0])
    assert table[-1]["count"] == 1
    assert table[-1]["bin"].endswith("]")


def test_calibration_gap_is_actual_minus_predicted() -> None:
    table = calibration_table([1, 1, 0, 0], [0.85, 0.85, 0.85, 0.85])
    row = next(r for r in table if r["count"])
    assert row["mean_prediction"] == pytest.approx(0.85)
    assert row["actual_home_win_rate"] == pytest.approx(0.5)
    assert row["calibration_gap"] == pytest.approx(-0.35)


def test_empty_bins_report_null_rather_than_zero() -> None:
    table = calibration_table([1, 0], [0.55, 0.55])
    empty = [r for r in table if r["count"] == 0]
    assert empty
    assert all(r["mean_prediction"] is None and r["calibration_gap"] is None for r in empty)


def test_sparse_bins_are_flagged() -> None:
    table = calibration_table([1] * 5 + [0] * 100, [0.95] * 5 + [0.45] * 100)
    sparse = [r for r in table if r["sparse"]]
    assert sparse, "a 5-game bin should be flagged as sparse"


def test_a_perfectly_calibrated_forecast_has_zero_ece() -> None:
    y = [1] * 30 + [0] * 70
    p = [0.3] * 100
    assert expected_calibration_error(y, p) == pytest.approx(0.0, abs=1e-12)


def test_ece_is_the_count_weighted_absolute_gap() -> None:
    y = [1, 1, 0, 0]
    p = [0.9, 0.9, 0.9, 0.9]
    assert expected_calibration_error(y, p) == pytest.approx(0.4)


# --- paired bootstrap ------------------------------------------------------


def test_a_better_model_gives_a_negative_difference() -> None:
    """Sign convention: negative means the model beats the benchmark."""
    result = paired_bootstrap(
        np.full(200, 0.10), np.full(200, 0.20), n_resamples=1000
    )
    assert result["mean_loss_difference"] == pytest.approx(-0.10)
    assert result["favours_model"] is True
    assert result["favours_benchmark"] is False
    assert "model_minus_benchmark < 0" in result["sign_convention"]


def test_a_worse_model_gives_a_positive_difference() -> None:
    result = paired_bootstrap(np.full(200, 0.30), np.full(200, 0.20), n_resamples=1000)
    assert result["mean_loss_difference"] == pytest.approx(0.10)
    assert result["favours_benchmark"] is True


def test_an_indistinguishable_pair_is_reported_inconclusive() -> None:
    rng = np.random.default_rng(1)
    a = rng.normal(0.2, 0.1, 500)
    result = paired_bootstrap(a, a + rng.normal(0, 1e-6, 500), n_resamples=2000)
    assert result["inconclusive"] is True
    assert result["ci_low"] <= 0.0 <= result["ci_high"]


def test_the_bootstrap_is_reproducible_from_its_seed() -> None:
    rng = np.random.default_rng(3)
    a, b = rng.normal(0.2, 0.05, 300), rng.normal(0.21, 0.05, 300)
    first = paired_bootstrap(a, b, n_resamples=2000, seed=42)
    second = paired_bootstrap(a, b, n_resamples=2000, seed=42)
    assert first == second


def test_a_different_seed_changes_the_interval() -> None:
    rng = np.random.default_rng(3)
    a, b = rng.normal(0.2, 0.05, 300), rng.normal(0.21, 0.05, 300)
    assert paired_bootstrap(a, b, n_resamples=2000, seed=1)["ci_low"] != paired_bootstrap(
        a, b, n_resamples=2000, seed=2
    )["ci_low"]


def test_the_interval_brackets_the_observed_mean() -> None:
    rng = np.random.default_rng(5)
    a, b = rng.normal(0.20, 0.05, 400), rng.normal(0.22, 0.05, 400)
    result = paired_bootstrap(a, b, n_resamples=5000)
    assert result["ci_low"] < result["mean_loss_difference"] < result["ci_high"]


def test_misaligned_pairs_are_rejected() -> None:
    with pytest.raises(ValueError, match="paired inputs must align"):
        paired_bootstrap(np.zeros(10), np.zeros(9))
