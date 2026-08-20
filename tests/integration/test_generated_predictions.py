"""Invariants of the generated Phase 3A1 artefacts. Run with ``pytest -m dataset``."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from nba_prediction_market.features.feature_spec import MODEL_FEATURES, TARGET
from nba_prediction_market.models.selection import HOLDOUT_SEASON

pytestmark = pytest.mark.dataset

PROCESSED = Path("data/processed")
FEATURES = PROCESSED / "nba_model_features_2006_26.parquet"
PREDICTIONS = PROCESSED / "nba_predictions_2025_26.parquet"
KALSHI = PROCESSED / "nba_kalshi_pregame_t30_2025_26.parquet"
REPORT = Path("data/reports/model_baselines_2025_26.json")


def _load(path: Path) -> pd.DataFrame:
    if not path.is_file():
        pytest.skip(f"{path} not generated; run build_baselines first")
    return pd.read_parquet(path)


@pytest.fixture(scope="module")
def features() -> pd.DataFrame:
    return _load(FEATURES)


@pytest.fixture(scope="module")
def predictions() -> pd.DataFrame:
    return _load(PREDICTIONS)


@pytest.fixture(scope="module")
def report() -> dict:
    if not REPORT.is_file():
        pytest.skip("report not generated")
    return json.loads(REPORT.read_text())


# --- feature dataset -------------------------------------------------------


def test_one_feature_row_per_eligible_regular_season_game(features: pd.DataFrame) -> None:
    assert len(features) == 24038
    assert features["nba_game_id"].is_unique
    assert sorted(features["season"].unique()) == list(range(2006, 2026))


def test_every_allowlisted_feature_exists(features: pd.DataFrame) -> None:
    for column in MODEL_FEATURES:
        assert column in features.columns


def test_elo_is_present_for_every_game(features: pd.DataFrame) -> None:
    assert features["elo_diff"].notna().all()
    assert features["elo_probability"].between(0, 1).all()


def test_features_are_ordered_by_actual_tipoff(features: pd.DataFrame) -> None:
    ordered = features.sort_values(["game_datetime_utc", "nba_game_id"], kind="stable")
    assert list(ordered["nba_game_id"]) == list(features["nba_game_id"])


def test_first_game_of_each_team_season_has_null_rest(features: pd.DataFrame) -> None:
    """600 team-seasons = 30 teams x 20 seasons, each with one opener."""
    nulls = int(features["home_rest_days"].isna().sum() + features["away_rest_days"].isna().sum())
    assert nulls == 600
    openers = features[features["home_rest_days"].isna()]
    assert not openers["home_back_to_back"].any()


def test_rolling_features_are_null_only_for_a_teams_first_game(
    features: pd.DataFrame,
) -> None:
    assert int(features["home_last5_win_pct"].isna().sum()) == int(
        (features["home_games_played"] == 0).sum()
    )


def test_no_rest_value_spans_an_offseason(features: pd.DataFrame) -> None:
    """A season opener must never show ~150 days of 'rest'."""
    assert float(features["home_rest_days"].max()) < 150.0


# --- holdout predictions ---------------------------------------------------


def test_exactly_1230_holdout_predictions(predictions: pd.DataFrame) -> None:
    assert len(predictions) == 1230
    assert predictions["nba_game_id"].is_unique
    assert set(predictions["season"].unique()) == {HOLDOUT_SEASON}


def test_prediction_ids_match_the_phase_2_dataset_exactly(
    predictions: pd.DataFrame,
) -> None:
    kalshi = _load(KALSHI)
    assert set(predictions["nba_game_id"]) == set(kalshi["nba_game_id"])


@pytest.mark.parametrize(
    "column", ["constant_probability", "elo_probability", "logistic_probability",
               "kalshi_home_probability_normalized"]
)
def test_every_probability_is_within_the_unit_interval(
    predictions: pd.DataFrame, column: str
) -> None:
    values = predictions[column]
    assert values.notna().all()
    assert values.between(0.0, 1.0).all()


def test_the_constant_baseline_is_a_single_pre_holdout_rate(
    predictions: pd.DataFrame, report: dict
) -> None:
    assert predictions["constant_probability"].nunique() == 1
    frozen = report["frozen_configuration"]
    assert max(frozen["constant_baseline_seasons"]) < HOLDOUT_SEASON
    assert predictions["constant_probability"].iloc[0] == pytest.approx(
        frozen["constant_baseline_probability"]
    )


def test_normalized_kalshi_removes_the_overround(predictions: pd.DataFrame) -> None:
    total = predictions["kalshi_home_midpoint"] + predictions["kalshi_away_midpoint"]
    expected = predictions["kalshi_home_midpoint"] / total
    assert (predictions["kalshi_home_probability_normalized"] - expected).abs().max() < 1e-12
    # Raw values are preserved alongside.
    assert predictions["kalshi_home_midpoint"].notna().all()


def test_no_kalshi_column_is_a_model_feature(predictions: pd.DataFrame) -> None:
    assert not any("kalshi" in c for c in MODEL_FEATURES)
    for column in ("kalshi_home_midpoint", "kalshi_home_probability_normalized"):
        assert column in predictions.columns
        assert column not in MODEL_FEATURES


def test_no_target_or_score_column_is_a_model_feature() -> None:
    assert TARGET not in MODEL_FEATURES
    assert not any("score" in c for c in MODEL_FEATURES)


# --- report ----------------------------------------------------------------


def test_the_frozen_configuration_is_recorded(report: dict) -> None:
    frozen = report["frozen_configuration"]
    assert set(frozen["elo"]) >= {"k_factor", "home_advantage", "regression_factor", "history"}
    assert set(frozen["logistic"]) >= {"training_history", "C"}
    assert frozen["feature_allowlist"] == list(MODEL_FEATURES)
    assert max(frozen["logistic_training_seasons"]) < HOLDOUT_SEASON
    assert frozen["frozen_at_utc"]


def test_every_candidate_configuration_is_recorded(report: dict) -> None:
    assert len(report["development"]["elo_candidates"]) == 288
    assert len(report["development"]["logistic_candidates"]) == 40
    for candidate in report["development"]["elo_candidates"]:
        assert len(candidate["folds"]) == 4
        assert all(f["season"] < HOLDOUT_SEASON for f in candidate["folds"])


def test_all_five_predictors_are_scored(report: dict) -> None:
    metrics = report["holdout"]["metrics"]
    assert set(metrics) == {
        "constant", "elo", "logistic", "kalshi_raw_midpoint", "kalshi_normalized"
    }
    for summary in metrics.values():
        assert summary["n"] == 1230


def test_paired_comparisons_are_reproducible_and_signed(report: dict) -> None:
    comparisons = report["holdout"]["paired_comparisons"]
    assert set(comparisons) == {
        "logistic_vs_kalshi_normalized", "elo_vs_kalshi_normalized", "logistic_vs_elo"
    }
    for comparison in comparisons.values():
        for metric in ("brier", "log_loss"):
            result = comparison[metric]
            assert result["n_resamples"] >= 10_000
            assert result["n_games"] == 1230
            assert result["ci_low"] <= result["mean_loss_difference"] <= result["ci_high"]
            assert "model_minus_benchmark < 0" in result["sign_convention"]


def test_calibration_is_reported_for_every_predictor(report: dict) -> None:
    for name, calibration in report["holdout"]["calibration"].items():
        assert len(calibration["bins"]) == 10, name
        assert calibration["expected_calibration_error"] >= 0.0
        assert sum(b["count"] for b in calibration["bins"]) == 1230


def test_leakage_audits_all_pass(report: dict) -> None:
    audits = report["leakage_audits"]
    for key, value in audits.items():
        if isinstance(value, bool):
            assert value is True, f"leakage audit failed: {key}"
