"""Invariants of the generated Phase 3A2 artefacts. Run with ``pytest -m dataset``."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from nba_prediction_market.models.bundles import BUNDLES_BY_NAME
from nba_prediction_market.models.selection import HOLDOUT_SEASON

pytestmark = pytest.mark.dataset

PROCESSED = Path("data/processed")
FEATURES = PROCESSED / "nba_team_strength_features_2006_26.parquet"
PREDICTIONS = PROCESSED / "nba_predictions_3a2_2025_26.parquet"
PHASE_3A1_PREDICTIONS = PROCESSED / "nba_predictions_2025_26.parquet"
REPORT = Path("data/reports/model_team_strength_2025_26.json")
PHASE_3A1_REPORT = Path("data/reports/model_baselines_2025_26.json")


def _load(path: Path) -> pd.DataFrame:
    if not path.is_file():
        pytest.skip(f"{path} not generated; run build_team_strength first")
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


def test_features_cover_every_eligible_game(features: pd.DataFrame) -> None:
    assert len(features) == 24038
    assert features["nba_game_id"].is_unique
    for column in ("elo_diff", "mov_elo_diff", "adjusted_margin_diff", "season_sos_diff"):
        assert column in features.columns


def test_new_ratings_are_populated(features: pd.DataFrame) -> None:
    assert features["mov_elo_diff"].notna().all()
    # Adjusted margin is deliberately missing while a team is too sparse.
    assert features["adjusted_margin_diff"].isna().any()
    assert features["adjusted_margin_diff"].notna().mean() > 0.9


def test_exactly_1230_holdout_predictions(predictions: pd.DataFrame) -> None:
    assert len(predictions) == 1230
    assert set(predictions["season"].unique()) == {HOLDOUT_SEASON}


def test_the_same_games_as_phase_3a1(predictions: pd.DataFrame) -> None:
    """Both phases must be judged on identical games."""
    phase_3a1 = _load(PHASE_3A1_PREDICTIONS)
    assert set(predictions["nba_game_id"]) == set(phase_3a1["nba_game_id"])


def test_phase_3a1_artefacts_are_unchanged() -> None:
    """Phase 3A2 must not mutate the Phase 3A1 result."""
    if not PHASE_3A1_REPORT.is_file():
        pytest.skip("Phase 3A1 report not generated")
    report = json.loads(PHASE_3A1_REPORT.read_text())
    assert report["frozen_configuration"]["logistic"] == {
        "training_history": 5, "half_life_seasons": None, "C": 10.0
    }
    assert report["holdout"]["metrics"]["logistic"]["brier_score"] == pytest.approx(
        0.20575, abs=1e-5
    )


def test_the_phase_3a1_control_matches_its_own_artefact(
    predictions: pd.DataFrame,
) -> None:
    phase_3a1 = _load(PHASE_3A1_PREDICTIONS)[["nba_game_id", "logistic_probability"]]
    merged = predictions.merge(phase_3a1, on="nba_game_id", validate="one_to_one")
    assert (
        merged["phase_3a1_logistic_probability"] - merged["logistic_probability"]
    ).abs().max() < 1e-12


@pytest.mark.parametrize(
    "column",
    ["phase_3a1_logistic_probability", "phase_3a2_logistic_probability",
     "elo_probability", "mov_elo_probability", "kalshi_home_probability_normalized"],
)
def test_probabilities_are_valid(predictions: pd.DataFrame, column: str) -> None:
    assert predictions[column].notna().all()
    assert predictions[column].between(0.0, 1.0).all()


def test_no_kalshi_column_is_in_the_frozen_bundle(report: dict) -> None:
    frozen = report["frozen_configuration"]
    assert not any("kalshi" in c for c in frozen["bundle_features"])
    assert "home_win" not in frozen["bundle_features"]


def test_the_frozen_bundle_is_a_declared_one(report: dict) -> None:
    frozen = report["frozen_configuration"]
    bundle = BUNDLES_BY_NAME[frozen["bundle"]]
    assert list(bundle.features) == frozen["bundle_features"]


def test_training_seasons_exclude_the_holdout(report: dict) -> None:
    frozen = report["frozen_configuration"]
    assert max(frozen["logistic_training_seasons"]) < HOLDOUT_SEASON


def test_every_development_experiment_is_recorded(report: dict) -> None:
    development = report["development"]
    assert len(development["refined_elo_candidates"]) == 63
    assert len(development["mov_elo_candidates"]) > 0
    # 6 bundles x 3 EWMA half-lives
    assert len(development["bundle_ablation"]) == 18
    assert len(development["blend_candidates"]) == 15
    for candidate in development["refined_elo_candidates"]:
        assert all(f["season"] < HOLDOUT_SEASON for f in candidate["folds"])


def test_paired_comparisons_are_present_and_signed(report: dict) -> None:
    comparisons = report["holdout"]["paired_comparisons"]
    assert set(comparisons) == {
        "phase_3a2_vs_phase_3a1", "phase_3a2_vs_kalshi", "phase_3a1_vs_kalshi"
    }
    for comparison in comparisons.values():
        for metric in ("brier", "log_loss"):
            result = comparison[metric]
            assert result["n_resamples"] >= 10_000
            assert result["n_games"] == 1230
            assert "model_minus_benchmark < 0" in result["sign_convention"]


def test_leakage_audits_all_pass(report: dict) -> None:
    for key, value in report["leakage_audits"].items():
        if isinstance(value, bool):
            assert value is True, f"leakage audit failed: {key}"
