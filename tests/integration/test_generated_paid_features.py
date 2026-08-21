"""Invariants of the generated Phase 3A3 artefacts. ``pytest -m dataset``."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from nba_prediction_market.models.paid_bundles import PAID_BUNDLES_BY_NAME, PAID_FAMILIES
from nba_prediction_market.models.selection import HOLDOUT_SEASON

pytestmark = pytest.mark.dataset

PROCESSED = Path("data/processed")
TEAM_GAMES = PROCESSED / "nba_team_game_paid_features_2006_26.parquet"
FEATURES = PROCESSED / "nba_model_features_3a3_2006_26.parquet"
PREDICTIONS = PROCESSED / "nba_predictions_3a3_2025_26.parquet"
KALSHI = PROCESSED / "nba_kalshi_pregame_t30_2025_26.parquet"
REPORT = Path("data/reports/model_paid_features_2025_26.json")


def _load(path: Path) -> pd.DataFrame:
    if not path.is_file():
        pytest.skip(f"{path} not generated; run build_paid_features first")
    return pd.read_parquet(path)


@pytest.fixture(scope="module")
def team_games() -> pd.DataFrame:
    return _load(TEAM_GAMES)


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


# --- team-game dataset -----------------------------------------------------


def test_two_team_observations_per_game(team_games: pd.DataFrame) -> None:
    assert len(team_games) == 24038 * 2
    assert not team_games.duplicated(subset=["nba_game_id", "team_id"]).any()
    assert set(team_games.groupby("nba_game_id").size().unique()) == {2}


def test_possessions_are_plausible(team_games: pd.DataFrame) -> None:
    possessions = team_games["estimated_possessions"].dropna()
    assert possessions.min() > 60.0
    assert possessions.quantile(0.50) == pytest.approx(96.5, abs=3.0)
    assert possessions.max() < 160.0


def test_overtime_games_estimate_more_possessions(team_games: pd.DataFrame) -> None:
    """The four corrected 4OT games ran 68 minutes."""
    overtime = team_games[team_games["nba_game_id"].isin({28012, 32587, 34714, 48851})]
    assert len(overtime) == 8
    assert overtime["estimated_possessions"].median() > 120.0


def test_unreconciled_box_scores_derive_nothing(team_games: pd.DataFrame) -> None:
    incomplete = team_games[~team_games["box_score_complete"]]
    assert len(incomplete) == 16
    assert incomplete["estimated_possessions"].isna().all()
    assert incomplete["offensive_efficiency"].isna().all()
    # The source totals are preserved.
    assert incomplete["pts"].notna().all()


@pytest.mark.parametrize(
    "column", ["efg_pct", "turnover_rate", "oreb_rate", "true_shooting_pct"]
)
def test_rate_statistics_are_bounded(team_games: pd.DataFrame, column: str) -> None:
    values = team_games[column].dropna()
    assert values.between(0.0, 1.0).all()


# --- feature dataset -------------------------------------------------------


def test_features_cover_every_eligible_game(features: pd.DataFrame) -> None:
    assert len(features) == 24038
    assert features["nba_game_id"].is_unique


def test_paid_features_are_present(features: pd.DataFrame) -> None:
    for column in ("season_net_efficiency_diff", "rotation_disruption_score_diff",
                   "expected_rotation_strength_diff", "top5_recent_minutes_share_diff"):
        assert column in features.columns


def test_rotation_features_are_null_for_a_teams_first_game(
    features: pd.DataFrame,
) -> None:
    openers = features[features["home_games_played"] == 0]
    assert openers["home_recent_rotation_player_count"].isna().all()
    assert openers["home_expected_rotation_strength"].isna().all()


def test_rotation_player_counts_are_plausible(features: pd.DataFrame) -> None:
    counts = features["home_recent_rotation_player_count"].dropna()
    assert counts.min() >= 5
    assert counts.max() <= 25


# --- holdout ---------------------------------------------------------------


def test_exactly_1230_holdout_predictions(predictions: pd.DataFrame) -> None:
    assert len(predictions) == 1230
    assert predictions["nba_game_id"].is_unique
    assert set(predictions["season"].unique()) == {HOLDOUT_SEASON}


def test_prediction_ids_match_phase_2_exactly(predictions: pd.DataFrame) -> None:
    kalshi = _load(KALSHI)
    assert set(predictions["nba_game_id"]) == set(kalshi["nba_game_id"])


@pytest.mark.parametrize(
    "column",
    ["phase_3a2_logistic_probability", "phase_3a3_logistic_probability",
     "mov_elo_probability", "kalshi_home_probability_normalized"],
)
def test_probabilities_are_valid(predictions: pd.DataFrame, column: str) -> None:
    assert predictions[column].notna().all()
    assert predictions[column].between(0.0, 1.0).all()


def test_earlier_phase_artefacts_are_untouched() -> None:
    """Phase 3A1 and 3A2 results must remain reproducible."""
    for path, key, expected in (
        (Path("data/reports/model_baselines_2025_26.json"),
         ("holdout", "metrics", "logistic", "brier_score"), 0.20575),
        (Path("data/reports/model_team_strength_2025_26.json"),
         ("holdout", "metrics", "phase_3a2_logistic", "brier_score"), 0.20451),
    ):
        if not path.is_file():
            continue
        value = json.loads(path.read_text())
        for part in key:
            value = value[part]
        assert value == pytest.approx(expected, abs=1e-5)


# --- report and leakage ----------------------------------------------------


def test_the_frozen_bundle_contains_no_leaky_column(report: dict) -> None:
    frozen = report["frozen_configuration"]
    for token in ("kalshi", "injur", "lineup", "starter", "home_win"):
        assert not any(token in c for c in frozen["bundle_features"])


def test_every_paid_feature_in_the_bundle_is_a_lagged_difference(report: dict) -> None:
    frozen = report["frozen_configuration"]
    control = set(PAID_BUNDLES_BY_NAME["A"].features)
    for column in frozen["bundle_features"]:
        assert column.endswith("_diff") or column in control


def test_leakage_audits_all_pass(report: dict) -> None:
    for key, value in report["leakage_audits"].items():
        if isinstance(value, bool):
            assert value is True, f"leakage audit failed: {key}"


def test_every_family_was_ablated(report: dict) -> None:
    tested = {a["config"]["name"] for a in report["development"]["bundle_ablation"]}
    assert {"A", "B", "C", "D", "E"} <= tested
    assert set(report["development"]["families_tested"]) == set(PAID_FAMILIES)


def test_training_excludes_the_holdout(report: dict) -> None:
    frozen = report["frozen_configuration"]
    assert max(frozen["logistic_training_seasons"]) < HOLDOUT_SEASON
    for candidate in report["development"]["bundle_ablation"]:
        assert all(f["season"] < HOLDOUT_SEASON for f in candidate["folds"])


def test_paired_comparisons_are_present(report: dict) -> None:
    comparisons = report["holdout"]["paired_comparisons"]
    assert set(comparisons) == {
        "phase_3a3_vs_phase_3a2", "phase_3a3_vs_mov_elo", "phase_3a3_vs_kalshi"
    }
    for comparison in comparisons.values():
        for metric in ("brier", "log_loss"):
            assert comparison[metric]["n_resamples"] >= 10_000
            assert comparison[metric]["n_games"] == 1230


def test_possession_validation_is_recorded(report: dict) -> None:
    validation = report["possession_validation"]
    assert validation["non_positive"] == 0
    assert "ESTIMATE" in validation["formula"]
    assert validation["corrected_4ot_median_possessions"] > 120.0
