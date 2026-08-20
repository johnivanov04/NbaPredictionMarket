"""Selection/holdout separation and the guards that keep them apart."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nba_prediction_market.features.feature_spec import MODEL_FEATURES, TARGET
from nba_prediction_market.models.selection import (
    DEVELOPMENT_VALIDATION_SEASONS,
    ELO_HISTORY_GRID,
    ELO_K_GRID,
    HOLDOUT_SEASON,
    LOGISTIC_C_GRID,
    LOGISTIC_HALF_LIVES,
    TIE_TOLERANCE,
    CandidateResult,
    FoldResult,
    HoldoutLeakageError,
    assert_no_holdout,
    elo_grid,
    logistic_grid,
    rank_candidates,
    tied_with_best,
)
from nba_prediction_market.pipelines.build_baselines import (
    PREDICTION_COLUMNS,
    PREDICTORS,
    diagnostic_segments,
    run_leakage_audits,
)

# --- holdout isolation -----------------------------------------------------


def test_the_holdout_season_is_2025_and_validation_is_the_four_before_it() -> None:
    assert HOLDOUT_SEASON == 2025
    assert DEVELOPMENT_VALIDATION_SEASONS == (2021, 2022, 2023, 2024)
    assert all(s < HOLDOUT_SEASON for s in DEVELOPMENT_VALIDATION_SEASONS)


def test_development_stages_refuse_holdout_seasons() -> None:
    with pytest.raises(HoldoutLeakageError, match="holdout season"):
        assert_no_holdout([2023, 2025], where="test")


def test_seasons_after_the_holdout_are_also_refused() -> None:
    with pytest.raises(HoldoutLeakageError):
        assert_no_holdout([2026], where="test")


def test_pre_holdout_seasons_pass() -> None:
    assert_no_holdout([2006, 2015, 2024], where="test")  # must not raise


# --- grids -----------------------------------------------------------------


def test_the_elo_grid_is_the_documented_size() -> None:
    grid = elo_grid()
    assert len(grid) == 288
    assert len(grid) == len(ELO_K_GRID) * 4 * 3 * len(ELO_HISTORY_GRID)
    assert {c.history for c in grid} == set(ELO_HISTORY_GRID)


def test_the_logistic_grid_covers_windows_and_recency_weighting() -> None:
    grid = logistic_grid()
    assert len(grid) == 40
    assert len(grid) == (6 * len(LOGISTIC_C_GRID)) + (len(LOGISTIC_HALF_LIVES) * 4)
    weighted = [c for c in grid if c.half_life is not None]
    assert {c.half_life for c in weighted} == set(LOGISTIC_HALF_LIVES)
    assert {c.c_value for c in grid} == set(LOGISTIC_C_GRID)


# --- ranking and ties ------------------------------------------------------


def candidate(label: str, brier: float, logloss: float, **config) -> CandidateResult:
    return CandidateResult(
        label=label,
        config=config,
        folds=[FoldResult(season=s, brier_score=brier, log_loss=logloss, n=1230)
               for s in DEVELOPMENT_VALIDATION_SEASONS],
    )


def test_ranking_is_brier_first_then_log_loss() -> None:
    a = candidate("a", 0.21, 0.63)
    b = candidate("b", 0.21, 0.62)
    c = candidate("c", 0.20, 0.99)
    assert [x.label for x in rank_candidates([a, b, c])] == ["c", "b", "a"]


def test_ties_are_grouped_within_the_documented_tolerance() -> None:
    best = candidate("best", 0.2100, 0.62)
    close = candidate("close", 0.2100 + TIE_TOLERANCE / 2, 0.62)
    far = candidate("far", 0.2100 + TIE_TOLERANCE * 10, 0.62)
    tied = tied_with_best(rank_candidates([best, close, far]))
    assert {c.label for c in tied} == {"best", "close"}


def test_the_tolerance_is_far_below_fold_to_fold_variation() -> None:
    """A difference this small carries no information about which model is better."""
    assert TIE_TOLERANCE == 1e-4
    assert TIE_TOLERANCE < 0.001


# --- leakage audits --------------------------------------------------------


def features_frame(n: int = 50) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    df = pd.DataFrame({c: rng.normal(size=n) for c in MODEL_FEATURES})
    df["home_games_played"] = list(range(n))
    df["away_games_played"] = list(range(n))
    df["home_rest_days"] = [None, *([2.0] * (n - 1))]
    df["home_back_to_back"] = [False] * n
    df["game_datetime_utc"] = pd.date_range("2021-10-19", periods=n, freq="D", tz="UTC")
    df[TARGET] = rng.integers(0, 2, n)
    return df


def test_the_audit_confirms_no_leaky_column_is_allowlisted() -> None:
    audits = run_leakage_audits(features_frame())
    assert audits["target_excluded_from_matrix"] is True
    assert audits["no_kalshi_columns_in_matrix"] is True
    assert audits["no_score_columns_in_matrix"] is True
    assert audits["no_provenance_columns_in_matrix"] is True
    assert audits["features_sorted_by_actual_tipoff"] is True


def test_the_audit_checks_first_game_rest_semantics() -> None:
    audits = run_leakage_audits(features_frame())
    assert audits["first_game_rest_is_null"] is True
    assert audits["no_first_game_back_to_back"] is True


def test_prediction_columns_carry_no_model_input_beyond_the_allowlist() -> None:
    inputs = [c for c in PREDICTION_COLUMNS if c in MODEL_FEATURES]
    assert set(inputs) == set(MODEL_FEATURES)
    assert "kalshi_home_midpoint" in PREDICTION_COLUMNS, "benchmark is stored, not modelled"
    assert "kalshi_home_midpoint" not in MODEL_FEATURES


def test_every_predictor_has_a_stored_column() -> None:
    assert set(PREDICTORS) == {
        "constant", "elo", "logistic", "kalshi_raw_midpoint", "kalshi_normalized"
    }
    for column in PREDICTORS.values():
        assert column in PREDICTION_COLUMNS


# --- diagnostic segments ---------------------------------------------------


def predictions_frame(n: int = 60) -> pd.DataFrame:
    rng = np.random.default_rng(1)
    df = pd.DataFrame(
        {
            "home_games_played": rng.integers(0, 40, n),
            "away_games_played": rng.integers(0, 40, n),
            "home_back_to_back": rng.integers(0, 2, n).astype(bool),
            "away_back_to_back": rng.integers(0, 2, n).astype(bool),
            "kalshi_home_probability_normalized": rng.uniform(0.1, 0.9, n),
            TARGET: rng.integers(0, 2, n),
        }
    )
    for column in PREDICTORS.values():
        df[column] = rng.uniform(0.2, 0.8, n)
    return df


def test_segments_partition_the_games_they_claim_to() -> None:
    df = predictions_frame()
    segments = diagnostic_segments(df, df[TARGET].to_numpy())
    total = segments["kalshi_home_favourite"]["n"] + segments["kalshi_home_underdog"]["n"]
    assert total == len(df)
    both = segments["back_to_back_involved"]["n"] + segments["no_back_to_back"]["n"]
    assert both == len(df)


def test_every_segment_scores_every_model() -> None:
    df = predictions_frame()
    segments = diagnostic_segments(df, df[TARGET].to_numpy())
    for segment in segments.values():
        assert set(segment["models"]) == set(PREDICTORS)
        assert segment["n"] > 0


# --- end-to-end selection and holdout on synthetic seasons -----------------


def synthetic_games(seasons=range(2016, 2026), teams=8, per_pair=2) -> pd.DataFrame:
    """A small league where home teams win 60% of the time."""
    rng = np.random.default_rng(11)
    rows, gid = [], 0
    strength = {t: rng.normal(0, 100) for t in range(1, teams + 1)}
    for season in seasons:
        day = 0.0
        for _ in range(per_pair):
            for home in range(1, teams + 1):
                for away in range(1, teams + 1):
                    if home == away:
                        continue
                    gid += 1
                    day += 0.35
                    edge = (strength[home] - strength[away]) / 400 + 0.25
                    home_win = bool(rng.random() < 1 / (1 + 10 ** (-edge)))
                    margin = rng.integers(1, 20)
                    rows.append(
                        {
                            "nba_game_id": gid,
                            "season": season,
                            "game_datetime_utc": pd.Timestamp(
                                f"{season}-10-20", tz="UTC"
                            ) + pd.Timedelta(days=day),
                            "home_franchise_id": home,
                            "away_franchise_id": away,
                            "home_team": f"T{home:02d}",
                            "away_team": f"T{away:02d}",
                            "home_score": 100 + (margin if home_win else 0),
                            "away_score": 100 + (0 if home_win else margin),
                            "home_win": home_win,
                        }
                    )
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def synthetic_features() -> pd.DataFrame:
    from nba_prediction_market.features.elo import EloConfig
    from nba_prediction_market.features.feature_engine import FEATURE_COLUMNS, build_features
    from nba_prediction_market.pipelines.build_baselines import attach_elo

    games = synthetic_games()
    features = pd.DataFrame(build_features(games.to_dict("records")), columns=list(FEATURE_COLUMNS))
    return attach_elo(
        features, EloConfig(k_factor=20, home_advantage=40, regression_factor=0.5)
    )


def test_elo_selection_runs_and_never_reads_the_holdout(synthetic_features) -> None:
    from nba_prediction_market.models.selection import select_elo

    development = synthetic_features[synthetic_features["season"] < HOLDOUT_SEASON]
    winner, ranked = select_elo(development, validation_seasons=(2023, 2024))

    assert len(ranked) == 288
    assert winner.k_factor in ELO_K_GRID
    for candidate in ranked:
        assert all(f["season"] < HOLDOUT_SEASON for f in candidate.to_dict()["folds"])


def test_elo_selection_refuses_a_frame_containing_the_holdout(synthetic_features) -> None:
    from nba_prediction_market.models.selection import select_elo

    with pytest.raises(HoldoutLeakageError):
        select_elo(synthetic_features, validation_seasons=(2023, 2024))


def test_logistic_selection_runs_and_trains_only_on_prior_seasons(
    synthetic_features,
) -> None:
    from nba_prediction_market.models.selection import select_logistic

    development = synthetic_features[synthetic_features["season"] < HOLDOUT_SEASON]
    winner, ranked = select_logistic(development, validation_seasons=(2023, 2024))

    assert len(ranked) == 40
    assert winner.c_value in LOGISTIC_C_GRID
    assert all(len(c.folds) == 2 for c in ranked)


def test_logistic_selection_refuses_the_holdout(synthetic_features) -> None:
    from nba_prediction_market.models.selection import select_logistic

    with pytest.raises(HoldoutLeakageError):
        select_logistic(synthetic_features, validation_seasons=(2023,))


def test_holdout_stage_uses_the_frozen_configuration(synthetic_features) -> None:
    """Requirement: coefficients are frozen before the holdout is scored."""
    from nba_prediction_market.pipelines.build_baselines import FrozenConfig, stage_holdout

    holdout_ids = synthetic_features[synthetic_features["season"] == HOLDOUT_SEASON][
        "nba_game_id"
    ]
    rng = np.random.default_rng(2)
    kalshi = pd.DataFrame(
        {
            "nba_game_id": holdout_ids.to_numpy(),
            "kalshi_home_midpoint": rng.uniform(0.3, 0.7, len(holdout_ids)),
        }
    )
    kalshi["kalshi_away_midpoint"] = 1.01 - kalshi["kalshi_home_midpoint"]
    kalshi["kalshi_home_probability_normalized"] = kalshi["kalshi_home_midpoint"] / (
        kalshi["kalshi_home_midpoint"] + kalshi["kalshi_away_midpoint"]
    )

    frozen = FrozenConfig(
        elo={"k_factor": 20.0, "home_advantage": 40.0, "regression_factor": 0.5,
             "history": "all_available", "initial_rating": 1500.0},
        logistic={"training_history": 5, "half_life_seasons": None, "C": 1.0},
        feature_allowlist=list(MODEL_FEATURES),
        constant_baseline_probability=0.6,
        constant_baseline_seasons=list(range(2016, 2025)),
        logistic_training_seasons=[2020, 2021, 2022, 2023, 2024],
        development_validation_seasons=[2023, 2024],
    )
    predictions, evaluation = stage_holdout(synthetic_features, frozen, kalshi)

    assert len(predictions) == len(holdout_ids)
    assert set(predictions["nba_game_id"]) == set(holdout_ids)
    assert (predictions["constant_probability"] == 0.6).all()
    assert predictions["logistic_probability"].between(0, 1).all()
    assert set(evaluation["metrics"]) == set(PREDICTORS)
    assert evaluation["logistic_training"]["seasons"] == [2020, 2021, 2022, 2023, 2024]
    assert max(evaluation["logistic_training"]["seasons"]) < HOLDOUT_SEASON


def test_holdout_requires_exact_id_equality_with_the_market_dataset(
    synthetic_features,
) -> None:
    from nba_prediction_market.pipelines.build_baselines import FrozenConfig, stage_holdout

    holdout_ids = synthetic_features[synthetic_features["season"] == HOLDOUT_SEASON][
        "nba_game_id"
    ].to_numpy()
    kalshi = pd.DataFrame(
        {
            "nba_game_id": [*holdout_ids, 999_999],  # one game the model cannot predict
            "kalshi_home_midpoint": 0.5,
            "kalshi_away_midpoint": 0.5,
            "kalshi_home_probability_normalized": 0.5,
        }
    )
    frozen = FrozenConfig(
        elo={"k_factor": 20.0, "home_advantage": 40.0, "regression_factor": 0.5,
             "history": "all_available", "initial_rating": 1500.0},
        logistic={"training_history": 5, "half_life_seasons": None, "C": 1.0},
        feature_allowlist=list(MODEL_FEATURES),
        constant_baseline_probability=0.6,
        constant_baseline_seasons=list(range(2016, 2025)),
        logistic_training_seasons=[2020, 2021, 2022, 2023, 2024],
        development_validation_seasons=[2023, 2024],
    )
    with pytest.raises(ValueError, match="exactly the same games"):
        stage_holdout(synthetic_features, frozen, kalshi)
