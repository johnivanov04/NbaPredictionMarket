"""Logistic pipeline: allowlist, train-only preprocessing, weighting."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nba_prediction_market.features.feature_spec import (
    MODEL_FEATURES,
    TARGET,
    validate_feature_matrix,
)
from nba_prediction_market.models.logistic import (
    HISTORY_ALL,
    HISTORY_WEIGHTED,
    LogisticConfig,
    build_pipeline,
    feature_matrix,
    fit_logistic,
    recency_weights,
    training_seasons,
)

rng = np.random.default_rng(7)


def frame(n: int = 400, season: int = 2020, missing: bool = False) -> pd.DataFrame:
    data = {c: rng.normal(size=n) for c in MODEL_FEATURES}
    data["home_back_to_back"] = rng.integers(0, 2, n).astype(float)
    data["away_back_to_back"] = rng.integers(0, 2, n).astype(float)
    data["home_games_played"] = rng.integers(0, 82, n).astype(float)
    data["away_games_played"] = rng.integers(0, 82, n).astype(float)
    df = pd.DataFrame(data)
    if missing:
        df.loc[: n // 10, "last5_win_pct_diff"] = np.nan
    df["season"] = season
    df[TARGET] = (df["elo_diff"] + rng.normal(scale=0.5, size=n) > 0).astype(int)
    return df


# --- feature allowlist -----------------------------------------------------


def test_the_allowlist_is_exactly_the_documented_features() -> None:
    assert MODEL_FEATURES == (
        "elo_diff", "win_pct_diff", "last5_win_pct_diff", "last10_win_pct_diff",
        "last5_point_diff_difference", "last10_point_diff_difference",
        "rest_days_diff", "home_back_to_back", "away_back_to_back",
        "home_games_played", "away_games_played",
    )
    assert TARGET not in MODEL_FEATURES


def test_the_matrix_contains_only_allowlisted_columns() -> None:
    df = frame()
    df["kalshi_home_midpoint"] = 0.5
    df["home_score"] = 110
    matrix = feature_matrix(df)

    assert list(matrix.columns) == list(MODEL_FEATURES)
    assert TARGET not in matrix.columns
    assert not any("kalshi" in c for c in matrix.columns)
    assert not any("score" in c for c in matrix.columns)


@pytest.mark.parametrize(
    "column",
    ["home_win", "kalshi_home_midpoint", "home_score", "source_home_score",
     "score_corrected", "logistic_probability", "home_team", "home_franchise_id"],
)
def test_leaky_columns_are_rejected_by_the_validator(column: str) -> None:
    with pytest.raises(ValueError):
        validate_feature_matrix([*MODEL_FEATURES, column])


def test_a_missing_required_feature_is_an_error() -> None:
    df = frame().drop(columns=["elo_diff"])
    with pytest.raises(ValueError, match="missing required columns"):
        feature_matrix(df)


# --- training-window selection ---------------------------------------------


@pytest.mark.parametrize(
    ("history", "expected"),
    [
        (3, [2018, 2019, 2020]),
        (5, [2016, 2017, 2018, 2019, 2020]),
        (HISTORY_ALL, list(range(2006, 2021))),
    ],
)
def test_training_seasons_are_strictly_before_validation(history, expected) -> None:
    config = LogisticConfig(training_history=history, c_value=1.0)
    available = list(range(2006, 2025))
    seasons = training_seasons(config, 2021, available)

    assert seasons == expected
    assert all(s < 2021 for s in seasons)


def test_weighted_history_uses_every_prior_season() -> None:
    config = LogisticConfig(training_history=HISTORY_WEIGHTED, c_value=1.0, half_life=3)
    assert training_seasons(config, 2021, list(range(2006, 2025))) == list(range(2006, 2021))


def test_a_window_longer_than_history_is_clipped_not_padded() -> None:
    config = LogisticConfig(training_history=15, c_value=1.0)
    assert training_seasons(config, 2008, [2006, 2007]) == [2006, 2007]


# --- recency weights -------------------------------------------------------


def test_the_most_recent_season_has_weight_one() -> None:
    weights = recency_weights(np.array([2018, 2019, 2020]), 2020, 2.0)
    assert weights[-1] == pytest.approx(1.0)


def test_weights_halve_every_half_life() -> None:
    weights = recency_weights(np.array([2016, 2018, 2020]), 2020, 2.0)
    assert weights[1] == pytest.approx(0.5)
    assert weights[0] == pytest.approx(0.25)


@pytest.mark.parametrize("half_life", [2.0, 3.0, 5.0, 8.0])
def test_weights_decay_monotonically(half_life: float) -> None:
    seasons = np.arange(2010, 2021)
    weights = recency_weights(seasons, 2020, half_life)
    assert list(weights) == sorted(weights)
    assert (weights > 0).all()


def test_sample_weights_actually_reach_the_estimator() -> None:
    """A pipeline that silently drops weights would make the experiment a lie."""
    recent = frame(n=300, season=2020)
    old = frame(n=300, season=2010)
    old[TARGET] = 1 - old[TARGET]  # contradictory labels in the old season
    training = pd.concat([old, recent], ignore_index=True)

    unweighted = fit_logistic(training, LogisticConfig(training_history=HISTORY_ALL, c_value=1.0))
    weighted = fit_logistic(
        training, LogisticConfig(training_history=HISTORY_WEIGHTED, c_value=1.0, half_life=2.0)
    )
    a = unweighted.pipeline.named_steps["model"].coef_
    b = weighted.pipeline.named_steps["model"].coef_
    assert not np.allclose(a, b), "weights had no effect on the fitted coefficients"


# --- pipeline behaviour ----------------------------------------------------


def test_preprocessing_is_fitted_on_training_data_only() -> None:
    training = frame(n=300, season=2019, missing=True)
    pipeline = build_pipeline(1.0)
    pipeline.fit(feature_matrix(training), training[TARGET])

    imputer = pipeline.named_steps["impute"]
    expected = np.nanmedian(feature_matrix(training).to_numpy(), axis=0)
    assert np.allclose(imputer.statistics_, expected, equal_nan=True)


def test_imputation_does_not_see_the_validation_split() -> None:
    training = frame(n=300, season=2019)
    training.loc[:, "elo_diff"] = 1.0
    validation = frame(n=100, season=2020)
    validation.loc[:, "elo_diff"] = 999.0
    validation.loc[:, "last5_win_pct_diff"] = np.nan

    fitted = fit_logistic(training, LogisticConfig(training_history=3, c_value=1.0))
    imputer = fitted.pipeline.named_steps["impute"]
    index = list(MODEL_FEATURES).index("elo_diff")
    assert imputer.statistics_[index] == pytest.approx(1.0), "validation must not shift the median"


def test_scaler_is_fitted_on_training_only() -> None:
    training = frame(n=300, season=2019)
    fitted = fit_logistic(training, LogisticConfig(training_history=3, c_value=1.0))
    scaler = fitted.pipeline.named_steps["scale"]
    assert np.allclose(scaler.mean_, feature_matrix(training).mean().to_numpy())


def test_predictions_are_probabilities() -> None:
    training = frame(n=400, season=2019)
    validation = frame(n=200, season=2020)
    fitted = fit_logistic(training, LogisticConfig(training_history=3, c_value=1.0))
    prob = fitted.predict_proba(validation)

    assert prob.shape == (200,)
    assert (prob > 0.0).all() and (prob < 1.0).all()


def test_predictions_are_deterministic() -> None:
    training = frame(n=400, season=2019)
    validation = frame(n=200, season=2020)
    config = LogisticConfig(training_history=3, c_value=1.0)
    first = fit_logistic(training, config).predict_proba(validation)
    second = fit_logistic(training, config).predict_proba(validation)
    assert np.array_equal(first, second)


def test_missing_values_are_handled_inside_the_pipeline() -> None:
    training = frame(n=400, season=2019, missing=True)
    validation = frame(n=200, season=2020, missing=True)
    fitted = fit_logistic(training, LogisticConfig(training_history=3, c_value=1.0))
    prob = fitted.predict_proba(validation)
    assert np.isfinite(prob).all()


# --- configuration validation ---------------------------------------------


@pytest.mark.parametrize("c", [0, -1.0])
def test_non_positive_c_is_rejected(c) -> None:
    with pytest.raises(ValueError, match="C must be positive"):
        LogisticConfig(training_history=3, c_value=c)


def test_weighted_history_requires_a_half_life() -> None:
    with pytest.raises(ValueError, match="half_life"):
        LogisticConfig(training_history=HISTORY_WEIGHTED, c_value=1.0)


def test_empty_training_data_is_rejected() -> None:
    with pytest.raises(ValueError, match="no training rows"):
        fit_logistic(frame(n=0), LogisticConfig(training_history=3, c_value=1.0))
