"""Feature bundles, blending, and Phase 3A2 holdout isolation."""

from __future__ import annotations

from itertools import pairwise

import numpy as np
import pytest

from nba_prediction_market.models.bundles import (
    BUNDLE_A_BASELINE,
    BUNDLES,
    BUNDLES_BY_NAME,
    Bundle,
    validate_bundle,
)
from nba_prediction_market.models.selection import HOLDOUT_SEASON, HoldoutLeakageError
from nba_prediction_market.models.team_strength_selection import (
    BLEND_WEIGHT_GRID,
    EWMA_HALF_LIFE_GRID,
    MOV_K_GRID,
    REFINED_HCA_GRID,
    blend,
    evaluate_blend,
    mov_elo_grid,
    refined_elo_grid,
)
from nba_prediction_market.pipelines.build_team_strength import PHASE_3A1_LOGISTIC

# --- bundles ---------------------------------------------------------------


def test_bundle_a_reproduces_the_phase_3a1_feature_set() -> None:
    from nba_prediction_market.features.feature_spec import MODEL_FEATURES

    assert BUNDLES_BY_NAME["A"].features == MODEL_FEATURES == BUNDLE_A_BASELINE


def test_bundles_are_strictly_nested() -> None:
    """Each step adds a family without removing one, so the ablation is readable."""
    for earlier, later in pairwise(BUNDLES):
        assert set(earlier.features) < set(later.features)
        assert later.features[: len(earlier.features)] == earlier.features


def test_every_bundle_is_free_of_leaky_columns() -> None:
    for bundle in BUNDLES:
        validate_bundle(bundle)


@pytest.mark.parametrize(
    "feature",
    ["kalshi_home_midpoint", "home_win", "home_score", "home_team", "elo_probability"],
)
def test_a_leaky_feature_in_a_bundle_is_rejected(feature: str) -> None:
    with pytest.raises(ValueError, match="leaky-looking"):
        validate_bundle(Bundle("X", "test", (*BUNDLE_A_BASELINE, feature)))


def test_a_bundle_cannot_repeat_a_feature() -> None:
    with pytest.raises(ValueError, match="repeats a feature"):
        Bundle("X", "test", ("elo_diff", "elo_diff"))


def test_bundle_names_are_unique_and_ordered() -> None:
    assert [b.name for b in BUNDLES] == ["A", "B", "C", "D", "E", "F"]


# --- grids -----------------------------------------------------------------


def test_the_refined_hca_grid_extends_below_the_phase_3a1_boundary() -> None:
    """Phase 3A1 selected 40 at its grid edge, so the search is widened downward."""
    assert min(REFINED_HCA_GRID) == 0.0
    assert 40.0 in REFINED_HCA_GRID
    assert len(refined_elo_grid()) == 3 * len(REFINED_HCA_GRID) * 3


def test_the_mov_grid_stays_small() -> None:
    grid = mov_elo_grid([40.0])
    assert len(grid) == 3 * len(MOV_K_GRID) * 1 * 2
    assert {c.multiplier for c in grid} == {"log", "fivethirtyeight", "sqrt"}


def test_the_ewma_grid_is_the_documented_one() -> None:
    assert EWMA_HALF_LIFE_GRID == (3.0, 5.0, 10.0)


# --- blending --------------------------------------------------------------


def test_blend_weight_is_the_share_given_to_the_partner() -> None:
    primary = np.array([0.2, 0.8])
    secondary = np.array([0.6, 0.4])
    assert np.allclose(blend(primary, secondary, 0.0), primary)
    assert np.allclose(blend(primary, secondary, 1.0), secondary)
    assert np.allclose(blend(primary, secondary, 0.5), [0.4, 0.6])


@pytest.mark.parametrize("weight", [-0.1, 1.1])
def test_an_out_of_range_blend_weight_is_rejected(weight: float) -> None:
    with pytest.raises(ValueError, match="blend weight"):
        blend(np.array([0.5]), np.array([0.5]), weight)


def test_the_blend_grid_includes_both_pure_models() -> None:
    assert BLEND_WEIGHT_GRID[0] == 0.0
    assert BLEND_WEIGHT_GRID[-1] == 1.0


def test_blend_evaluation_scores_every_development_fold() -> None:
    rng = np.random.default_rng(0)
    y = {s: rng.integers(0, 2, 50) for s in (2021, 2022)}
    primary = {s: rng.uniform(0.3, 0.7, 50) for s in (2021, 2022)}
    secondary = {s: rng.uniform(0.3, 0.7, 50) for s in (2021, 2022)}
    result = evaluate_blend(y, primary, secondary, 0.5, "partner")

    assert len(result.folds) == 2
    assert result.label == "partner_w=0.5"
    assert result.config == {"partner": "partner", "weight": 0.5}


def test_blend_evaluation_refuses_holdout_seasons() -> None:
    y = {HOLDOUT_SEASON: np.array([1, 0])}
    p = {HOLDOUT_SEASON: np.array([0.5, 0.5])}
    with pytest.raises(HoldoutLeakageError):
        evaluate_blend(y, p, p, 0.5, "partner")


# --- the Phase 3A1 control -------------------------------------------------


def test_the_phase_3a1_control_is_reproduced_exactly() -> None:
    """The comparison is only meaningful if the control is the real thing."""
    assert PHASE_3A1_LOGISTIC.training_history == 5
    assert PHASE_3A1_LOGISTIC.c_value == 10.0
    assert PHASE_3A1_LOGISTIC.half_life is None


# --- end-to-end selection on synthetic seasons -----------------------------


@pytest.fixture(scope="module")
def synthetic_features():
    """A small synthetic league run through the whole Phase 3A2 feature stack."""
    import pandas as pd

    from nba_prediction_market.features.adjusted_margin import run_adjusted_margin
    from nba_prediction_market.features.elo import EloConfig, run_elo
    from nba_prediction_market.features.feature_engine import (
        FEATURE_COLUMNS,
        build_features,
    )
    from nba_prediction_market.features.mov_elo import MovEloConfig, run_mov_elo
    from nba_prediction_market.features.team_strength import (
        TEAM_STRENGTH_COLUMNS,
        build_team_strength_features,
    )

    rng = np.random.default_rng(5)
    strength = {t: rng.normal(0, 100) for t in range(1, 9)}
    rows, gid = [], 0
    for season in range(2018, 2026):
        day = 0.0
        for _ in range(3):
            for home in range(1, 9):
                for away in range(1, 9):
                    if home == away:
                        continue
                    gid += 1
                    day += 0.3
                    edge = (strength[home] - strength[away]) / 400 + 0.25
                    home_win = bool(rng.random() < 1 / (1 + 10 ** (-edge)))
                    margin = int(rng.integers(1, 20))
                    rows.append(
                        {
                            "nba_game_id": gid, "season": season,
                            "game_datetime_utc": pd.Timestamp(f"{season}-10-20", tz="UTC")
                            + pd.Timedelta(days=day),
                            "home_franchise_id": home, "away_franchise_id": away,
                            "home_team": f"T{home:02d}", "away_team": f"T{away:02d}",
                            "home_score": 100 + (margin if home_win else 0),
                            "away_score": 100 + (0 if home_win else margin),
                            "home_win": home_win,
                        }
                    )
    records = rows
    frame = pd.DataFrame(build_features(records), columns=list(FEATURE_COLUMNS))
    elo = run_elo(records, EloConfig(k_factor=20, home_advantage=40, regression_factor=0.5))
    frame = frame.merge(
        pd.DataFrame([{"nba_game_id": p.nba_game_id, "elo_diff": p.elo_diff,
                       "elo_probability": p.home_win_probability} for p in elo]),
        on="nba_game_id", validate="one_to_one")
    mov = run_mov_elo(
        records,
        MovEloConfig(k_factor=20, home_advantage=40, regression_factor=0.5, multiplier="sqrt"),
    )
    frame = frame.merge(
        pd.DataFrame([{"nba_game_id": p.nba_game_id, "mov_elo_diff": p.mov_elo_diff,
                       "mov_elo_probability": p.home_win_probability} for p in mov]),
        on="nba_game_id", validate="one_to_one")
    adjusted = run_adjusted_margin(records)
    frame = frame.merge(
        pd.DataFrame([{"nba_game_id": p.nba_game_id,
                       "adjusted_margin_diff": p.adjusted_margin_diff} for p in adjusted]),
        on="nba_game_id", validate="one_to_one")
    strength_rows = build_team_strength_features(
        records, {p.nba_game_id: (1500.0, 1500.0) for p in elo}
    )
    return frame.merge(
        pd.DataFrame(strength_rows, columns=list(TEAM_STRENGTH_COLUMNS)),
        on="nba_game_id", validate="one_to_one")


def test_refined_elo_selection_runs_on_development_only(synthetic_features) -> None:
    import pandas as pd

    from nba_prediction_market.models.team_strength_selection import select_refined_elo

    games = synthetic_features[synthetic_features["season"] < HOLDOUT_SEASON]
    games = games.assign(
        home_score=110, away_score=100
    )  # scores unused by binary Elo
    winner, ranked = select_refined_elo(games, validation_seasons=(2023, 2024))
    assert len(ranked) == 63
    assert winner.home_advantage in REFINED_HCA_GRID
    assert isinstance(pd.DataFrame(), pd.DataFrame)


def test_mov_selection_runs_on_development_only(synthetic_features) -> None:
    from nba_prediction_market.models.team_strength_selection import select_mov_elo

    games = synthetic_features[synthetic_features["season"] < HOLDOUT_SEASON]
    games = games.assign(home_score=110, away_score=100)
    winner, ranked = select_mov_elo(
        games, hca_grid=[40.0], validation_seasons=(2023, 2024)
    )
    assert len(ranked) == 18
    assert winner.multiplier in {"log", "fivethirtyeight", "sqrt"}


def test_bundle_evaluation_refuses_the_holdout(synthetic_features) -> None:
    from nba_prediction_market.models.team_strength_selection import (
        BUNDLE_PROBE_CONFIG,
        evaluate_bundle,
    )

    with pytest.raises(HoldoutLeakageError):
        evaluate_bundle(
            synthetic_features, BUNDLES_BY_NAME["A"], BUNDLE_PROBE_CONFIG, (2023,)
        )


def test_every_bundle_can_be_evaluated(synthetic_features) -> None:
    from nba_prediction_market.models.team_strength_selection import (
        BUNDLE_PROBE_CONFIG,
        evaluate_bundle,
    )

    development = synthetic_features[synthetic_features["season"] < HOLDOUT_SEASON]
    for bundle in BUNDLES:
        result = evaluate_bundle(development, bundle, BUNDLE_PROBE_CONFIG, (2023, 2024))
        assert len(result.folds) == 2
        assert 0.0 < result.mean_brier < 0.5
        assert result.config["n_features"] == len(bundle.features)
