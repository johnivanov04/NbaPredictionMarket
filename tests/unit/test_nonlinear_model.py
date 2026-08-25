"""Phase 3A4: feature-set discipline, calibration safety, selection rules."""

from __future__ import annotations

import numpy as np
import pandas as pd

from nba_prediction_market.models.nonlinear import (
    ALL_CONFIGS,
    CANDIDATE_RANGES,
    HISTGB_GRID,
    RANDOM_SEED,
    XGBOOST_GRID,
    build_estimator,
)
from nba_prediction_market.models.nonlinear_bundles import (
    ALL_ALLOWED_FEATURES,
    CORE,
    EXTENDED,
    EXTENDED_ADDITIONS,
    FORBIDDEN_TOKENS,
)
from nba_prediction_market.models.probability_calibration import (
    ISOTONIC,
    MIN_CALIBRATION_ROWS,
    MIN_ISOTONIC_ROWS,
    SIGMOID,
    chronological_oof_predictions,
    fit_calibrator,
)


class TestFeatureSets:
    def test_core_is_exactly_the_frozen_3a3c_allowlist(self):
        from nba_prediction_market.models.availability_bundles import (
            PHASE_3A3_FEATURES,
            ROLE_MINUTE_FAMILY,
        )

        assert CORE.features == PHASE_3A3_FEATURES + ROLE_MINUTE_FAMILY
        assert len(CORE.features) == 19

    def test_extended_strictly_contains_core(self):
        assert set(CORE.features) < set(EXTENDED.features)

    def test_extended_adds_only_previously_built_families(self):
        added = set(EXTENDED.features) - set(CORE.features)
        declared = {f for family in EXTENDED_ADDITIONS.values() for f in family}
        assert added == declared

    def test_no_forbidden_token_reaches_any_allowlist(self):
        for feature in ALL_ALLOWED_FEATURES:
            for token in FORBIDDEN_TOKENS:
                assert token not in feature.lower()

    def test_kalshi_can_never_be_a_feature(self):
        assert not [f for f in ALL_ALLOWED_FEATURES if "kalshi" in f.lower()]

    def test_post_anchor_anchors_are_absent(self):
        for feature in ALL_ALLOWED_FEATURES:
            assert "t15" not in feature and "t5m" not in feature

    def test_allowlists_have_no_duplicates(self):
        for bundle in (CORE, EXTENDED):
            assert len(set(bundle.features)) == len(bundle.features)


class TestSearchSpace:
    def test_the_grid_is_compact_not_a_cartesian_product(self):
        cartesian = 1
        for values in CANDIDATE_RANGES.values():
            cartesian *= len(values)
        assert cartesian == 128
        assert 12 <= len(XGBOOST_GRID) <= 24

    def test_trees_stay_shallow(self):
        for config in ALL_CONFIGS:
            depth = config.params.get("max_depth")
            assert depth is not None and depth <= 3

    def test_every_config_is_seeded(self):
        for config in ALL_CONFIGS:
            assert config.params.get("random_state") == RANDOM_SEED

    def test_a_second_family_is_present_for_generality(self):
        assert {c.family for c in ALL_CONFIGS} == {"xgboost", "histgb"}
        assert len(HISTGB_GRID) <= 6

    def test_config_names_are_unique(self):
        names = [c.name for c in ALL_CONFIGS]
        assert len(set(names)) == len(names)

    def test_estimators_are_reproducible(self):
        rng = np.random.RandomState(0)
        x = pd.DataFrame(rng.randn(300, 4), columns=list("abcd"))
        y = (x["a"] + rng.randn(300) * 0.5 > 0).astype(int)
        first = build_estimator(XGBOOST_GRID[0]).fit(x, y).predict_proba(x)[:, 1]
        second = build_estimator(XGBOOST_GRID[0]).fit(x, y).predict_proba(x)[:, 1]
        assert np.allclose(first, second)


class TestCalibrationSafety:
    def _oof_frame(self):
        rng = np.random.RandomState(1)
        n = 3000
        probability = rng.uniform(0.1, 0.9, n)
        outcome = (rng.uniform(size=n) < probability).astype(int)
        return probability, outcome

    def test_a_calibrator_is_refused_when_support_is_thin(self):
        probability, outcome = self._oof_frame()
        thin = fit_calibrator(SIGMOID, probability[:10], outcome[:10])
        assert thin.method == "none"
        assert "below the floor" in thin.reason

    def test_isotonic_needs_more_support_than_sigmoid(self):
        probability, outcome = self._oof_frame()
        n = (MIN_CALIBRATION_ROWS + MIN_ISOTONIC_ROWS) // 2
        assert fit_calibrator(SIGMOID, probability[:n], outcome[:n]).method == SIGMOID
        assert fit_calibrator(ISOTONIC, probability[:n], outcome[:n]).method == "none"

    def test_a_single_class_calibration_set_is_refused(self):
        probability = np.linspace(0.2, 0.8, 1000)
        assert fit_calibrator(SIGMOID, probability, np.ones(1000, int)).method == "none"

    def test_an_unfitted_calibrator_returns_probabilities_unchanged(self):
        calibrator = fit_calibrator("none", [0.3, 0.7], [0, 1])
        assert np.allclose(calibrator.transform(np.array([0.3, 0.7])), [0.3, 0.7])

    def test_calibrated_probabilities_stay_in_range(self):
        probability, outcome = self._oof_frame()
        for method in (SIGMOID, ISOTONIC):
            calibrator = fit_calibrator(method, probability, outcome)
            values = calibrator.transform(np.array([0.0, 0.001, 0.5, 0.999, 1.0]))
            assert np.all((values > 0.0) & (values < 1.0))


class TestChronologicalOutOfFold:
    def _frame(self):
        return pd.DataFrame({
            "season": [2019] * 50 + [2020] * 50 + [2021] * 50 + [2022] * 50,
            "home_win": ([0, 1] * 25) * 4,
        })

    def test_no_inner_fold_trains_on_the_season_it_scores(self):
        seen: list[tuple[set, set]] = []

        def fit_predict(train, predict):
            seen.append((set(train["season"]), set(predict["season"])))
            return np.full(len(predict), 0.5)

        chronological_oof_predictions(
            self._frame(), [2019, 2020, 2021, 2022], fit_predict
        )
        for train_seasons, predict_seasons in seen:
            assert not (train_seasons & predict_seasons)

    def test_every_inner_training_season_precedes_what_it_scores(self):
        seen: list[tuple[set, set]] = []

        def fit_predict(train, predict):
            seen.append((set(train["season"]), set(predict["season"])))
            return np.full(len(predict), 0.5)

        chronological_oof_predictions(
            self._frame(), [2019, 2020, 2021, 2022], fit_predict
        )
        for train_seasons, predict_seasons in seen:
            assert max(train_seasons) < min(predict_seasons)

    def test_the_earliest_season_supplies_training_but_is_never_scored(self):
        oof = chronological_oof_predictions(
            self._frame(), [2019, 2020, 2021, 2022],
            lambda tr, pr: np.full(len(pr), 0.5),
        )
        assert 2019 not in set(oof["season"])
        assert set(oof["season"]) == {2020, 2021, 2022}

    def test_the_validation_season_never_enters_its_own_calibration(self):
        # The outer fold's season is simply not in the training list, so it
        # cannot appear in any inner split.
        training = [2019, 2020, 2021]
        oof = chronological_oof_predictions(
            self._frame(), training, lambda tr, pr: np.full(len(pr), 0.5)
        )
        assert 2022 not in set(oof["season"])

    def test_a_single_training_season_yields_no_calibration_rows(self):
        oof = chronological_oof_predictions(
            self._frame(), [2019], lambda tr, pr: np.full(len(pr), 0.5)
        )
        assert len(oof) == 0


class TestSelectionRules:
    def test_a_negligible_gain_is_not_material(self):
        from nba_prediction_market.pipelines.build_nonlinear_model import (
            MATERIAL_BRIER_GAIN,
            select_candidate,
        )

        summary = pd.DataFrame([
            {"candidate": "xgb01_CORE_none", "mean_brier": 0.21295,
             "mean_log_loss": 0.61},
        ])
        chosen = select_candidate(summary, control_brier=0.21297)
        assert chosen["gain_vs_control"] < MATERIAL_BRIER_GAIN
        assert chosen["material"] is False

    def test_a_real_gain_is_material(self):
        from nba_prediction_market.pipelines.build_nonlinear_model import (
            select_candidate,
        )

        summary = pd.DataFrame([
            {"candidate": "xgb01_CORE_none", "mean_brier": 0.21000,
             "mean_log_loss": 0.60},
        ])
        assert select_candidate(summary, control_brier=0.21297)["material"] is True

    def test_a_worse_candidate_is_never_material(self):
        from nba_prediction_market.pipelines.build_nonlinear_model import (
            select_candidate,
        )

        summary = pd.DataFrame([
            {"candidate": "xgb01_CORE_none", "mean_brier": 0.21500,
             "mean_log_loss": 0.62},
        ])
        chosen = select_candidate(summary, control_brier=0.21297)
        assert chosen["gain_vs_control"] < 0
        assert chosen["material"] is False

    def test_blend_weights_are_predetermined(self):
        from nba_prediction_market.pipelines.build_nonlinear_model import (
            BLEND_WEIGHTS,
        )

        assert BLEND_WEIGHTS == (0.0, 0.25, 0.5, 0.75, 1.0)


class TestPhase3A3CPreserved:
    """Phase 3A4 must not disturb anything Phase 3A3C froze."""

    def test_the_3a3c_bundle_definition_is_untouched(self):
        from nba_prediction_market.models.availability_bundles import (
            AVAILABILITY_BUNDLES_BY_NAME,
            PHASE_3A3_FEATURES,
            ROLE_MINUTE_FAMILY,
        )

        assert AVAILABILITY_BUNDLES_BY_NAME["C"].features == (
            PHASE_3A3_FEATURES + ROLE_MINUTE_FAMILY
        )

    def test_the_3a3c_control_c_is_reused_not_retuned(self):
        from nba_prediction_market.pipelines.build_nonlinear_model import CONTROL_C

        assert CONTROL_C == 0.1

    def test_phase_3a4_reuses_the_3a3c_history_policy(self):
        from nba_prediction_market.pipelines.build_availability_model import (
            MAX_TRAINING_SEASONS,
            training_seasons_for,
        )
        from nba_prediction_market.pipelines.build_nonlinear_model import (
            MAX_TRAINING_SEASONS as reused,
        )

        assert reused == MAX_TRAINING_SEASONS
        assert training_seasons_for(2024, (2019, 2020, 2021, 2022, 2023, 2024)) == [
            2019, 2020, 2021, 2022, 2023
        ]


class TestCalibrationDemonstration:
    def test_the_pipeline_can_prove_its_calibration_is_safe(self):
        from nba_prediction_market.pipelines.build_nonlinear_model import (
            calibration_is_leakage_safe,
        )

        result = calibration_is_leakage_safe()
        assert result["inner_folds_share_no_season"] is True
        assert result["earliest_training_season_not_scored"] is True
