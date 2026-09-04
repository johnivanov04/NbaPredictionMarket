"""Ablation design, materiality bands, and the diagnostics that come after."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nba_prediction_market.pipelines.build_referee_model import (
    C_GRID,
    materiality,
)
from nba_prediction_market.referees.bundles import (
    BASE_BUNDLES,
    CONTROL_FEATURES,
    REFEREE_FAMILIES,
    UNBUILDABLE_BUNDLES,
    build_bundle_f,
)
from nba_prediction_market.referees.diagnostics import (
    MIN_INTERACTION_GAMES,
    segment_report,
    team_referee_support,
)
from nba_prediction_market.referees.state import FEATURE_ALLOWLIST


class TestBundleDesign:
    def test_control_is_the_frozen_3a3c_selection(self):
        assert BASE_BUNDLES[0].name == "A"
        assert BASE_BUNDLES[0].features == CONTROL_FEATURES

    def test_every_bundle_extends_the_control_exactly(self):
        """One family at a time, or the ablation answers nothing."""
        for bundle in BASE_BUNDLES[1:]:
            assert bundle.features[:len(CONTROL_FEATURES)] == CONTROL_FEATURES
            added = bundle.features[len(CONTROL_FEATURES):]
            assert set(added) <= set(FEATURE_ALLOWLIST)
            assert added == REFEREE_FAMILIES[bundle.name]

    def test_no_bundle_reuses_a_control_feature(self):
        for bundle in BASE_BUNDLES:
            assert len(bundle.features) == len(set(bundle.features))

    def test_the_referee_feature_count_stays_small(self):
        assert len(FEATURE_ALLOWLIST) <= 12, "families must not sprawl"

    def test_crew_chief_bundle_is_declared_unbuildable_with_a_reason(self):
        assert "D" in UNBUILDABLE_BUNDLES
        assert "alphabetical" in UNBUILDABLE_BUNDLES["D"]
        assert "D" not in {b.name for b in BASE_BUNDLES}

    def test_bundle_f_is_none_when_nothing_helped(self):
        assert build_bundle_f([]) is None

    def test_bundle_f_unions_only_the_families_that_helped(self):
        bundle = build_bundle_f(["B", "E"])
        added = bundle.features[len(CONTROL_FEATURES):]
        assert set(added) == set(REFEREE_FAMILIES["B"]) | set(REFEREE_FAMILIES["E"])
        assert not set(added) & set(REFEREE_FAMILIES["C"])

    def test_c_grid_is_the_tiny_predetermined_one(self):
        assert C_GRID == (0.1, 1.0, 10.0)


class TestMateriality:
    def test_a_worse_model_is_reported_as_no_improvement(self):
        assert "no improvement" in materiality(+0.001)

    def test_tiny_improvements_are_called_noise(self):
        assert materiality(-0.00005) == "effectively no useful signal"

    def test_small_band_demands_fold_consistency(self):
        assert "requires strong fold consistency" in materiality(-0.0003)

    def test_meaningful_band(self):
        assert materiality(-0.0007) == "meaningful"

    def test_large_improvements_invite_suspicion_not_celebration(self):
        assert "audit aggressively" in materiality(-0.005)

    def test_band_edges_fall_on_the_stated_side(self):
        assert materiality(-0.0001) != "effectively no useful signal"
        assert materiality(-0.0010) == (
            "surprisingly strong; audit aggressively before believing it"
        )


class TestTeamRefereeDiagnostic:
    def _assignments(self, n_games: int) -> pd.DataFrame:
        return pd.DataFrame([
            {"home_team": "BOS", "away_team": "DET",
             "referee_slugs": np.array(["a99r", "b99r", "c99r"])}
            for _ in range(n_games)
        ])

    def test_reports_support_not_win_rates(self):
        """Producing the win-rate table is how a spurious finding is made."""
        result = team_referee_support(self._assignments(40))
        assert "win_rate" not in json_keys(result)
        assert "best_cell" not in json_keys(result)
        assert result["cells"] == 6  # 2 teams x 3 officials

    def test_sparse_support_is_called_insufficient(self):
        result = team_referee_support(self._assignments(5))
        assert result["verdict"].startswith("insufficient")

    def test_ample_support_is_reported_without_endorsing_a_feature(self):
        result = team_referee_support(self._assignments(MIN_INTERACTION_GAMES + 5))
        assert result["share_at_or_above_min"] == 1.0
        assert "later phase" in result["verdict"]

    def test_games_without_a_crew_contribute_nothing(self):
        frame = pd.DataFrame([
            {"home_team": "BOS", "away_team": "DET",
             "referee_slugs": np.array([], dtype=object)}
        ])
        assert team_referee_support(frame)["cells"] == 0


def json_keys(obj) -> set:
    return set(obj) if isinstance(obj, dict) else set()


class TestSegments:
    def _frame(self) -> pd.DataFrame:
        rng = np.random.default_rng(0)
        n = 400
        return pd.DataFrame({
            "home_win": rng.integers(0, 2, n),
            "control_probability": rng.uniform(0.2, 0.8, n),
            "referee_probability": rng.uniform(0.2, 0.8, n),
            "ref_crew_pf_rel": rng.normal(0, 2, n),
            "ref_crew_home_win_residual": rng.normal(0, 0.05, n),
            "ref_crew_experience_mean": rng.integers(0, 800, n),
            "avail_expected_minutes_lost_diff": rng.normal(0, 15, n),
        })

    def test_all_named_segments_are_reported(self):
        rows = segment_report(self._frame())
        labels = {r["segment"] for r in rows}
        for expected in ("all games", "high-whistle crews (top quintile)",
                         "veteran crews", "close games (0.45-0.55)",
                         "home favourites (>=0.65)",
                         "high availability-burden games"):
            assert expected in labels

    def test_each_segment_carries_both_briers_and_a_delta(self):
        for row in segment_report(self._frame()):
            if row["n_games"]:
                assert "control_brier" in row and "referee_brier" in row
                assert row["delta_brier"] == pytest.approx(
                    row["referee_brier"] - row["control_brier"], abs=1e-9
                )

    def test_missing_columns_simply_drop_their_segments(self):
        minimal = pd.DataFrame({"home_win": [1, 0, 1, 0]})
        rows = segment_report(minimal)
        assert [r["segment"] for r in rows] == ["all games"]


class TestHoldoutCannotSelectFeatures:
    """2025-26 must not reach any selection decision."""

    def test_development_seasons_exclude_the_holdout(self):
        from nba_prediction_market.models.selection import (
            DEVELOPMENT_VALIDATION_SEASONS,
            HOLDOUT_SEASON,
        )

        assert HOLDOUT_SEASON not in DEVELOPMENT_VALIDATION_SEASONS
        assert max(DEVELOPMENT_VALIDATION_SEASONS) < HOLDOUT_SEASON

    def test_training_on_the_holdout_raises(self):
        from nba_prediction_market.models.selection import (
            HOLDOUT_SEASON,
            HoldoutLeakageError,
            assert_no_holdout,
        )

        with pytest.raises(HoldoutLeakageError):
            assert_no_holdout([2023, HOLDOUT_SEASON], where="test")

    def test_evaluate_refuses_holdout_training_data(self):
        """The guard sits inside the fold evaluator, not just around it."""
        from nba_prediction_market.models.selection import HoldoutLeakageError
        from nba_prediction_market.pipelines.build_referee_model import evaluate

        with pytest.raises(HoldoutLeakageError):
            evaluate(pd.DataFrame(), BASE_BUNDLES[0], 2024, [2023, 2025], 1.0)

    def test_bundle_f_is_assembled_only_from_development_evidence(self):
        """Its inputs are family names decided on development folds."""
        import inspect

        from nba_prediction_market.referees import bundles

        source = inspect.getsource(bundles.build_bundle_f)
        assert "2025" not in source and "holdout" not in source.lower()

    def test_the_frozen_family_set_is_fixed_before_any_result(self):
        """Families are declared as constants, not derived from an outcome."""
        assert set(REFEREE_FAMILIES) == {"B", "C", "E"}
        for name, features in REFEREE_FAMILIES.items():
            assert features, f"family {name} is empty"
            assert set(features) <= set(FEATURE_ALLOWLIST)


class TestResidualGapDiagnostic:
    """A segment's home-win rate alone is not evidence of a referee effect."""

    def _frame(self, n=400, home_rate=0.6, predicted=0.55):
        rng = np.random.default_rng(1)
        return pd.DataFrame({
            "home_win": (rng.uniform(size=n) < home_rate).astype(int),
            "control_probability": np.full(n, predicted),
            "ref_crew_home_win_residual": rng.normal(0, 0.05, n),
        })

    def test_gap_is_reported_against_what_the_control_predicted(self):
        rows = segment_report(self._frame())
        overall = next(r for r in rows if r["segment"] == "all games")
        assert "control_predicted_home_rate" in overall
        assert overall["residual_gap"] == pytest.approx(
            overall["home_win_rate"] - overall["control_predicted_home_rate"],
            abs=1e-4,
        )

    def test_gap_carries_a_standard_error_so_it_can_be_judged(self):
        rows = segment_report(self._frame(n=400))
        overall = next(r for r in rows if r["segment"] == "all games")
        assert overall["residual_gap_stderr"] == pytest.approx(0.025, abs=1e-3)
        assert overall["gap_in_stderrs"] == pytest.approx(
            overall["residual_gap"] / overall["residual_gap_stderr"], abs=0.02
        )

    def test_a_smaller_segment_gets_a_wider_standard_error(self):
        """Otherwise a tail bin's gap looks as solid as the whole season's."""
        big = segment_report(self._frame(n=1200))
        small = segment_report(self._frame(n=200))
        b = next(r for r in big if r["segment"] == "all games")
        s = next(r for r in small if r["segment"] == "all games")
        assert s["residual_gap_stderr"] > b["residual_gap_stderr"]

    def test_a_perfectly_calibrated_segment_shows_no_gap(self):
        frame = pd.DataFrame({
            "home_win": [1, 0, 1, 0],
            "control_probability": [0.5, 0.5, 0.5, 0.5],
        })
        overall = segment_report(frame)[0]
        assert overall["residual_gap"] == pytest.approx(0.0)

    def test_no_gap_is_computed_without_control_predictions(self):
        frame = pd.DataFrame({"home_win": [1, 0, 1, 0]})
        assert "residual_gap" not in segment_report(frame)[0]
