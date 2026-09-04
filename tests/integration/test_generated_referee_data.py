"""Invariants of the generated referee artefacts. ``pytest -m dataset``."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from nba_prediction_market.referees.state import FEATURE_ALLOWLIST

pytestmark = pytest.mark.dataset

PROCESSED = Path("data/processed")
ASSIGNMENTS = PROCESSED / "nba_referee_assignments_2019_26.parquet"
FEATURES = PROCESSED / "nba_referee_features_2019_26.parquet"


@pytest.fixture(scope="module")
def assignments() -> pd.DataFrame:
    if not ASSIGNMENTS.is_file():
        pytest.skip("referee assignments not built yet")
    return pd.read_parquet(ASSIGNMENTS)


@pytest.fixture(scope="module")
def features() -> pd.DataFrame:
    if not FEATURES.is_file():
        pytest.skip("referee features not built yet")
    return pd.read_parquet(FEATURES)


class TestAssignments:
    def test_one_row_per_game(self, assignments):
        assert assignments["nba_game_id"].is_unique

    def test_every_row_carries_its_source(self, assignments):
        assert (assignments["source"] == "basketball_reference").all()
        assert assignments["source_game_identifier"].notna().all()

    def test_slugs_and_names_align_per_game(self, assignments):
        """A mismatch would mean officials were paired by position blindly."""
        bad = [
            row.nba_game_id for row in assignments.itertuples()
            if len(row.referee_slugs) != len(row.referee_names)
        ]
        assert bad == []

    def test_crew_size_matches_the_slug_count(self, assignments):
        counts = assignments["referee_slugs"].apply(len)
        assert (counts == assignments["crew_size"]).all()

    def test_crews_are_plausible_sizes(self, assignments):
        assert set(assignments["crew_size"].unique()) <= {0, 2, 3, 4}

    def test_no_official_appears_twice_in_one_crew(self, assignments):
        dupes = [
            row.nba_game_id for row in assignments.itertuples()
            if len(set(row.referee_slugs)) != len(row.referee_slugs)
        ]
        assert dupes == []

    def test_mapping_quality_is_from_the_known_set(self, assignments):
        allowed = {"mapped", "no_boxscore_page", "no_officials_block", "empty_crew"}
        assert set(assignments["mapping_quality"].unique()) <= allowed

    def test_seasons_are_in_scope(self, assignments):
        assert assignments["season"].between(2019, 2025).all()


class TestFeatures:
    def test_one_row_per_game(self, features):
        assert features["nba_game_id"].is_unique

    def test_columns_match_the_allowlist_exactly(self, features):
        produced = [c for c in features.columns if c.startswith("ref_crew_")]
        assert sorted(produced) == sorted(FEATURE_ALLOWLIST)

    def test_no_referee_feature_is_null(self, features):
        """Absence is encoded as zero plus a coverage flag, never as a null."""
        assert features[list(FEATURE_ALLOWLIST)].notna().all().all()

    def test_games_without_a_crew_have_zeroed_features(self, features):
        blind = features[~features["referee_crew_known"]]
        if blind.empty:
            pytest.skip("every game has a crew")
        assert (blind[list(FEATURE_ALLOWLIST)] == 0).all().all()

    def test_experience_is_never_negative(self, features):
        assert (features["ref_crew_experience_mean"] >= 0).all()
        assert (features["ref_crew_experience_min"] >= 0).all()

    def test_state_grows_monotonically_in_time(self, features):
        """Referee state accumulates; it can never shrink as the season runs."""
        ordered = features.sort_values("game_datetime_utc")["referee_state_games"]
        assert (ordered.diff().dropna() >= 0).all()

    def test_the_earliest_game_has_no_prior_state(self, features):
        first = features.sort_values("game_datetime_utc").iloc[0]
        assert first["referee_state_games"] == 0
        assert all(first[c] == 0.0 for c in FEATURE_ALLOWLIST)

    def test_dispersion_is_never_negative(self, features):
        assert (features["ref_crew_pf_rel_dispersion"] >= 0).all()

    def test_home_residual_stays_within_a_sane_range(self, features):
        """A shrunk probability residual cannot exceed one in magnitude."""
        assert features["ref_crew_home_win_residual"].abs().max() <= 1.0


class TestKalshiReference:
    """The 2025-26 context line must actually resolve, not silently vanish."""

    def test_reference_is_computed_from_real_columns(self):
        from nba_prediction_market.config import load_settings
        from nba_prediction_market.pipelines.build_referee_model import (
            kalshi_benchmark,
        )

        path = PROCESSED / "nba_kalshi_pregame_t30_2025_26.parquet"
        if not path.is_file():
            pytest.skip("Kalshi T-30 frame not built")
        result = kalshi_benchmark(load_settings())
        assert result is not None, "a present frame must yield a reference"
        assert result["n_games"] > 1000
        assert 0.15 < result["brier_score"] < 0.25
        assert 0.5 < result["mean_predicted_probability"] < 0.6

    def test_reference_normalises_the_two_sided_pair(self):
        from nba_prediction_market.config import load_settings
        from nba_prediction_market.pipelines.build_referee_model import (
            kalshi_benchmark,
        )

        if not (PROCESSED / "nba_kalshi_pregame_t30_2025_26.parquet").is_file():
            pytest.skip("Kalshi T-30 frame not built")
        result = kalshi_benchmark(load_settings())
        assert "home + away midpoint" in result["definition"]


CANONICAL_3A3C = PROCESSED / "nba_predictions_3a3c_2025_26.parquet"


class TestControlIsTheCanonicalFrozenModel:
    """Phase 4A4's control must BE the frozen 3A3C model, not a near-copy.

    A control that is merely close silently changes what every referee family
    is measured against. This reproduces the canonical holdout predictions
    exactly -- the same bar Phase 3A4 met.
    """

    @staticmethod
    @pytest.fixture(scope="class")
    def compared():
        import pandas as pd

        from nba_prediction_market.config import load_settings
        from nba_prediction_market.pipelines.build_referee_model import (
            BASE_BUNDLES,
            evaluate,
            frame_for_k,
            load_frame,
        )

        if not (CANONICAL_3A3C.is_file() and FEATURES.is_file()):
            pytest.skip("canonical or referee artefacts not built")
        settings = load_settings()
        frame = frame_for_k(settings, load_frame(settings), 25.0)
        prior = sorted(s for s in frame["season"].unique() if s < 2025)
        result = evaluate(frame, BASE_BUNDLES[0], 2025, prior, 0.1)
        mine = pd.DataFrame({
            "nba_game_id": result["nba_game_id"],
            "p": result["probability"],
        })
        canon = pd.read_parquet(CANONICAL_3A3C)[
            ["nba_game_id", "home_win", "probability_3a3c_native"]
        ]
        return canon.merge(mine, on="nba_game_id", how="outer", indicator=True), result

    def test_game_id_sets_are_equal(self, compared):
        merged, _ = compared
        assert (merged["_merge"] == "both").all()
        assert not merged["nba_game_id"].duplicated().any()

    def test_probabilities_reproduce_exactly(self, compared):
        merged, _ = compared
        diff = (merged["p"] - merged["probability_3a3c_native"]).abs()
        assert diff.max() == 0.0, f"max abs diff {diff.max():.3g}, expected 0"

    def test_the_rolling_training_window_matches_the_frozen_model(self, compared):
        """fit_logistic fits whatever rows it is given; the window is ours to apply."""
        _, result = compared
        assert list(result["training_seasons"]) == [2020, 2021, 2022, 2023, 2024]

    def test_control_metrics_match_the_published_figures(self, compared):
        from nba_prediction_market.models import metrics

        merged, _ = compared
        y = merged["home_win"].astype(int)
        assert round(metrics.brier_score(y, merged["p"]), 5) == 0.20039
        assert round(metrics.log_loss(y, merged["p"]), 5) == 0.58539
        assert round(metrics.roc_auc(y, merged["p"]), 5) == 0.75051
        assert round(metrics.expected_calibration_error(y, merged["p"]), 5) == 0.02754

    def test_no_referee_feature_reaches_the_control(self):
        from nba_prediction_market.models.availability_bundles import (
            AVAILABILITY_BUNDLES_BY_NAME,
        )
        from nba_prediction_market.referees.bundles import CONTROL_FEATURES

        assert AVAILABILITY_BUNDLES_BY_NAME["C"].features == CONTROL_FEATURES
        assert not [f for f in CONTROL_FEATURES if f in set(FEATURE_ALLOWLIST)]
        assert not [f for f in CONTROL_FEATURES if f.startswith("ref_")]
