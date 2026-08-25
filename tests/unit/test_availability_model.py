"""Phase 3A3C: leakage guarantees and availability feature semantics."""

from __future__ import annotations

from typing import ClassVar

import pandas as pd
import pytest

from nba_prediction_market.features.availability_features import (
    NOT_REPORTED,
    ROLE_WINDOW,
    STATUS_ORDER,
    STATUS_RANK,
    PlayerRoleState,
    ReportedPlayer,
    expected_minutes_lost,
    expected_quality_lost,
    status_transitions,
    team_count_features,
    team_minute_features,
)
from nba_prediction_market.models.availability_bundles import (
    ALL_AVAILABILITY_FEATURES,
    AVAILABILITY_BUNDLES_BY_NAME,
    PHASE_3A3_FEATURES,
    conditional_bundle_g,
)
from nba_prediction_market.models.status_calibration import (
    MIN_OBSERVATIONS,
    StatusObservation,
    calibrate,
    check_ordering,
)


class TestPlayerRole:
    """A player's role for game G comes only from games before G."""

    def test_current_game_minutes_cannot_define_the_current_role(self):
        state = PlayerRoleState()
        assert state.expected_minutes("p1") is None      # no history at all
        state.record_game({"p1": 30.0})
        first = state.expected_minutes("p1")
        # A later game changes the weight only for games after it.
        state.record_game({"p1": 0.0})
        assert state.expected_minutes("p1") != first or first is None

    def test_a_player_with_no_history_has_unknown_role_not_zero(self):
        state = PlayerRoleState()
        state.record_game({"known": 30.0})
        assert state.expected_minutes("stranger") is None
        assert state.expected_minutes("known") is not None

    def test_the_role_window_is_bounded(self):
        state = PlayerRoleState()
        for _ in range(ROLE_WINDOW + 5):
            state.record_game({"p1": 30.0})
        for _ in range(ROLE_WINDOW):
            state.record_game({"p1": 0.0, "other": 10.0})
        # Every game in the window has him at zero, so his weight collapses.
        assert state.expected_minutes("p1") is None

    def test_a_thin_record_is_shrunk_toward_zero(self):
        thin, thick = PlayerRoleState(), PlayerRoleState()
        thin.record_game({"p1": 30.0})
        for _ in range(10):
            thick.record_game({"p1": 30.0})
        assert thin.expected_minutes("p1") < thick.expected_minutes("p1")

    def test_quality_is_unknown_without_history(self):
        assert PlayerRoleState().player_quality("p1") is None


class TestStatusSemantics:
    def test_absence_from_a_report_is_never_available(self):
        assert NOT_REPORTED not in STATUS_ORDER
        assert NOT_REPORTED != "available"

    def test_the_documented_order_runs_available_to_out(self):
        assert STATUS_ORDER[0] == "available"
        assert STATUS_ORDER[-1] == "out"
        assert STATUS_RANK["out"] > STATUS_RANK["questionable"]

    def test_counts_and_minutes_measure_different_things(self):
        players = [
            ReportedPlayer("star", "out", 34.0, 2.0),
            ReportedPlayer("fringe", "out", 2.0, -1.0),
        ]
        counts = team_count_features(players)
        minutes = team_minute_features(players)
        assert counts["out_count"] == 2.0
        assert minutes["out_expected_minutes"] == pytest.approx(36.0)

    def test_a_player_with_unknown_role_adds_no_minutes(self):
        players = [ReportedPlayer("unknown", "out", None, None)]
        assert team_count_features(players)["out_count"] == 1.0
        assert team_minute_features(players)["out_expected_minutes"] == 0.0


class TestExpectedLoss:
    MAPPING: ClassVar[dict[str, float]] = {
        "out": 0.0, "questionable": 0.55, "available": 0.9,
    }

    def test_a_starter_out_dominates_a_fringe_player_out(self):
        starter = [ReportedPlayer("s", "out", 34.0, None)]
        fringe = [ReportedPlayer("f", "out", 3.0, None)]
        assert (expected_minutes_lost(starter, self.MAPPING)
                > 10 * expected_minutes_lost(fringe, self.MAPPING))

    def test_an_uncalibrated_status_contributes_nothing(self):
        players = [ReportedPlayer("p", "doubtful", 30.0, None)]
        assert expected_minutes_lost(players, self.MAPPING) == 0.0

    def test_quality_loss_is_zero_without_a_quality_estimate(self):
        players = [ReportedPlayer("p", "out", 30.0, None)]
        assert expected_minutes_lost(players, self.MAPPING) > 0
        assert expected_quality_lost(players, self.MAPPING) == 0.0


class TestLateNews:
    WEIGHTS: ClassVar[dict[str, float]] = {"p1": 30.0}

    def test_a_missing_earlier_report_yields_unknown_not_no_change(self):
        # Collapsing these would claim stability we never observed.
        assert status_transitions(None, {"p1": "out"}, self.WEIGHTS) is None

    def test_a_downgrade_to_out_carries_the_players_minutes(self):
        moved = status_transitions(
            {"p1": "questionable"}, {"p1": "out"}, self.WEIGHTS
        )
        assert moved["late_downgrades"] == 1.0
        assert moved["newly_out_expected_minutes"] == pytest.approx(30.0)

    def test_an_upgrade_is_counted_separately(self):
        moved = status_transitions(
            {"p1": "doubtful"}, {"p1": "probable"}, self.WEIGHTS
        )
        assert moved["late_upgrades"] == 1.0
        assert moved["late_downgrades"] == 0.0
        assert moved["newly_out_expected_minutes"] == 0.0

    def test_an_unchanged_status_moves_nothing(self):
        moved = status_transitions({"p1": "out"}, {"p1": "out"}, self.WEIGHTS)
        assert moved == {"late_downgrades": 0.0, "late_upgrades": 0.0,
                         "newly_out_expected_minutes": 0.0}


class TestStatusCalibration:
    def _observations(self, season: int, n: int = 100):
        return [
            StatusObservation(season, "out", 30.0, 0.0) for _ in range(n)
        ] + [
            StatusObservation(season, "available", 30.0, 28.0) for _ in range(n)
        ]

    def test_a_validation_season_cannot_calibrate_itself(self):
        observations = self._observations(2023) + self._observations(2024)
        calibration = calibrate(observations, [2023])
        assert calibration.training_seasons == (2023,)
        assert 2024 not in calibration.training_seasons

    def test_observations_outside_training_are_dropped_not_trusted(self):
        # A caller that mistakenly passes a validation season must not be able
        # to contaminate the fold.
        only_holdout = self._observations(2025)
        calibration = calibrate(only_holdout, [2023])
        assert calibration.estimates == {}

    def test_a_thin_status_gets_no_multiplier(self):
        observations = [StatusObservation(2023, "doubtful", 20.0, 0.0)
                        for _ in range(MIN_OBSERVATIONS - 1)]
        calibration = calibrate(observations, [2023])
        assert calibration.multiplier("doubtful") is None
        assert "doubtful" not in calibration.as_mapping()

    def test_out_retains_far_less_than_available(self):
        calibration = calibrate(self._observations(2023), [2023])
        assert calibration.multiplier("out") < calibration.multiplier("available")

    def test_the_learned_order_can_be_checked_against_the_documented_one(self):
        calibration = calibrate(self._observations(2023), [2023])
        result = check_ordering(calibration, STATUS_RANK)
        assert result["learned_order"] == ["available", "out"]
        assert result["agrees"] is True

    def test_play_rate_and_minutes_are_both_reported(self):
        calibration = calibrate(self._observations(2023), [2023])
        payload = calibration.estimates["out"].to_dict()
        assert payload["play_rate"] == 0.0
        assert payload["observations"] == 100
        assert "play_rate_stderr" in payload


class TestBundles:
    def test_the_control_is_exactly_the_frozen_phase_3a3_set(self):
        assert AVAILABILITY_BUNDLES_BY_NAME["A"].features == PHASE_3A3_FEATURES
        assert len(PHASE_3A3_FEATURES) == 15

    def test_every_bundle_contains_the_control(self):
        for bundle in AVAILABILITY_BUNDLES_BY_NAME.values():
            assert set(PHASE_3A3_FEATURES) <= set(bundle.features)

    def test_no_bundle_admits_kalshi(self):
        for bundle in AVAILABILITY_BUNDLES_BY_NAME.values():
            for feature in bundle.features:
                assert "kalshi" not in feature.lower()

    def test_no_availability_feature_encodes_same_game_participation(self):
        # Scoped to the availability surface on purpose. The control's
        # ``home_games_played`` counts a team's *prior* games, which is
        # legitimate pregame information and was frozen in Phase 3A3; the risk
        # this guards is a player's participation in the game being predicted.
        banned = ("actual_minutes", "did_play", "participation", "starter",
                  "minutes_played", "same_game")
        for feature in ALL_AVAILABILITY_FEATURES:
            assert not any(token in feature.lower() for token in banned)

    def test_the_control_adds_nothing_of_its_own(self):
        # Anything a bundle contains beyond the control must be an
        # availability feature from the declared families.
        for bundle in AVAILABILITY_BUNDLES_BY_NAME.values():
            extra = set(bundle.features) - set(PHASE_3A3_FEATURES)
            assert extra <= set(ALL_AVAILABILITY_FEATURES)

    def test_no_post_anchor_anchor_appears_in_any_feature(self):
        # T-15m and T-5m fall after the prediction anchor.
        for feature in ALL_AVAILABILITY_FEATURES:
            assert "t15" not in feature
            assert "t5m" not in feature

    def test_bundle_g_requires_both_families_to_have_helped(self):
        assert conditional_bundle_g("expected_loss", include_late_news=False) is None
        assert conditional_bundle_g("late_news", include_late_news=True) is None
        built = conditional_bundle_g("expected_loss", include_late_news=True)
        assert built is not None and built.name == "G"

    def test_bundle_features_are_unique(self):
        for bundle in AVAILABILITY_BUNDLES_BY_NAME.values():
            assert len(set(bundle.features)) == len(bundle.features)


class TestTrainingPolicy:
    def test_a_fold_never_trains_on_its_own_or_a_later_season(self):
        from nba_prediction_market.pipelines.build_availability_model import (
            AVAILABILITY_SEASONS,
            DEVELOPMENT_SEASONS,
            training_seasons_for,
        )

        for season in DEVELOPMENT_SEASONS:
            training = training_seasons_for(season, AVAILABILITY_SEASONS)
            assert all(s < season for s in training)

    def test_the_holdout_never_appears_in_any_training_split(self):
        from nba_prediction_market.pipelines.build_availability_model import (
            AVAILABILITY_SEASONS,
            DEVELOPMENT_SEASONS,
            HOLDOUT_SEASON,
            training_seasons_for,
        )

        for season in (*DEVELOPMENT_SEASONS, HOLDOUT_SEASON):
            assert HOLDOUT_SEASON not in training_seasons_for(
                season, AVAILABILITY_SEASONS
            )

    def test_training_history_is_capped(self):
        from nba_prediction_market.pipelines.build_availability_model import (
            MAX_TRAINING_SEASONS,
            training_seasons_for,
        )

        seasons = tuple(range(2010, 2025))
        assert len(training_seasons_for(2024, seasons)) == MAX_TRAINING_SEASONS


class TestDerivedFeatures:
    def _frame(self):
        row = {"avail_t3h_covered": True, "avail_t1h_covered": True}
        for side in ("home", "away"):
            for anchor in ("t30", "t1h", "t3h"):
                for status in STATUS_ORDER:
                    row[f"avail_{side}_{anchor}_{status}_count"] = 0.0
                    row[f"avail_{side}_{anchor}_{status}_expected_minutes"] = 0.0
                    row[f"avail_{side}_{anchor}_{status}_expected_quality_minutes"] = 0.0
            for anchor in ("t3h", "t1h"):
                for base in ("late_downgrades", "late_upgrades",
                             "newly_out_expected_minutes"):
                    row[f"avail_{side}_{anchor}_to_t30_{base}"] = 0.0
        row["avail_home_t30_out_expected_minutes"] = 30.0
        return pd.DataFrame([row])

    def test_a_folds_calibration_changes_its_features(self):
        from nba_prediction_market.pipelines.build_availability_model import (
            derive_features,
        )

        frame = self._frame()
        strict = calibrate(
            [StatusObservation(2023, "out", 30.0, 0.0) for _ in range(100)], [2023]
        )
        lenient = calibrate(
            [StatusObservation(2023, "out", 30.0, 30.0) for _ in range(100)], [2023]
        )
        a = derive_features(frame, strict)["avail_expected_minutes_lost_diff"].iloc[0]
        b = derive_features(frame, lenient)["avail_expected_minutes_lost_diff"].iloc[0]
        assert a > b

    def test_late_news_is_null_when_the_earlier_report_is_missing(self):
        from nba_prediction_market.pipelines.build_availability_model import (
            derive_features,
        )

        frame = self._frame()
        frame.loc[0, "avail_t3h_covered"] = False
        calibration = calibrate(
            [StatusObservation(2023, "out", 30.0, 0.0) for _ in range(100)], [2023]
        )
        derived = derive_features(frame, calibration)
        assert pd.isna(derived["avail_late_downgrades_diff"].iloc[0])
        assert pd.isna(derived["avail_expected_loss_change_3h_diff"].iloc[0])


class TestReasonCategories:
    """Diagnostic classification only; no model consumes these."""

    @pytest.mark.parametrize(
        ("reason", "expected"),
        [
            ("Injury/Illness - Left Ankle; Sprain", "injury"),
            ("G League - Two-Way", "assignment_g_league"),
            ("G League - On Assignment", "assignment_g_league"),
            ("Injury/Illness - Illness; Non-Covid", "illness"),
            ("Personal Reasons", "personal"),
            ("Not With Team", "personal"),
            ("Injury Management", "rest_management"),
            ("League Suspension", "suspension_other"),
            ("-", "not_given"),
            ("", "not_given"),
            (None, "not_given"),
        ],
    )
    def test_known_phrases_map_to_their_category(self, reason, expected):
        from nba_prediction_market.availability.reason_categories import classify

        assert classify(reason) == expected

    def test_an_assignment_outranks_an_incidental_knock(self):
        # A two-way assignment that also mentions an injury is still an
        # assignment; ordering the rules is what decides this deterministically.
        from nba_prediction_market.availability.reason_categories import classify

        assert classify(
            "G League - Two-Way Injury/Illness - Left Knee; Soreness"
        ) == "assignment_g_league"

    def test_unrecognised_text_is_other_not_a_guess(self):
        from nba_prediction_market.availability.reason_categories import classify

        assert classify("Coach's Decision About Something Novel") == "other"

    def test_classification_is_deterministic(self):
        from nba_prediction_market.availability.reason_categories import classify

        text = "Injury/Illness - Right Wrist; Soreness"
        assert classify(text) == classify(text) == classify(text)

    def test_the_frequency_table_states_it_feeds_no_model(self):
        from nba_prediction_market.availability.reason_categories import frequency_table

        table = frequency_table(["Injury/Illness - X", "G League - Two-Way", None])
        assert table["total"] == 3
        assert "no reason-derived feature" in table["note"]
        assert table["by_category"]["injury"]["count"] == 1


class TestSelectionRule:
    """Within the tolerance band the simplest configuration wins."""

    def _bundles(self):
        return AVAILABILITY_BUNDLES_BY_NAME

    def test_a_clearly_better_bundle_wins_outright(self):
        from nba_prediction_market.pipelines.build_availability_model import (
            select_configuration,
        )

        summary = [
            {"bundle": "A", "C": 1.0, "mean_brier": 0.2140, "mean_log_loss": 0.62},
            {"bundle": "D", "C": 0.1, "mean_brier": 0.2100, "mean_log_loss": 0.61},
        ]
        assert select_configuration(summary, self._bundles())["bundle"] == "D"

    def test_a_near_tie_prefers_fewer_added_features(self):
        # D adds 3 availability features, C adds 4. A 7e-5 gap is noise, so the
        # smaller addition should win even though C sorts first alphabetically.
        from nba_prediction_market.pipelines.build_availability_model import (
            select_configuration,
        )

        summary = [
            {"bundle": "D", "C": 0.1, "mean_brier": 0.212005, "mean_log_loss": 0.6116},
            {"bundle": "C", "C": 0.1, "mean_brier": 0.212079, "mean_log_loss": 0.6128},
        ]
        chosen = select_configuration(summary, self._bundles())
        assert chosen["bundle"] == "D"
        assert chosen["availability_features_added"] == 3

    def test_the_control_wins_when_availability_adds_nothing(self):
        from nba_prediction_market.pipelines.build_availability_model import (
            select_configuration,
        )

        summary = [
            {"bundle": "A", "C": 1.0, "mean_brier": 0.21200, "mean_log_loss": 0.61},
            {"bundle": "F", "C": 0.1, "mean_brier": 0.21203, "mean_log_loss": 0.61},
        ]
        chosen = select_configuration(summary, self._bundles())
        assert chosen["bundle"] == "A"
        assert chosen["availability_features_added"] == 0

    def test_the_rule_is_recorded_in_the_result(self):
        from nba_prediction_market.pipelines.build_availability_model import (
            select_configuration,
        )

        summary = [{"bundle": "A", "C": 1.0, "mean_brier": 0.21, "mean_log_loss": 0.6}]
        assert "fewest availability features" in select_configuration(
            summary, self._bundles()
        )["rule"]


class TestCadenceHarmonization:
    """Thinning to the legacy cadence must not discard legacy reports."""

    def _events(self, rows):
        frame = pd.DataFrame(rows)
        frame["report_timestamp_utc"] = pd.to_datetime(
            frame["report_timestamp_utc"], utc=True
        )
        return frame

    def test_legacy_named_reports_are_all_kept(self):
        # Legacy reports are stamped at :30 almost always -- but on 2025-12-19
        # the league used :45, and filtering on the minute would drop them.
        from nba_prediction_market.pipelines.build_availability_features import (
            restrict_to_legacy_cadence,
        )

        events = self._events([
            {"source_filename": "Injury-Report_2025-12-19_04PM.pdf",
             "report_timestamp_utc": "2025-12-19T21:45:00Z"},
            {"source_filename": "Injury-Report_2024-01-15_05PM.pdf",
             "report_timestamp_utc": "2024-01-15T22:30:00Z"},
        ])
        assert len(restrict_to_legacy_cadence(events)) == 2

    def test_the_modern_era_is_thinned_to_one_report_an_hour(self):
        from nba_prediction_market.pipelines.build_availability_features import (
            restrict_to_legacy_cadence,
        )

        events = self._events([
            {"source_filename": "Injury-Report_2026-01-15_05_00PM.pdf",
             "report_timestamp_utc": "2026-01-15T22:00:00Z"},
            {"source_filename": "Injury-Report_2026-01-15_05_30PM.pdf",
             "report_timestamp_utc": "2026-01-15T22:30:00Z"},
        ])
        kept = restrict_to_legacy_cadence(events)
        assert len(kept) == 1
        assert kept.iloc[0]["source_filename"].endswith("05_30PM.pdf")

    def test_a_purely_legacy_season_is_unchanged(self):
        from nba_prediction_market.pipelines.build_availability_features import (
            restrict_to_legacy_cadence,
        )

        events = self._events([
            {"source_filename": f"Injury-Report_2024-01-15_{h:02d}PM.pdf",
             "report_timestamp_utc": f"2024-01-15T{12 + h}:30:00Z"}
            for h in range(1, 6)
        ])
        assert len(restrict_to_legacy_cadence(events)) == len(events)
