"""Verified identity corrections, and the absence of any fuzzy fallback."""

from __future__ import annotations

import pytest

from nba_prediction_market.availability.player_aliases import (
    ALIASES_BY_KEY,
    KNOWN_UNRESOLVED,
    NAMING_CONVENTION,
    PRE_FIRST_APPEARANCE,
    PREFERRED_NAME,
    VERIFIED_ALIASES,
    resolve_alias,
)
from nba_prediction_market.availability.postponements import (
    POSTPONEMENTS,
    postponement_for,
)


class TestAliasTable:
    def test_every_alias_carries_evidence_and_a_reason(self):
        for alias in VERIFIED_ALIASES:
            assert alias.evidence.strip()
            assert alias.reason in (PREFERRED_NAME, NAMING_CONVENTION,
                                    PRE_FIRST_APPEARANCE)
            assert isinstance(alias.player_id, int)
            assert alias.canonical_name.strip()

    def test_keys_are_unique(self):
        assert len(ALIASES_BY_KEY) == len(VERIFIED_ALIASES)

    def test_the_seventeen_phase_b1_pairs_are_accounted_for(self):
        from nba_prediction_market.availability.player_aliases import ALIASES_2025_26

        # 16 resolved plus 1 investigated-and-left-unresolved.
        assert len(ALIASES_2025_26) + len(KNOWN_UNRESOLVED) == 17

    def test_the_2024_25_pairs_surfaced_by_recovery_are_all_resolved(self):
        from nba_prediction_market.availability.player_aliases import ALIASES_2024_25

        assert len(ALIASES_2024_25) == 13

    def test_a_player_reported_under_two_teams_has_an_entry_for_each(self):
        # The preferred-name evidence is that team's roster, so it does not
        # transfer to another team on its own.
        assert resolve_alias("Reddish, Cam", "LAL").player_id == 666860
        assert resolve_alias("Reddish, Cam", "CHA").player_id == 666860
        assert resolve_alias("Reddish, Cam", "BOS") is None

    def test_one_player_can_appear_across_seasons_under_different_teams(self):
        # David Jones: Utah in 2024-25, San Antonio in 2025-26, same id.
        assert (resolve_alias("Jones Garcia, David", "UTA").player_id
                == resolve_alias("Jones Garcia, David", "SAS").player_id
                == 1028245237)

    @pytest.mark.parametrize(
        ("name", "team", "player_id"),
        [
            ("Carrington, Bub", "WAS", 1028025235),
            ("Sarr, Alex", "WAS", 1028028405),
            ("Hyland, Bones", "MIN", 17896031),
            ("Jones Garcia, David", "SAS", 1028245237),
            ("Hayes-Davis, Nigel", "PHX", 2221),
            ("Gordon, Eric", "MEM", 178),
        ],
    )
    def test_known_pairs_resolve_to_their_verified_id(self, name, team, player_id):
        assert resolve_alias(name, team).player_id == player_id

    def test_the_same_player_on_two_teams_maps_to_one_id(self):
        assert (resolve_alias("Hayes-Davis, Nigel", "PHX").player_id
                == resolve_alias("Hayes-Davis, Nigel", "MIL").player_id)
        assert (resolve_alias("Conley, Mike", "CHI").player_id
                == resolve_alias("Conley, Mike", "CHA").player_id)


class TestNoFuzzyMatching:
    def test_an_unlisted_name_is_refused(self):
        assert resolve_alias("Nobody, Someone", "BOS") is None

    def test_a_near_miss_spelling_is_refused(self):
        # One character away from a listed alias, and still refused: matching
        # here is exact, never similarity-based.
        assert resolve_alias("Carrington, Bob", "WAS") is None
        assert resolve_alias("Sarr, Alexx", "WAS") is None

    def test_the_right_name_on_the_wrong_team_is_refused(self):
        assert resolve_alias("Carrington, Bub", "BOS") is None

    def test_a_missing_team_is_refused(self):
        assert resolve_alias("Carrington, Bub", None) is None


class TestKnownUnresolved:
    def test_the_unresolved_entry_states_why(self):
        entry = KNOWN_UNRESOLVED[0]
        assert entry.report_name == "Djurisic, Nikola"
        assert "no BALLDONTLIE player record" in entry.reason

    def test_it_is_not_also_aliased(self):
        for entry in KNOWN_UNRESOLVED:
            assert resolve_alias(entry.report_name, entry.report_team) is None


class TestPostponements:
    def test_every_recorded_postponement_is_evidenced(self):
        # 8 COVID-era, 5 from 2024-25 (the January 2025 cluster), 4 from 2025-26.
        assert len(POSTPONEMENTS) == 17

    def test_each_names_its_independent_evidence(self):
        for entry in POSTPONEMENTS:
            assert "ESPN" in entry.evidence
            assert entry.replayed_date != entry.original_date

    @pytest.mark.parametrize(
        ("game_date", "away", "home", "replayed"),
        [
            ("2026-01-08", "MIA", "CHI", "2026-01-29"),
            ("2026-01-24", "GSW", "MIN", "2026-01-25"),
            ("2026-01-25", "DEN", "MEM", "2026-03-18"),
            ("2026-01-25", "DAL", "MIL", "2026-03-31"),
        ],
    )
    def test_each_original_date_maps_to_its_replay(self, game_date, away, home, replayed):
        found = postponement_for(game_date, away, home)
        assert found is not None
        assert found.replayed_date.isoformat() == replayed

    def test_a_game_that_was_played_as_scheduled_is_not_flagged(self):
        assert postponement_for("2026-01-24", "NYK", "PHI") is None

    def test_the_replay_date_is_not_itself_treated_as_postponed(self):
        # Rows for the replayed game are ordinary and must still match.
        assert postponement_for("2026-01-25", "GSW", "MIN") is None

    def test_the_modeling_treatment_is_stated(self):
        payload = POSTPONEMENTS[0].to_dict()
        assert "not attached to the replayed game" in payload["modeling_treatment"]


class TestReportOmissions:
    """A game the league never reported is a source gap, not a parser gap."""

    def test_the_known_omission_is_recorded_with_evidence(self):
        from nba_prediction_market.availability.postponements import (
            REPORT_OMISSIONS,
            omission_for,
        )

        assert len(REPORT_OMISSIONS) == 1
        found = omission_for("2023-10-25", "BOS", "NYK")
        assert found is not None
        assert "STATUS_FINAL" in found.evidence
        assert "omit" in found.evidence

    def test_an_ordinary_game_is_not_flagged_as_omitted(self):
        from nba_prediction_market.availability.postponements import omission_for

        assert omission_for("2023-10-25", "ATL", "CHA") is None

    def test_an_omission_is_never_filled_from_another_game(self):
        from nba_prediction_market.availability.postponements import omission_for

        payload = omission_for("2023-10-25", "BOS", "NYK").to_dict()
        assert "never filled from another game" in payload["modeling_treatment"]

    def test_omissions_and_postponements_are_distinct_concepts(self):
        from nba_prediction_market.availability.postponements import (
            omission_for,
            postponement_for,
        )

        assert postponement_for("2023-10-25", "BOS", "NYK") is None
        assert omission_for("2026-01-24", "GSW", "MIN") is None
