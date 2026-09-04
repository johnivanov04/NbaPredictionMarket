"""Referee identity, shrinkage, and the sequential-state leakage contract."""

from __future__ import annotations

import pytest

from nba_prediction_market.referees.bbref_source import (
    canonical_team,
    parse_officials,
    parse_schedule_page,
)
from nba_prediction_market.referees.identity import (
    RefereeRegistry,
    build_registry,
    normalize_name,
)
from nba_prediction_market.referees.state import (
    FEATURE_ALLOWLIST,
    FEATURE_FAMILIES,
    GameOutcome,
    RefereeTendencyState,
    shrink,
    shrunk_relative,
)


def outcome(*, pf=40.0, fta=45.0, pts=225.0, home_fta=23.0, away_fta=22.0,
            home_pf=20.0, away_pf=20.0, home_win=1, expected=0.55):
    return GameOutcome(
        total_personal_fouls=pf, total_free_throw_attempts=fta, total_points=pts,
        home_free_throw_attempts=home_fta, away_free_throw_attempts=away_fta,
        home_personal_fouls=home_pf, away_personal_fouls=away_pf,
        home_win=home_win, expected_home_win_prob=expected,
    )


class TestIdentityNormalization:
    def test_accents_and_punctuation_folded(self):
        assert normalize_name("José O'Neill") == "jose oneill"

    def test_generational_suffix_stripped(self):
        assert normalize_name("Ken Mauer Jr.") == "ken mauer"
        assert normalize_name("Sean Wright III") == "sean wright"

    def test_whitespace_collapsed_and_case_folded(self):
        assert normalize_name("  SCOTT   Foster ") == "scott foster"

    def test_identity_is_the_slug_not_the_name(self):
        reg = RefereeRegistry()
        a = reg.add("fostersc99r", "Scott Foster")
        b = reg.add("fostersc99r", "Scott E. Foster")
        assert a is b
        assert reg.renamed["fostersc99r"] == ["Scott E. Foster", "Scott Foster"]

    def test_two_officials_sharing_a_name_are_reported_not_merged(self):
        reg = RefereeRegistry()
        reg.add("willija01r", "James Williams")
        reg.add("willija02r", "James Williams")
        assert len(reg.by_id) == 2
        assert reg.shared_names["james williams"] == ["willija01r", "willija02r"]

    def test_an_unknown_slug_is_never_guessed(self):
        reg = RefereeRegistry()
        reg.add("fostersc99r", "Scott Foster")
        assert reg.resolve("fosterxx99r") is None

    def test_a_name_without_a_slug_is_rejected(self):
        with pytest.raises(ValueError, match="not an identity"):
            RefereeRegistry().add("", "Scott Foster")

    def test_mismatched_slug_and_name_counts_refuse_to_pair(self):
        with pytest.raises(ValueError, match="refusing to pair"):
            build_registry([{
                "nba_game_id": 1,
                "referee_slugs": ["a99r", "b99r"],
                "referee_names": ["Only One"],
            }])


class TestSourceParsing:
    def test_officials_parsed_from_a_boxscore_block(self):
        html = (
            "<div><strong>Officials:&nbsp;</strong>"
            "<a href='/referees/cutleke99r.html'>Kevin Cutler</a>, "
            "<a href='/referees/twardsc99r.html'>Scott Twardoski</a>, "
            "<a href='/referees/willija99r.html'>James Williams</a></div>"
        )
        parsed = parse_officials(html)
        assert parsed.crew_size == 3
        assert parsed.officials[0] == ("cutleke99r", "Kevin Cutler")
        assert parsed.found_block

    def test_a_missing_block_is_distinct_from_an_empty_one(self):
        """A parser regression must not look like a game without officials."""
        absent = parse_officials("<div>no officials here</div>")
        assert absent.found_block is False and absent.crew_size == 0

        empty = parse_officials("<div><strong>Officials:&nbsp;</strong></div>")
        assert empty.found_block is True and empty.crew_size == 0

    def test_two_official_crew_is_preserved_not_padded(self):
        html = (
            "<div><strong>Officials:&nbsp;</strong>"
            "<a href='/referees/a99r.html'>A One</a>, "
            "<a href='/referees/b99r.html'>B Two</a></div>"
        )
        assert parse_officials(html).crew_size == 2

    def test_schedule_row_yields_url_date_and_both_teams(self):
        html = (
            '<tr><th csk="202310240DEN"></th>'
            '<td data-stat="visitor_team_name"><a href="/teams/LAL/2024.html">L</a></td>'
            '<td data-stat="home_team_name"><a href="/teams/DEN/2024.html">D</a></td>'
            '<td><a href="/boxscores/202310240DEN.html">Box</a></td></tr>'
        )
        games = parse_schedule_page(html)
        assert len(games) == 1
        assert games[0].key == ("2023-10-24", "LAL", "DEN")

    def test_bbref_team_codes_map_to_canonical(self):
        assert canonical_team("PHO") == "PHX"
        assert canonical_team("BRK") == "BKN"
        assert canonical_team("CHO") == "CHA"
        assert canonical_team("DEN") == "DEN"


class TestShrinkage:
    def test_no_history_returns_the_league_baseline(self):
        assert shrink(99.0, 0, 40.0, 50.0) == 40.0

    def test_relative_value_is_zero_without_history(self):
        assert shrunk_relative(99.0, 0, 40.0, 50.0) == 0.0

    def test_shrinkage_is_monotone_in_sample_size(self):
        five = shrunk_relative(50.0, 5, 40.0, 50.0)
        five_hundred = shrunk_relative(50.0, 500, 40.0, 50.0)
        assert 0 < five < five_hundred < 10.0

    def test_five_games_and_five_hundred_differ_sharply(self):
        """The whole point of shrinkage: confidence must scale with evidence."""
        assert shrunk_relative(50.0, 5, 40.0, 50.0) == pytest.approx(10 * 5 / 55)
        assert shrunk_relative(50.0, 500, 40.0, 50.0) == pytest.approx(10 * 500 / 550)

    def test_at_n_equals_k_the_weight_is_exactly_half(self):
        assert shrink(60.0, 50, 40.0, 50.0) == pytest.approx(50.0)

    def test_zero_k_is_rejected(self):
        with pytest.raises(ValueError, match="k must be positive"):
            shrink(1.0, 1, 0.0, 0.0)


class TestSequentialState:
    def test_current_game_cannot_influence_its_own_features(self):
        """The central leakage property."""
        state = RefereeTendencyState(k=10.0)
        for _ in range(20):
            state.update(["a"], outcome(pf=40.0))
        before = state.features_for(["a"])

        loud = RefereeTendencyState(k=10.0)
        for _ in range(20):
            loud.update(["a"], outcome(pf=40.0))
        after = loud.features_for(["a"])
        # Reading twice, with the current game not yet applied, is stable.
        assert before == after

        loud.update(["a"], outcome(pf=70.0))
        assert loud.features_for(["a"]) != before

    def test_future_games_cannot_influence_earlier_features(self):
        state = RefereeTendencyState(k=10.0)
        state.update(["a"], outcome(pf=40.0))
        early = state.features_for(["a"])
        for _ in range(50):
            state.update(["a"], outcome(pf=70.0))
        assert state.features_for(["a"]) != early
        # The earlier reading is a value, unaffected by anything after it.
        assert early["ref_crew_pf_rel"] == pytest.approx(0.0)

    def test_experience_counts_only_prior_games(self):
        state = RefereeTendencyState()
        assert state.features_for(["a"])["ref_crew_experience_mean"] == 0
        state.update(["a"], outcome())
        assert state.features_for(["a"])["ref_crew_experience_mean"] == 1
        state.update(["a"], outcome())
        assert state.features_for(["a"])["ref_crew_experience_mean"] == 2

    def test_experience_min_reflects_the_least_experienced_official(self):
        state = RefereeTendencyState()
        for _ in range(10):
            state.update(["vet"], outcome())
        state.update(["rookie"], outcome())
        feats = state.features_for(["vet", "rookie"])
        assert feats["ref_crew_experience_min"] == 1
        assert feats["ref_crew_experience_mean"] == pytest.approx(5.5)

    def test_league_baseline_is_sequential_not_final(self):
        """A tendency measured against a final-season average would leak."""
        state = RefereeTendencyState(k=1.0)
        state.update(["a"], outcome(pf=40.0))
        state.update(["b"], outcome(pf=40.0))
        assert state.league.mean("total_personal_fouls") == 40.0
        state.update(["c"], outcome(pf=60.0))
        assert state.league.mean("total_personal_fouls") == pytest.approx(140 / 3)

    def test_a_referee_matching_the_league_has_zero_tendency(self):
        state = RefereeTendencyState(k=5.0)
        for _ in range(30):
            state.update(["a"], outcome(pf=40.0))
        assert state.features_for(["a"])["ref_crew_pf_rel"] == pytest.approx(0.0)

    def test_a_high_whistle_referee_scores_positive(self):
        state = RefereeTendencyState(k=5.0)
        for _ in range(30):
            state.update(["calm"], outcome(pf=30.0))
        for _ in range(30):
            state.update(["loud"], outcome(pf=50.0))
        assert state.features_for(["loud"])["ref_crew_pf_rel"] > 3.0
        assert state.features_for(["calm"])["ref_crew_pf_rel"] < -3.0

    def test_expectation_adjusted_home_effect_uses_the_residual(self):
        """A crew whose home teams won exactly as expected scores zero."""
        state = RefereeTendencyState(k=1.0)
        for _ in range(50):
            state.update(["fair"], outcome(home_win=1, expected=1.0))
        assert state.features_for(["fair"])["ref_crew_home_win_residual"] == (
            pytest.approx(0.0)
        )

    def test_home_favouring_crew_scores_above_a_neutral_league(self):
        state = RefereeTendencyState(k=5.0)
        for _ in range(40):
            state.update(["neutral"], outcome(home_win=1, expected=1.0))
        for _ in range(40):
            state.update(["homer"], outcome(home_win=1, expected=0.5))
        assert state.features_for(["homer"])["ref_crew_home_win_residual"] > 0.2

    def test_crew_features_average_across_officials(self):
        state = RefereeTendencyState(k=1.0)
        for _ in range(50):
            state.update(["hi"], outcome(pf=60.0))
            state.update(["lo"], outcome(pf=20.0))
        hi = state.features_for(["hi"])["ref_crew_pf_rel"]
        lo = state.features_for(["lo"])["ref_crew_pf_rel"]
        both = state.features_for(["hi", "lo"])["ref_crew_pf_rel"]
        assert both == pytest.approx((hi + lo) / 2)

    def test_dispersion_is_zero_for_an_agreeing_crew(self):
        state = RefereeTendencyState(k=1.0)
        for _ in range(20):
            state.update(["a"], outcome(pf=40.0))
            state.update(["b"], outcome(pf=40.0))
        assert state.features_for(["a", "b"])["ref_crew_pf_rel_dispersion"] == (
            pytest.approx(0.0)
        )

    def test_dispersion_is_positive_for_a_split_crew(self):
        state = RefereeTendencyState(k=1.0)
        for _ in range(30):
            state.update(["hi"], outcome(pf=60.0))
            state.update(["lo"], outcome(pf=20.0))
        assert state.features_for(["hi", "lo"])["ref_crew_pf_rel_dispersion"] > 5

    def test_home_fta_differential_is_league_relative(self):
        state = RefereeTendencyState(k=2.0)
        for _ in range(40):
            state.update(["a"], outcome(home_fta=25.0, away_fta=20.0))
        # Every game in the league has the same differential, so nobody deviates.
        assert state.features_for(["a"])["ref_crew_home_fta_diff_rel"] == (
            pytest.approx(0.0)
        )

    def test_empty_crew_yields_zeroed_features_not_nulls(self):
        feats = RefereeTendencyState().features_for([])
        assert set(feats) == set(FEATURE_ALLOWLIST)
        assert all(v == 0.0 for v in feats.values())

    def test_an_unseen_referee_contributes_no_signal(self):
        state = RefereeTendencyState(k=10.0)
        for _ in range(50):
            state.update(["known"], outcome(pf=60.0))
        assert state.features_for(["debutant"])["ref_crew_pf_rel"] == (
            pytest.approx(0.0)
        )


class TestFeatureAllowlist:
    def test_features_match_the_allowlist_exactly(self):
        feats = RefereeTendencyState().features_for(["a", "b", "c"])
        assert set(feats) == set(FEATURE_ALLOWLIST)

    def test_families_partition_the_allowlist(self):
        listed = [f for fam in FEATURE_FAMILIES.values() for f in fam]
        assert sorted(listed) == sorted(FEATURE_ALLOWLIST)
        assert len(listed) == len(set(listed)), "a feature appears in two families"

    def test_no_position_features_are_claimed(self):
        """BBRef lists officials alphabetically; position is not recoverable."""
        assert not [f for f in FEATURE_ALLOWLIST if "chief" in f or "umpire" in f]


class TestEraShifts:
    """League-wide regime changes must land on the baseline, not on officials.

    The 2019-20 bubble and much of 2020-21 were played in empty or
    near-empty arenas, which shifts home advantage for everyone. Because every
    tendency is a deviation from a *sequential* league baseline, an era-wide
    shift is absorbed by that baseline instead of being attributed to whichever
    officials happened to work those games.
    """

    def test_a_league_wide_home_collapse_creates_no_referee_tendency(self):
        state = RefereeTendencyState(k=5.0)
        # Normal era: home teams win as the strength model expects.
        for i in range(60):
            state.update([f"r{i % 6}"], outcome(home_win=1, expected=1.0))
        # Empty-arena era: every official's games lose the home edge equally.
        for i in range(60):
            state.update([f"r{i % 6}"], outcome(home_win=0, expected=1.0))
        for i in range(6):
            assert state.features_for([f"r{i}"])["ref_crew_home_win_residual"] == (
                pytest.approx(0.0, abs=1e-9)
            )

    def test_an_official_who_differs_from_their_era_still_shows(self):
        """Absorbing the era must not also absorb a genuine deviation."""
        state = RefereeTendencyState(k=5.0)
        for i in range(60):
            state.update([f"r{i % 6}"], outcome(home_win=0, expected=1.0))
        for _ in range(60):
            state.update(["odd"], outcome(home_win=1, expected=1.0))
        assert state.features_for(["odd"])["ref_crew_home_win_residual"] > 0.3

    def test_a_league_wide_whistle_shift_creates_no_tendency(self):
        state = RefereeTendencyState(k=5.0)
        for i in range(40):
            state.update([f"r{i % 4}"], outcome(pf=30.0))
        for i in range(40):
            state.update([f"r{i % 4}"], outcome(pf=50.0))
        for i in range(4):
            assert state.features_for([f"r{i}"])["ref_crew_pf_rel"] == (
                pytest.approx(0.0, abs=1e-9)
            )

    def test_experience_accumulates_across_eras_without_reset(self):
        state = RefereeTendencyState()
        for _ in range(30):
            state.update(["a"], outcome())
        before = state.experience("a")
        for _ in range(30):
            state.update(["a"], outcome(pf=60.0))
        assert state.experience("a") == before + 30
