"""The chronological walk that turns referee state into per-game features."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pandas as pd
import pytest

from nba_prediction_market.pipelines.build_referee_features import (
    ASSUMED_GAME_DURATION_HOURS,
    build_features,
)
from nba_prediction_market.referees.state import FEATURE_ALLOWLIST

START = datetime(2021, 10, 19, 23, 0, tzinfo=UTC)


def frame(rows):
    return pd.DataFrame(rows).sort_values(
        ["game_datetime_utc", "nba_game_id"], kind="mergesort"
    ).reset_index(drop=True)


def game(gid, tip, crew, *, pf=40.0, home_win=1, expected=0.55, season=2021):
    return {
        "nba_game_id": gid, "season": season, "game_datetime_utc": tip,
        "home_team": "AAA", "away_team": "BBB",
        "home_score": 110, "away_score": 100, "home_win": home_win,
        "modeling_eligible": True,
        "referee_slugs": crew, "crew_size": len(crew), "mapping_quality": "mapped",
        "mov_elo_probability": expected,
        "total_personal_fouls": pf, "total_free_throw_attempts": 45.0,
        "total_points": 225.0,
        "home_fta": 23.0, "away_fta": 22.0, "home_pf": 20.0, "away_pf": 20.0,
    }


class TestChronology:
    def test_first_game_has_no_referee_information(self):
        out = build_features(frame([game(1, START, ["a", "b", "c"])]))
        assert out.loc[0, "referee_state_games"] == 0
        assert all(out.loc[0, f] == 0.0 for f in FEATURE_ALLOWLIST)

    def test_a_later_game_sees_an_earlier_completed_one(self):
        rows = [
            game(1, START, ["a", "b", "c"]),
            game(2, START + timedelta(days=1), ["a", "b", "c"]),
        ]
        out = build_features(frame(rows))
        assert out.loc[0, "referee_state_games"] == 0
        assert out.loc[1, "referee_state_games"] == 1
        assert out.loc[1, "ref_crew_experience_mean"] == 1

    def test_same_night_games_do_not_inform_each_other(self):
        """A 19:00 game is still being played when the 19:30 game tips."""
        rows = [
            game(1, START, ["a", "b", "c"]),
            game(2, START + timedelta(minutes=30), ["a", "b", "c"]),
        ]
        out = build_features(frame(rows))
        assert out.loc[1, "referee_state_games"] == 0
        assert out.loc[1, "ref_crew_experience_mean"] == 0

    def test_a_game_is_visible_once_it_has_plausibly_finished(self):
        rows = [
            game(1, START, ["a", "b", "c"]),
            game(2, START + timedelta(hours=ASSUMED_GAME_DURATION_HOURS),
                 ["a", "b", "c"]),
        ]
        out = build_features(frame(rows))
        assert out.loc[1, "referee_state_games"] == 1

    def test_changing_a_games_own_result_never_moves_its_own_features(self):
        base = [game(i, START + timedelta(days=i), ["a", "b", "c"]) for i in range(6)]
        first = build_features(frame(base))

        flipped = [dict(r) for r in base]
        flipped[-1]["total_personal_fouls"] = 90.0
        flipped[-1]["home_win"] = 0
        second = build_features(frame(flipped))

        last = len(base) - 1
        for column in FEATURE_ALLOWLIST:
            assert first.loc[last, column] == pytest.approx(second.loc[last, column])

    def test_changing_a_future_game_never_moves_earlier_features(self):
        base = [game(i, START + timedelta(days=i), ["a", "b", "c"]) for i in range(6)]
        first = build_features(frame(base))

        changed = [dict(r) for r in base]
        changed[-1]["total_personal_fouls"] = 90.0
        second = build_features(frame(changed))

        for column in FEATURE_ALLOWLIST:
            for i in range(len(base) - 1):
                assert first.loc[i, column] == pytest.approx(second.loc[i, column])


class TestMissingData:
    def test_a_game_without_a_crew_is_marked_and_zeroed(self):
        out = build_features(frame([game(1, START, [])]))
        assert not out.loc[0, "referee_crew_known"]
        assert out.loc[0, "referee_crew_size"] == 0

    def test_a_game_without_a_crew_does_not_update_state(self):
        rows = [
            game(1, START, []),
            game(2, START + timedelta(days=1), ["a", "b", "c"]),
        ]
        out = build_features(frame(rows))
        assert out.loc[1, "referee_state_games"] == 0

    def test_a_game_with_missing_measures_is_never_imputed(self):
        rows = [
            game(1, START, ["a", "b", "c"]),
            game(2, START + timedelta(days=1), ["a", "b", "c"]),
        ]
        rows[0]["total_free_throw_attempts"] = None
        out = build_features(frame(rows))
        # The unmeasurable game updates nobody rather than contributing a guess.
        assert out.loc[1, "referee_state_games"] == 0

    def test_a_missing_expectation_excludes_the_game_from_state(self):
        rows = [
            game(1, START, ["a", "b", "c"]),
            game(2, START + timedelta(days=1), ["a", "b", "c"]),
        ]
        rows[0]["mov_elo_probability"] = None
        out = build_features(frame(rows))
        assert out.loc[1, "referee_state_games"] == 0

    def test_two_official_crews_are_handled_without_padding(self):
        rows = [
            game(1, START, ["a", "b"]),
            game(2, START + timedelta(days=1), ["a", "b"]),
        ]
        out = build_features(frame(rows))
        assert out.loc[1, "referee_crew_size"] == 2
        assert out.loc[1, "ref_crew_experience_mean"] == 1


class TestSeasonTransitions:
    def test_state_carries_across_a_season_boundary(self):
        """Officials do not reset in October; their history is continuous."""
        rows = [
            game(1, START, ["a", "b", "c"], season=2021),
            game(2, START + timedelta(days=250), ["a", "b", "c"], season=2022),
        ]
        out = build_features(frame(rows))
        assert out.loc[1, "ref_crew_experience_mean"] == 1
        assert out.loc[1, "season"] == 2022

    def test_output_columns_match_the_allowlist(self):
        out = build_features(frame([game(1, START, ["a"])]))
        produced = [c for c in out.columns if c.startswith("ref_crew_")]
        assert sorted(produced) == sorted(FEATURE_ALLOWLIST)


class TestShrinkageGridIsReal:
    """k is baked into the feature values, so the grid must rebuild them."""

    def _built(self, k: float) -> pd.DataFrame:
        rows = []
        for i in range(60):
            # One loud official, one quiet one, so a deviation exists to shrink.
            crew = ["loud"] if i % 2 else ["quiet"]
            rows.append(game(i, START + timedelta(days=i), crew,
                             pf=60.0 if i % 2 else 20.0))
        return build_features(frame(rows), k=k)

    def test_larger_k_shrinks_tendencies_harder(self):
        spreads = [
            self._built(k)["ref_crew_pf_rel"].abs().max() for k in (10.0, 50.0, 200.0)
        ]
        assert spreads[0] > spreads[1] > spreads[2] > 0

    def test_different_k_yields_different_features(self):
        """Reading one parquet for three k values would compare it to itself."""
        a = self._built(10.0)["ref_crew_pf_rel"]
        b = self._built(200.0)["ref_crew_pf_rel"]
        assert not a.equals(b)

    def test_experience_is_unaffected_by_shrinkage(self):
        """Experience is a count, not an estimate; k must not touch it."""
        a = self._built(10.0)["ref_crew_experience_mean"]
        b = self._built(200.0)["ref_crew_experience_mean"]
        assert a.equals(b)
