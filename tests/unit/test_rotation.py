"""Rotation continuity, disruption, player quality, and their leakage safety."""

from __future__ import annotations

import copy
from datetime import UTC, datetime, timedelta

import pytest

from nba_prediction_market.features.paid_features import (
    PAID_FEATURE_COLUMNS,
    build_paid_features,
)
from nba_prediction_market.features.rotation import (
    HIGH_MINUTES_THRESHOLD,
    PlayerQuality,
    TeamRotationState,
    herfindahl,
    mean_minutes,
    minute_shares,
    overlap,
    top_n_share,
)

BASE = datetime(2021, 10, 19, tzinfo=UTC)


def even_game(players=(1, 2, 3, 4, 5), minutes=48.0) -> dict[int, float]:
    return dict.fromkeys(players, minutes)


# --- primitives ------------------------------------------------------------


def test_minute_shares_sum_to_one() -> None:
    shares = minute_shares([{1: 30.0, 2: 10.0}, {1: 20.0, 3: 20.0}])
    assert sum(shares.values()) == pytest.approx(1.0)
    assert shares[1] == pytest.approx(50 / 80)


def test_a_player_absent_from_a_game_contributes_zero() -> None:
    shares = minute_shares([{1: 30.0}, {1: 30.0, 2: 30.0}])
    assert shares[2] == pytest.approx(30 / 90)


def test_hhi_is_lowest_for_an_even_rotation() -> None:
    even = herfindahl(minute_shares([even_game()]))
    concentrated = herfindahl(minute_shares([{1: 200.0, 2: 10.0, 3: 10.0}]))
    assert even == pytest.approx(0.2)
    assert concentrated > even


def test_top_n_share_is_monotonic() -> None:
    shares = minute_shares([{i: float(40 - i) for i in range(1, 11)}])
    assert top_n_share(shares, 5) <= top_n_share(shares, 8) <= 1.0


def test_overlap_of_identical_distributions_is_one() -> None:
    a = minute_shares([{1: 30.0, 2: 20.0}])
    assert overlap(a, a) == pytest.approx(1.0)


def test_overlap_of_disjoint_rotations_is_zero() -> None:
    a = minute_shares([{1: 30.0, 2: 20.0}])
    b = minute_shares([{8: 30.0, 9: 20.0}])
    assert overlap(a, b) == pytest.approx(0.0)


def test_overlap_is_minutes_weighted_not_a_head_count() -> None:
    """Losing a starter must cost more than losing a fringe player."""
    baseline = minute_shares([{1: 36.0, 2: 36.0, 3: 4.0}])
    without_starter = minute_shares([{2: 36.0, 3: 4.0}])
    without_fringe = minute_shares([{1: 36.0, 2: 36.0}])
    assert overlap(baseline, without_starter) < overlap(baseline, without_fringe)


def test_mean_minutes_divides_by_games_not_appearances() -> None:
    """A player who missed a game should show a lower average, not the same."""
    played_both = mean_minutes([{1: 30.0}, {1: 30.0}])
    played_one = mean_minutes([{1: 30.0}, {2: 30.0}])
    assert played_both[1] == pytest.approx(30.0)
    assert played_one[1] == pytest.approx(15.0)


def test_empty_windows_yield_none() -> None:
    assert herfindahl({}) is None
    assert top_n_share({}, 5) is None
    assert overlap({}, {1: 1.0}) is None
    assert mean_minutes([]) == {}


# --- player quality --------------------------------------------------------


def test_an_unknown_player_rates_exactly_zero() -> None:
    assert PlayerQuality().rating() == 0.0


def test_a_tiny_sample_is_shrunk_hard_toward_zero() -> None:
    """A 0.3-minute cameo must not produce a huge rating."""
    tiny = PlayerQuality()
    tiny.record(0.3, 5)          # +5 in 18 seconds is +600 per 36 unshrunk
    assert abs(tiny.rating()) < 40.0


def test_shrinkage_relaxes_as_games_accumulate() -> None:
    few, many = PlayerQuality(), PlayerQuality()
    for _ in range(2):
        few.record(30.0, 6)
    for _ in range(60):
        many.record(30.0, 6)
    assert abs(many.rating()) > abs(few.rating())


def test_a_dnp_never_enters_the_quality_estimate() -> None:
    quality = PlayerQuality()
    quality.record(None, 0)
    quality.record(0.0, 0)
    assert quality.games == 0
    assert quality.rating() == 0.0


def test_quality_sign_follows_plus_minus() -> None:
    good, bad = PlayerQuality(), PlayerQuality()
    for _ in range(30):
        good.record(30.0, 10)
        bad.record(30.0, -10)
    assert good.rating() > 0 > bad.rating()


# --- team rotation state ---------------------------------------------------


def test_no_history_yields_all_null_features() -> None:
    features = TeamRotationState().features()
    assert set(features) == set(
        f for f in features
    )
    assert all(value is None for value in features.values())


def test_rotation_features_appear_once_games_are_played() -> None:
    state = TeamRotationState()
    for _ in range(6):
        state.record({p: (30.0, 2) for p in range(1, 10)})
    features = state.features()
    assert features["recent_rotation_player_count"] == 9
    assert features["recent_rotation_minutes_hhi"] == pytest.approx(1 / 9)
    assert features["top5_recent_minutes_share"] == pytest.approx(5 / 9)


def test_a_disappearing_starter_registers_as_disruption() -> None:
    state = TeamRotationState()
    for _ in range(10):
        state.record({1: (36.0, 4), 2: (30.0, 2), 3: (25.0, 0), 4: (20.0, 0)})
    for _ in range(3):
        state.record({2: (34.0, 2), 3: (30.0, 0), 4: (25.0, 0)})   # player 1 gone
    features = state.features()

    assert features["expected_rotation_minutes_missing"] > 30.0
    assert features["high_minutes_player_absence_count"] >= 1
    assert features["rotation_disruption_score"] > 0.0


def test_a_stable_rotation_shows_almost_no_disruption() -> None:
    state = TeamRotationState()
    for _ in range(13):
        state.record({1: (36.0, 4), 2: (30.0, 2), 3: (25.0, 0), 4: (20.0, 0)})
    features = state.features()
    assert features["rotation_disruption_score"] == pytest.approx(0.0, abs=0.02)
    assert features["high_minutes_player_absence_count"] == 0


def test_only_high_minute_players_count_as_absences() -> None:
    state = TeamRotationState()
    for _ in range(10):
        state.record({1: (36.0, 4), 2: (30.0, 2), 9: (3.0, 0)})
    for _ in range(3):
        state.record({1: (36.0, 4), 2: (30.0, 2)})   # only the 3-minute player gone
    features = state.features()
    assert features["high_minutes_player_absence_count"] == 0
    assert HIGH_MINUTES_THRESHOLD == 20.0


def test_expected_rotation_strength_follows_net_team_impact() -> None:
    """Minute-share weighting of a per-36 rating gives a team-level quality."""
    good, bad = TeamRotationState(), TeamRotationState()
    for _ in range(30):
        good.record({1: (40.0, 10), 2: (5.0, -1)})
        bad.record({1: (40.0, -10), 2: (5.0, 1)})
    assert good.features()["expected_rotation_strength"] > 0
    assert bad.features()["expected_rotation_strength"] < 0


def test_expected_rotation_strength_is_dominated_by_high_minute_players() -> None:
    """A fringe player's extreme per-minute rate must not swamp a starter."""
    state = TeamRotationState()
    for _ in range(30):
        state.record({1: (40.0, 8), 2: (2.0, -4)})
    # The starter contributes +8 per game against the bench player's -4 in a
    # twentieth of the minutes, so the team rating stays positive.
    assert state.features()["expected_rotation_strength"] > 0


# --- leakage ---------------------------------------------------------------


def game(gid, day, home, away, season=2021):
    return {
        "nba_game_id": gid, "season": season,
        "game_datetime_utc": BASE + timedelta(days=day),
        "home_franchise_id": home, "away_franchise_id": away,
    }


def observation(gid, team, **over):
    row = {"nba_game_id": gid, "team_id": team, "box_score_complete": True,
           "offensive_efficiency": 110.0, "defensive_efficiency": 105.0,
           "net_efficiency": 5.0, "estimated_pace": 98.0, "efg_pct": 0.52,
           "turnover_rate": 0.13, "oreb_rate": 0.25, "ft_rate": 0.27}
    row.update(over)
    return row


def test_a_games_own_result_cannot_reach_its_own_row() -> None:
    games = [game(1, 0, 14, 11), game(2, 2, 14, 12)]
    team_games = {
        (1, 14): observation(1, 14, offensive_efficiency=90.0),
        (1, 11): observation(1, 11),
        (2, 14): observation(2, 14), (2, 12): observation(2, 12),
    }
    changed = copy.deepcopy(team_games)
    changed[(2, 14)]["offensive_efficiency"] = 150.0
    appearances = {k: {1: (30.0, 5)} for k in team_games}

    first = build_paid_features(games, team_games, appearances)
    second = build_paid_features(games, changed, appearances)
    assert first == second, "changing a game's own result changed its own features"


def test_a_future_result_cannot_reach_an_earlier_row() -> None:
    games = [game(i, i * 2, 14, 10 + i) for i in range(1, 5)]
    team_games = {}
    appearances = {}
    for g in games:
        for team in (14, g["away_franchise_id"]):
            team_games[(g["nba_game_id"], team)] = observation(g["nba_game_id"], team)
            appearances[(g["nba_game_id"], team)] = {1: (30.0, 5)}
    changed = copy.deepcopy(team_games)
    changed[(4, 14)]["offensive_efficiency"] = 200.0

    original = build_paid_features(games, team_games, appearances)
    updated = build_paid_features(games, changed, appearances)
    assert original[:3] == updated[:3]


def test_current_game_minutes_never_enter_rotation_features() -> None:
    games = [game(1, 0, 14, 11)]
    appearances = {(1, 14): {1: (48.0, 10)}, (1, 11): {2: (48.0, -10)}}
    rows = build_paid_features(games, {}, appearances)
    assert rows[0]["home_recent_rotation_player_count"] is None
    assert rows[0]["home_expected_rotation_strength"] is None


def test_state_resets_at_a_season_boundary() -> None:
    games = [game(i, i, 14, 11) for i in range(1, 8)]
    games.append(game(99, 300, 14, 11, season=2022))
    team_games = {}
    appearances = {}
    for g in games:
        for team in (14, 11):
            team_games[(g["nba_game_id"], team)] = observation(g["nba_game_id"], team)
            appearances[(g["nba_game_id"], team)] = {1: (30.0, 5)}
    rows = {r["nba_game_id"]: r for r in build_paid_features(games, team_games, appearances)}

    assert rows[99]["home_season_net_efficiency"] is None
    assert rows[99]["home_recent_rotation_player_count"] is None


def test_an_incomplete_box_score_is_excluded_from_rolling_efficiency() -> None:
    games = [game(1, 0, 14, 11), game(2, 2, 14, 12)]
    team_games = {
        (1, 14): observation(1, 14, box_score_complete=False, offensive_efficiency=None),
        (1, 11): observation(1, 11),
        (2, 14): observation(2, 14), (2, 12): observation(2, 12),
    }
    rows = {r["nba_game_id"]: r for r in build_paid_features(games, team_games, {})}
    assert rows[2]["home_season_offensive_efficiency"] is None
    assert rows[2]["both_teams_box_score_complete_history"] is False


def test_output_has_the_documented_schema() -> None:
    rows = build_paid_features([game(1, 0, 14, 11)], {}, {})
    assert list(rows[0]) == list(PAID_FEATURE_COLUMNS)


def test_feature_generation_is_deterministic() -> None:
    games = [game(i, i, 14, 10 + (i % 3)) for i in range(1, 12)]
    team_games = {
        (g["nba_game_id"], t): observation(g["nba_game_id"], t)
        for g in games for t in (14, g["away_franchise_id"])
    }
    appearances = {k: {1: (30.0, 5), 2: (20.0, -2)} for k in team_games}
    assert build_paid_features(games, team_games, appearances) == build_paid_features(
        copy.deepcopy(games), team_games, appearances
    )
