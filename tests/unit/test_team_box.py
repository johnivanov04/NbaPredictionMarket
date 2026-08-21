"""Team-game aggregation, possessions, and four factors."""

from __future__ import annotations

import pandas as pd
import pytest

from nba_prediction_market.features.team_box import (
    COUNTING_STATS,
    TEAM_GAME_COLUMNS,
    aggregate_team_games,
    estimate_possessions,
    four_factors,
)


def totals(**over) -> dict[str, float]:
    base = {"fgm": 40, "fga": 85, "fg3m": 10, "fg3a": 28, "ftm": 15, "fta": 20,
            "oreb": 10, "dreb": 33, "reb": 43, "ast": 24, "stl": 8, "blk": 5,
            "turnover": 14, "pts": 105}
    base.update(over)
    return base


def player(gid=1, team=14, pid=1, minutes=30.0, pts=20, **over):
    row = {"nba_game_id": gid, "team_id": team, "player_id": pid, "minutes": minutes,
           "pts": pts, "fgm": 8, "fga": 17, "fg3m": 2, "fg3a": 6, "ftm": 2, "fta": 3,
           "oreb": 2, "dreb": 6, "reb": 8, "ast": 5, "stl": 1, "blk": 1, "turnover": 3,
           "plus_minus": 5}
    row.update(over)
    return row


# --- possessions -----------------------------------------------------------


def test_possession_estimate_is_plausible_for_a_normal_box_score() -> None:
    possessions = estimate_possessions(totals(), totals())
    assert 85.0 < possessions < 110.0


def test_possessions_average_both_teams() -> None:
    """Both teams in a game share one possession count."""
    a, b = totals(fga=95, turnover=18), totals(fga=78, turnover=10)
    assert estimate_possessions(a, b) == pytest.approx(estimate_possessions(b, a))


def test_more_turnovers_raise_the_possession_estimate() -> None:
    assert estimate_possessions(totals(turnover=25), totals()) > estimate_possessions(
        totals(turnover=5), totals()
    )


def test_offensive_rebounds_lower_the_estimate() -> None:
    """An offensive board extends a possession rather than starting a new one."""
    many = estimate_possessions(totals(oreb=25), totals())
    few = estimate_possessions(totals(oreb=2), totals())
    assert many < few


def test_a_missing_total_yields_no_estimate_rather_than_a_guess() -> None:
    assert estimate_possessions(totals(fga=None), totals()) is None
    assert estimate_possessions(totals(), totals(dreb=None)) is None


def test_zero_rebound_chances_do_not_divide_by_zero() -> None:
    assert estimate_possessions(totals(oreb=0), totals(dreb=0)) is not None


# --- four factors ----------------------------------------------------------


def test_efg_credits_the_extra_point_for_threes() -> None:
    factors = four_factors(totals(fgm=40, fg3m=10, fga=80), totals())
    assert factors["efg_pct"] == pytest.approx((40 + 5) / 80)


def test_turnover_rate_is_per_play() -> None:
    factors = four_factors(totals(turnover=14, fga=85, fta=20), totals())
    assert factors["turnover_rate"] == pytest.approx(14 / (85 + 0.44 * 20 + 14))


def test_oreb_rate_uses_opponent_defensive_rebounds() -> None:
    factors = four_factors(totals(oreb=10), totals(dreb=30))
    assert factors["oreb_rate"] == pytest.approx(10 / 40)


def test_ft_rate_and_true_shooting() -> None:
    factors = four_factors(totals(fta=20, fga=85, pts=105), totals())
    assert factors["ft_rate"] == pytest.approx(20 / 85)
    assert factors["true_shooting_pct"] == pytest.approx(105 / (2 * (85 + 0.44 * 20)))


@pytest.mark.parametrize("factor", ["efg_pct", "turnover_rate", "oreb_rate"])
def test_factors_are_bounded_for_realistic_inputs(factor: str) -> None:
    value = four_factors(totals(), totals())[factor]
    assert 0.0 <= value <= 1.0


def test_a_zero_denominator_yields_none_not_an_exception() -> None:
    factors = four_factors(totals(fga=0, fta=0, turnover=0), totals())
    assert factors["efg_pct"] is None
    assert factors["ft_rate"] is None


# --- aggregation -----------------------------------------------------------


def games_frame() -> pd.DataFrame:
    return pd.DataFrame([{
        "nba_game_id": 1, "season": 2020, "home_team_source_id": 14,
        "away_team_source_id": 11, "home_score": 40, "away_score": 30,
    }])


def test_player_rows_aggregate_into_two_team_observations() -> None:
    rows = pd.DataFrame([
        player(pid=1, team=14, pts=20), player(pid=2, team=14, pts=20),
        player(pid=3, team=11, pts=15), player(pid=4, team=11, pts=15),
    ])
    frame, quality = aggregate_team_games(rows, games_frame())

    assert len(frame) == 2
    assert list(frame.columns) == [c for c in TEAM_GAME_COLUMNS if c in frame.columns]
    home = frame[frame["team_id"] == 14].iloc[0]
    assert home["pts"] == 40
    assert home["is_home"] is True or bool(home["is_home"])
    assert home["opponent_team_id"] == 11
    assert home["opp_pts"] == 30
    assert quality["points_reconcile"] == 2


def test_an_unreconciled_box_score_is_flagged_and_derives_nothing() -> None:
    """Nothing is fabricated: incomplete input yields null efficiency."""
    rows = pd.DataFrame([
        player(pid=1, team=14, pts=18),   # 2 points short of the trusted 40
        player(pid=2, team=14, pts=20),
        player(pid=3, team=11, pts=15), player(pid=4, team=11, pts=15),
    ])
    frame, quality = aggregate_team_games(rows, games_frame())
    home = frame[frame["team_id"] == 14].iloc[0]

    assert bool(home["points_reconcile"]) is False
    assert home["points_delta_vs_trusted"] == -2
    assert bool(home["box_score_complete"]) is False
    assert home["estimated_possessions"] is None or pd.isna(home["estimated_possessions"])
    assert pd.isna(home["offensive_efficiency"])
    assert quality["points_mismatch"] == 1
    # The source rows are preserved, not dropped.
    assert home["pts"] == 38


def test_efficiency_is_per_100_possessions() -> None:
    rows = pd.DataFrame([
        player(pid=1, team=14, pts=40, minutes=48.0),
        player(pid=2, team=11, pts=30, minutes=48.0),
    ])
    frame, _ = aggregate_team_games(rows, games_frame())
    home = frame[frame["team_id"] == 14].iloc[0]
    assert home["offensive_efficiency"] == pytest.approx(
        100.0 * 40 / home["estimated_possessions"]
    )
    assert home["net_efficiency"] == pytest.approx(
        home["offensive_efficiency"] - home["defensive_efficiency"]
    )


def test_defensive_efficiency_is_the_opponents_offence() -> None:
    rows = pd.DataFrame([player(pid=1, team=14, pts=40), player(pid=2, team=11, pts=30)])
    frame, _ = aggregate_team_games(rows, games_frame())
    home = frame[frame["team_id"] == 14].iloc[0]
    away = frame[frame["team_id"] == 11].iloc[0]
    assert home["defensive_efficiency"] == pytest.approx(away["offensive_efficiency"])


def test_dnp_rows_contribute_nothing_but_are_counted() -> None:
    rows = pd.DataFrame([
        player(pid=1, team=14, pts=40, minutes=48.0),
        player(pid=9, team=14, pts=0, minutes=None, fgm=0, fga=0, fg3m=0, fg3a=0,
               ftm=0, fta=0, oreb=0, dreb=0, reb=0, ast=0, stl=0, blk=0, turnover=0),
        player(pid=2, team=11, pts=30),
    ])
    frame, _ = aggregate_team_games(rows, games_frame())
    home = frame[frame["team_id"] == 14].iloc[0]
    assert home["player_rows"] == 2
    assert home["players_with_minutes"] == 1
    assert home["pts"] == 40


def test_overtime_games_show_more_possessions() -> None:
    """A 4OT game runs 68 minutes and should estimate well above a normal game."""
    normal = estimate_possessions(totals(), totals())
    overtime = estimate_possessions(
        totals(fga=115, fta=30, turnover=19), totals(fga=115, fta=30, turnover=19)
    )
    assert overtime > normal * 1.2


def test_every_counting_stat_is_summed() -> None:
    rows = pd.DataFrame([
        player(pid=1, team=14, pts=20), player(pid=2, team=14, pts=20),
        player(pid=3, team=11, pts=30),
    ])
    frame, _ = aggregate_team_games(rows, games_frame())
    home = frame[frame["team_id"] == 14].iloc[0]
    for stat in COUNTING_STATS:
        assert stat in frame.columns
    assert home["ast"] == 10
    assert home["fga"] == 34
