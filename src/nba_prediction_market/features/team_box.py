"""Player-game rows aggregated into one trustworthy observation per team-game.

Two rules govern this module:

* **Nothing is fabricated.** Where the player feed does not reconcile with the
  trusted game record, the aggregate is flagged incomplete and the derived
  values become null rather than being filled in. The Phase 3A3A0 audit found 16
  team-games where player points fall short of the recorded team score; those
  rows are preserved and marked, not repaired.
* **Everything here is a game *result*.** These values may only ever influence
  *later* games. The lagging is done in
  :mod:`nba_prediction_market.features.paid_features`; this module just measures
  what happened.

Possessions are an **estimate**, not an official NBA statistic. The formula is
Dean Oliver's, as used by Basketball-Reference: each team's possession count is
estimated from its own and its opponent's box score, and the two estimates are
averaged so both teams in a game share one possession count (which is very
nearly true in reality, since possessions alternate).
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from typing import Any, Final

import pandas as pd

logger = logging.getLogger(__name__)

#: Box-score totals summed from player rows.
COUNTING_STATS: Final[tuple[str, ...]] = (
    "fgm", "fga", "fg3m", "fg3a", "ftm", "fta",
    "oreb", "dreb", "reb", "ast", "stl", "blk", "turnover", "pts",
)

#: Oliver's coefficients. 0.44 is the standard estimate of the share of free
#: throws that end a possession; 1.07 corrects the offensive-rebound term.
FT_POSSESSION_WEIGHT: Final = 0.44
OREB_CORRECTION: Final = 1.07

TEAM_GAME_COLUMNS: Final[tuple[str, ...]] = (
    "nba_game_id", "season", "team_id", "opponent_team_id", "is_home",
    *COUNTING_STATS,
    *[f"opp_{s}" for s in COUNTING_STATS],
    "player_rows", "players_with_minutes", "team_minutes",
    "points_reconcile", "points_delta_vs_trusted",
    "box_score_complete",
    "estimated_possessions", "offensive_efficiency", "defensive_efficiency",
    "net_efficiency", "estimated_pace",
    "efg_pct", "turnover_rate", "oreb_rate", "ft_rate", "true_shooting_pct",
    "opp_efg_pct", "opp_turnover_rate", "opp_oreb_rate", "opp_ft_rate",
)


def _safe_divide(numerator: float | None, denominator: float | None) -> float | None:
    """Division that yields ``None`` rather than an exception or an infinity."""
    if numerator is None or denominator is None:
        return None
    if denominator == 0:
        return None
    return numerator / denominator


def estimate_possessions(team: Mapping[str, float], opponent: Mapping[str, float]) -> float | None:
    """Oliver's possession estimate, averaged across both teams.

    ``0.5 * (team_estimate + opponent_estimate)`` where each estimate is::

        FGA + 0.44*FTA - 1.07*(OREB / (OREB + Opp_DREB)) * (FGA - FGM) + TOV

    Returns ``None`` when any required total is missing -- a partial box score
    must not silently produce a confident-looking possession count.
    """
    def side(a: Mapping[str, float], b: Mapping[str, float]) -> float | None:
        required = ("fga", "fta", "fgm", "oreb", "turnover")
        if any(a.get(k) is None for k in required) or b.get("dreb") is None:
            return None
        oreb_chances = a["oreb"] + b["dreb"]
        oreb_share = 0.0 if oreb_chances == 0 else a["oreb"] / oreb_chances
        missed = a["fga"] - a["fgm"]
        return (
            a["fga"]
            + FT_POSSESSION_WEIGHT * a["fta"]
            - OREB_CORRECTION * oreb_share * missed
            + a["turnover"]
        )

    ours, theirs = side(team, opponent), side(opponent, team)
    if ours is None or theirs is None:
        return None
    return 0.5 * (ours + theirs)


def four_factors(
    team: Mapping[str, float], opponent: Mapping[str, float]
) -> dict[str, float | None]:
    """The four factors plus true shooting, from box-score totals.

    * ``efg_pct = (FGM + 0.5*FG3M) / FGA`` -- shooting, crediting the extra point.
    * ``turnover_rate = TOV / (FGA + 0.44*FTA + TOV)`` -- turnovers per play.
    * ``oreb_rate = OREB / (OREB + Opp_DREB)`` -- share of available offensive boards.
    * ``ft_rate = FTA / FGA`` -- how often a team gets to the line.
    * ``true_shooting_pct = PTS / (2 * (FGA + 0.44*FTA))``.
    """
    fga, fgm, fg3m = team.get("fga"), team.get("fgm"), team.get("fg3m")
    fta, tov, pts = team.get("fta"), team.get("turnover"), team.get("pts")
    oreb, opp_dreb = team.get("oreb"), opponent.get("dreb")

    efg = None
    if None not in (fgm, fg3m, fga):
        efg = _safe_divide(fgm + 0.5 * fg3m, fga)
    turnover_rate = None
    if None not in (tov, fga, fta):
        turnover_rate = _safe_divide(tov, fga + FT_POSSESSION_WEIGHT * fta + tov)
    oreb_rate = None
    if None not in (oreb, opp_dreb):
        oreb_rate = _safe_divide(oreb, oreb + opp_dreb)
    true_shooting = None
    if None not in (pts, fga, fta):
        true_shooting = _safe_divide(pts, 2.0 * (fga + FT_POSSESSION_WEIGHT * fta))
    return {
        "efg_pct": efg,
        "turnover_rate": turnover_rate,
        "oreb_rate": oreb_rate,
        "ft_rate": _safe_divide(fta, fga),
        "true_shooting_pct": true_shooting,
    }


def aggregate_team_games(
    player_rows: pd.DataFrame,
    games: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Aggregate player rows into two observations per game, with quality flags.

    ``games`` supplies the trusted score and identity, which is what the player
    totals are reconciled against. A team-game whose player points do not match
    the trusted score is marked ``box_score_complete = False`` and every derived
    efficiency value is set to ``None``.
    """
    totals = (
        player_rows.groupby(["nba_game_id", "team_id"], as_index=False)
        .agg(
            **{stat: (stat, "sum") for stat in COUNTING_STATS},
            player_rows=("player_id", "size"),
            players_with_minutes=("minutes", lambda s: int((s.fillna(0) > 0).sum())),
            team_minutes=("minutes", "sum"),
        )
    )

    home = games[["nba_game_id", "season", "home_team_source_id", "away_team_source_id",
                  "home_score"]].rename(
        columns={"home_team_source_id": "team_id", "away_team_source_id": "opponent_team_id",
                 "home_score": "trusted_points"}
    )
    home["is_home"] = True
    away = games[["nba_game_id", "season", "away_team_source_id", "home_team_source_id",
                  "away_score"]].rename(
        columns={"away_team_source_id": "team_id", "home_team_source_id": "opponent_team_id",
                 "away_score": "trusted_points"}
    )
    away["is_home"] = False
    identity = pd.concat([home, away], ignore_index=True)

    merged = identity.merge(totals, on=["nba_game_id", "team_id"], how="left")
    merged["points_delta_vs_trusted"] = merged["pts"] - merged["trusted_points"]
    merged["points_reconcile"] = merged["points_delta_vs_trusted"] == 0
    merged["box_score_complete"] = (
        merged["points_reconcile"].fillna(False) & merged["fga"].notna()
    )

    # Attach the opponent's totals so possessions and rebound rates can be formed.
    opponent = merged[["nba_game_id", "team_id", *COUNTING_STATS]].rename(
        columns={"team_id": "opponent_team_id", **{s: f"opp_{s}" for s in COUNTING_STATS}}
    )
    merged = merged.merge(opponent, on=["nba_game_id", "opponent_team_id"], how="left")

    derived = [_derive(row) for row in merged.to_dict("records")]
    frame = pd.concat(
        [merged.reset_index(drop=True), pd.DataFrame(derived)], axis=1
    )
    frame = frame.loc[:, [c for c in TEAM_GAME_COLUMNS if c in frame.columns]]
    frame = frame.sort_values(["season", "nba_game_id", "team_id"], kind="stable").reset_index(
        drop=True
    )

    quality = {
        "team_games": len(frame),
        "expected_team_games": len(games) * 2,
        "with_player_data": int(frame["player_rows"].notna().sum()),
        "points_reconcile": int(frame["points_reconcile"].fillna(False).sum()),
        "points_mismatch": int((frame["points_reconcile"] == False).sum()),  # noqa: E712
        "box_score_complete": int(frame["box_score_complete"].sum()),
        "possessions_available": int(frame["estimated_possessions"].notna().sum()),
    }
    return frame, quality


def _derive(row: Mapping[str, Any]) -> dict[str, Any]:
    """Possessions, efficiency and four factors for one team-game."""
    blank = {
        "estimated_possessions": None, "offensive_efficiency": None,
        "defensive_efficiency": None, "net_efficiency": None, "estimated_pace": None,
        "efg_pct": None, "turnover_rate": None, "oreb_rate": None, "ft_rate": None,
        "true_shooting_pct": None, "opp_efg_pct": None, "opp_turnover_rate": None,
        "opp_oreb_rate": None, "opp_ft_rate": None,
    }
    # An unreconciled box score must not produce confident-looking efficiency.
    if not row.get("box_score_complete"):
        return blank

    team = {stat: row.get(stat) for stat in COUNTING_STATS}
    opponent = {stat: row.get(f"opp_{stat}") for stat in COUNTING_STATS}
    if any(v is None or pd.isna(v) for v in opponent.values()):
        return blank

    possessions = estimate_possessions(team, opponent)
    if possessions is None or possessions <= 0:
        return blank

    factors = four_factors(team, opponent)
    opp_factors = four_factors(opponent, team)
    offensive = 100.0 * row["pts"] / possessions
    defensive = 100.0 * row["opp_pts"] / possessions
    minutes = row.get("team_minutes")
    # Pace is possessions per 48 minutes of team play; team_minutes is five
    # players' minutes summed, hence the division by five.
    pace = None
    if minutes and minutes > 0:
        pace = 48.0 * possessions / (minutes / 5.0)

    return {
        "estimated_possessions": possessions,
        "offensive_efficiency": offensive,
        "defensive_efficiency": defensive,
        "net_efficiency": offensive - defensive,
        "estimated_pace": pace,
        **factors,
        **{f"opp_{k}": v for k, v in opp_factors.items() if k != "true_shooting_pct"},
    }


def team_game_records(frame: pd.DataFrame) -> Sequence[dict[str, Any]]:
    """Rows as dicts, for the sequential feature builder."""
    return frame.to_dict("records")
