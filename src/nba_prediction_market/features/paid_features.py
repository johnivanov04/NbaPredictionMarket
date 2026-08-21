"""Lookahead-safe pregame features from the paid feeds.

Same discipline as every earlier engine: a game's row is emitted **before** its
own result touches any state, and sequencing uses the trusted actual
``game_datetime_utc``. Everything measured in :mod:`team_box` and
:mod:`rotation` describes a completed game and can therefore only influence
*later* games.

Three families are produced, each as a home-minus-away difference plus the
underlying per-team values:

* **Efficiency** -- season-to-date, rolling, and exponentially weighted
  possession-adjusted efficiency and four factors.
* **League-relative efficiency** -- the same values expressed against the
  league average *of games already played*, so a 2006 offence is not judged
  against 2025 scoring.
* **Rotation** -- minute-share shape, disruption, and shrunk player quality.

State resets at each season boundary.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Final

from nba_prediction_market.features.rotation import ROTATION_FEATURES, TeamRotationState

#: Efficiency measures carried forward from each completed team-game.
EFFICIENCY_METRICS: Final[tuple[str, ...]] = (
    "offensive_efficiency", "defensive_efficiency", "net_efficiency",
    "estimated_pace", "efg_pct", "turnover_rate", "oreb_rate", "ft_rate",
)
ROLLING_WINDOWS: Final[tuple[int, ...]] = (3, 5, 10)
DEFAULT_EWMA_HALF_LIFE: Final = 5.0
#: League-relative treatment only makes sense for the scoring-rate metrics.
LEAGUE_RELATIVE_METRICS: Final[tuple[str, ...]] = (
    "offensive_efficiency", "defensive_efficiency", "efg_pct",
)


def _feature_names() -> tuple[str, ...]:
    names: list[str] = ["nba_game_id"]
    for metric in EFFICIENCY_METRICS:
        names += [f"home_season_{metric}", f"away_season_{metric}", f"season_{metric}_diff"]
        for window in ROLLING_WINDOWS:
            names += [
                f"home_last{window}_{metric}",
                f"away_last{window}_{metric}",
                f"last{window}_{metric}_diff",
            ]
        names += [f"home_ewma_{metric}", f"away_ewma_{metric}", f"ewma_{metric}_diff"]
    for metric in LEAGUE_RELATIVE_METRICS:
        names += [
            f"home_{metric}_vs_league",
            f"away_{metric}_vs_league",
            f"{metric}_vs_league_diff",
        ]
    names.append("league_avg_offensive_efficiency_prior")
    for feature in ROTATION_FEATURES:
        names += [f"home_{feature}", f"away_{feature}", f"{feature}_diff"]
    names.append("both_teams_box_score_complete_history")
    return tuple(names)


PAID_FEATURE_COLUMNS: Final[tuple[str, ...]] = _feature_names()


def _mean(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _diff(home: float | None, away: float | None) -> float | None:
    if home is None or away is None:
        return None
    return home - away


@dataclass
class TeamEfficiencyState:
    """Completed-game efficiency measures for one team this season."""

    ewma_half_life: float = DEFAULT_EWMA_HALF_LIFE
    history: dict[str, list[float]] = field(default_factory=dict)
    complete_games: int = 0
    incomplete_games: int = 0

    def season(self, metric: str) -> float | None:
        return _mean(self.history.get(metric, []))

    def rolling(self, metric: str, window: int) -> float | None:
        return _mean(self.history.get(metric, [])[-window:])

    def ewma(self, metric: str) -> float | None:
        values = self.history.get(metric, [])
        if not values:
            return None
        decay = 0.5 ** (1.0 / self.ewma_half_life)
        weight, total, weight_sum = 1.0, 0.0, 0.0
        for value in reversed(values):
            total += weight * value
            weight_sum += weight
            weight *= decay
        return total / weight_sum

    def record(self, observation: Mapping[str, Any]) -> None:
        """Fold one completed team-game in, skipping unreconciled box scores."""
        if not observation.get("box_score_complete"):
            self.incomplete_games += 1
            return
        for metric in EFFICIENCY_METRICS:
            value = observation.get(metric)
            if value is not None and value == value:  # not NaN
                self.history.setdefault(metric, []).append(float(value))
        self.complete_games += 1


def build_paid_features(
    games: Sequence[Mapping[str, Any]],
    team_games: Mapping[tuple[Any, Any], Mapping[str, Any]],
    appearances: Mapping[tuple[Any, Any], Mapping[Any, tuple[float | None, float | None]]],
    *,
    ewma_half_life: float = DEFAULT_EWMA_HALF_LIFE,
) -> list[dict[str, Any]]:
    """One pregame row per game, from prior completed games only.

    ``team_games`` maps ``(game_id, team_id)`` to that team's completed-game
    measures; ``appearances`` maps the same key to ``player -> (minutes,
    plus_minus)``. Both are consumed only *after* a game's row is emitted.
    """
    efficiency: dict[Any, TeamEfficiencyState] = {}
    rotation: dict[Any, TeamRotationState] = {}
    current_season: int | None = None
    league_offense: list[float] = []
    league_efg: list[float] = []
    rows: list[dict[str, Any]] = []

    for game in games:
        season = int(game["season"])
        if season != current_season:
            efficiency, rotation = {}, {}
            league_offense, league_efg = [], []
            current_season = season

        home_id, away_id = game["home_franchise_id"], game["away_franchise_id"]
        home_eff = efficiency.setdefault(home_id, TeamEfficiencyState(ewma_half_life))
        away_eff = efficiency.setdefault(away_id, TeamEfficiencyState(ewma_half_life))
        home_rot = rotation.setdefault(home_id, TeamRotationState())
        away_rot = rotation.setdefault(away_id, TeamRotationState())

        row: dict[str, Any] = {"nba_game_id": game["nba_game_id"]}
        for metric in EFFICIENCY_METRICS:
            h, a = home_eff.season(metric), away_eff.season(metric)
            row[f"home_season_{metric}"] = h
            row[f"away_season_{metric}"] = a
            row[f"season_{metric}_diff"] = _diff(h, a)
            for window in ROLLING_WINDOWS:
                hw, aw = home_eff.rolling(metric, window), away_eff.rolling(metric, window)
                row[f"home_last{window}_{metric}"] = hw
                row[f"away_last{window}_{metric}"] = aw
                row[f"last{window}_{metric}_diff"] = _diff(hw, aw)
            he, ae = home_eff.ewma(metric), away_eff.ewma(metric)
            row[f"home_ewma_{metric}"] = he
            row[f"away_ewma_{metric}"] = ae
            row[f"ewma_{metric}_diff"] = _diff(he, ae)

        league = {
            "offensive_efficiency": _mean(league_offense),
            "defensive_efficiency": _mean(league_offense),
            "efg_pct": _mean(league_efg),
        }
        row["league_avg_offensive_efficiency_prior"] = league["offensive_efficiency"]
        for metric in LEAGUE_RELATIVE_METRICS:
            baseline = league[metric]
            h, a = home_eff.season(metric), away_eff.season(metric)
            hv = None if h is None or baseline is None else h - baseline
            av = None if a is None or baseline is None else a - baseline
            row[f"home_{metric}_vs_league"] = hv
            row[f"away_{metric}_vs_league"] = av
            row[f"{metric}_vs_league_diff"] = _diff(hv, av)

        home_features, away_features = home_rot.features(), away_rot.features()
        for feature in ROTATION_FEATURES:
            h, a = home_features[feature], away_features[feature]
            row[f"home_{feature}"] = h
            row[f"away_{feature}"] = a
            row[f"{feature}_diff"] = _diff(h, a)

        row["both_teams_box_score_complete_history"] = bool(
            home_eff.incomplete_games == 0 and away_eff.incomplete_games == 0
        )
        rows.append({key: row.get(key) for key in PAID_FEATURE_COLUMNS})

        # --- state updates only below this line ---
        for team_id, state, rot in (
            (home_id, home_eff, home_rot), (away_id, away_eff, away_rot)
        ):
            observation = team_games.get((game["nba_game_id"], team_id))
            if observation is not None:
                state.record(observation)
                if observation.get("box_score_complete"):
                    if observation.get("offensive_efficiency") is not None:
                        league_offense.append(float(observation["offensive_efficiency"]))
                    if observation.get("efg_pct") is not None:
                        league_efg.append(float(observation["efg_pct"]))
            played = appearances.get((game["nba_game_id"], team_id))
            if played:
                rot.record(played)

    return rows
