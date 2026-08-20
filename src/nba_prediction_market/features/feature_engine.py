"""Sequential, lookahead-safe pregame features.

Every feature for a game is captured from state that contains **only games
already played before it**. The single ordering rule is enforced here:

* games are processed in ascending ``game_datetime_utc`` -- the trusted, actual
  tipoff. BALLDONTLIE's scheduled ``date`` is never used to sequence anything,
  because 49 postponed games have a ``date`` up to 116 days before their actual
  tipoff.
* a game's row is emitted **before** its own result touches any state.

Current-season state (record, rolling form, point differential) resets at each
season boundary; it is never carried across an offseason. A team with no prior
games this season gets explicit nulls rather than fabricated statistics.

Elo is deliberately *not* computed here. It is a separate configurable system
(:mod:`nba_prediction_market.features.elo`) whose history policy is chosen by
validation, and is joined on afterwards.

Only regular-season games update this state. Playoffs, play-in, and NBA Cup
finals are excluded so the baseline measures regular-season form; studying their
effect later remains possible because the source tables retain them.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Final

#: Rolling window lengths built for every team.
ROLLING_WINDOWS: Final[tuple[int, ...]] = (5, 10)

#: A back-to-back is a game played within this many days of the previous one.
BACK_TO_BACK_MAX_DAYS: Final = 1.5

FEATURE_COLUMNS: Final[tuple[str, ...]] = (
    # identity / context
    "nba_game_id",
    "season",
    "game_datetime_utc",
    "home_franchise_id",
    "away_franchise_id",
    "home_team",
    "away_team",
    # target
    "home_win",
    # current-season record
    "home_games_played",
    "away_games_played",
    "home_win_pct_prior",
    "away_win_pct_prior",
    "win_pct_diff",
    # rolling form
    "home_last5_win_pct",
    "away_last5_win_pct",
    "last5_win_pct_diff",
    "home_last10_win_pct",
    "away_last10_win_pct",
    "last10_win_pct_diff",
    # rolling point differential
    "home_last5_point_diff",
    "away_last5_point_diff",
    "last5_point_diff_difference",
    "home_last10_point_diff",
    "away_last10_point_diff",
    "last10_point_diff_difference",
    # rest
    "home_rest_days",
    "away_rest_days",
    "rest_days_diff",
    "home_back_to_back",
    "away_back_to_back",
)


@dataclass
class TeamSeasonState:
    """A team's within-season history, all of it strictly prior to the current game."""

    games_played: int = 0
    wins: int = 0
    recent_results: deque[int] = field(default_factory=lambda: deque(maxlen=max(ROLLING_WINDOWS)))
    recent_point_diffs: deque[int] = field(
        default_factory=lambda: deque(maxlen=max(ROLLING_WINDOWS))
    )

    def win_pct(self) -> float | None:
        return None if self.games_played == 0 else self.wins / self.games_played

    def rolling_win_pct(self, window: int) -> float | None:
        if not self.recent_results:
            return None
        recent = list(self.recent_results)[-window:]
        return sum(recent) / len(recent)

    def rolling_point_diff(self, window: int) -> float | None:
        if not self.recent_point_diffs:
            return None
        recent = list(self.recent_point_diffs)[-window:]
        return sum(recent) / len(recent)

    def record(self, won: bool, point_diff: int) -> None:
        self.games_played += 1
        self.wins += int(won)
        self.recent_results.append(int(won))
        self.recent_point_diffs.append(point_diff)


def _diff(home: float | None, away: float | None) -> float | None:
    """Home-minus-away, or ``None`` when either side is unavailable."""
    if home is None or away is None:
        return None
    return home - away


def _rest_days(previous: datetime | None, current: datetime) -> float | None:
    """Days since the team's previous actual tipoff, or ``None`` for a first game."""
    if previous is None:
        return None
    return (current - previous).total_seconds() / 86400.0


def build_features(games: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Build one pregame feature row per game, in chronological order.

    ``games`` must be regular-season, modelling-eligible rows sorted by
    ``game_datetime_utc``. Sorting is asserted rather than performed, so an
    unordered input fails instead of silently producing leaky features.
    """
    _assert_sorted(games)

    season_state: dict[Any, TeamSeasonState] = {}
    last_played: dict[Any, datetime] = {}
    current_season: int | None = None
    rows: list[dict[str, Any]] = []

    for game in games:
        season = int(game["season"])
        if season != current_season:
            # Neither current-season form nor rest survives the offseason. Not
            # resetting last_played would give a team's season opener ~150 days
            # of "rest", which is meaningless as a fatigue signal.
            season_state = {}
            last_played = {}
            current_season = season

        home_id, away_id = game["home_franchise_id"], game["away_franchise_id"]
        home = season_state.setdefault(home_id, TeamSeasonState())
        away = season_state.setdefault(away_id, TeamSeasonState())
        tipoff = game["game_datetime_utc"]

        home_rest = _rest_days(last_played.get(home_id), tipoff)
        away_rest = _rest_days(last_played.get(away_id), tipoff)

        row: dict[str, Any] = {
            "nba_game_id": game["nba_game_id"],
            "season": season,
            "game_datetime_utc": tipoff,
            "home_franchise_id": home_id,
            "away_franchise_id": away_id,
            "home_team": game["home_team"],
            "away_team": game["away_team"],
            "home_win": game["home_win"],
            "home_games_played": home.games_played,
            "away_games_played": away.games_played,
            "home_win_pct_prior": home.win_pct(),
            "away_win_pct_prior": away.win_pct(),
            "home_rest_days": home_rest,
            "away_rest_days": away_rest,
            # A first game of the season has no previous game, so it is not a
            # back-to-back -- and its rest is null, never an offseason length.
            "home_back_to_back": bool(
                home_rest is not None and home_rest <= BACK_TO_BACK_MAX_DAYS
            ),
            "away_back_to_back": bool(
                away_rest is not None and away_rest <= BACK_TO_BACK_MAX_DAYS
            ),
        }
        row["win_pct_diff"] = _diff(row["home_win_pct_prior"], row["away_win_pct_prior"])
        row["rest_days_diff"] = _diff(home_rest, away_rest)

        for window in ROLLING_WINDOWS:
            hw, aw = home.rolling_win_pct(window), away.rolling_win_pct(window)
            hp, ap = home.rolling_point_diff(window), away.rolling_point_diff(window)
            row[f"home_last{window}_win_pct"] = hw
            row[f"away_last{window}_win_pct"] = aw
            row[f"last{window}_win_pct_diff"] = _diff(hw, aw)
            row[f"home_last{window}_point_diff"] = hp
            row[f"away_last{window}_point_diff"] = ap
            row[f"last{window}_point_diff_difference"] = _diff(hp, ap)

        rows.append({key: row.get(key) for key in FEATURE_COLUMNS})

        # --- everything below mutates state and must stay after the append ---
        home_win = game["home_win"]
        if home_win is None:
            continue
        margin = int(game["home_score"]) - int(game["away_score"])
        home.record(bool(home_win), margin)
        away.record(not bool(home_win), -margin)
        last_played[home_id] = tipoff
        last_played[away_id] = tipoff

    return rows


def _assert_sorted(games: Sequence[Mapping[str, Any]]) -> None:
    previous: datetime | None = None
    for game in games:
        tipoff = game.get("game_datetime_utc")
        if tipoff is None:
            raise ValueError(
                f"game {game.get('nba_game_id')} has no game_datetime_utc; it cannot be "
                "sequenced. Resolve or exclude it explicitly."
            )
        if previous is not None and tipoff < previous:
            raise ValueError(
                "games must be sorted by actual game_datetime_utc before building "
                f"features; {game.get('nba_game_id')} is out of order"
            )
        previous = tipoff
