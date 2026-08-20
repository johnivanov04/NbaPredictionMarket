"""Phase 3A2 team-strength features, all built from prior games only.

Every family here follows the same discipline as the Phase 3A1 engine: a game's
row is emitted before its own result touches any state, and sequencing uses the
trusted ``game_datetime_utc`` rather than the scheduled ``date``.

Families:

* **Strength of schedule.** When a game is played its opponents' *pregame* Elo
  ratings are recorded. Averages are then taken over those stored values, so an
  old game is never re-scored with an opponent's later rating.
* **Scoring.** Points scored, allowed, and margin -- season-to-date, rolling, and
  exponentially weighted. Deliberately *not* called offensive/defensive rating:
  without possession counts these are per-game, not per-possession.
* **League environment.** Scoring compared against the running league average of
  games already played, never a final-season average.
* **Home/away splits.** Performance in the venue role the team is about to
  occupy, using only prior games in that role.
* **Schedule fatigue.** Game counts in trailing windows measured from actual
  tipoffs, so a postponement moves the load to when the game was really played.

State resets at each season boundary.
"""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Final

ROLLING_WINDOWS: Final[tuple[int, ...]] = (5, 10)
FATIGUE_WINDOWS_DAYS: Final[tuple[int, ...]] = (3, 5, 7)
DEFAULT_EWMA_HALF_LIFE: Final = 5.0
#: A back-to-back: a game within this many days of the previous one.
BACK_TO_BACK_MAX_DAYS: Final = 1.5


def _diff(home: float | None, away: float | None) -> float | None:
    if home is None or away is None:
        return None
    return home - away


def _mean(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None


@dataclass
class TeamStrengthState:
    """One team's within-season history."""

    ewma_half_life: float = DEFAULT_EWMA_HALF_LIFE
    opponent_elos: list[float] = field(default_factory=list)
    points_for: list[float] = field(default_factory=list)
    points_against: list[float] = field(default_factory=list)
    margins: list[float] = field(default_factory=list)
    home_results: list[tuple[int, float]] = field(default_factory=list)
    away_results: list[tuple[int, float]] = field(default_factory=list)
    tipoffs: deque[datetime] = field(default_factory=lambda: deque(maxlen=32))

    # --- strength of schedule ---
    def avg_opponent_elo(self, window: int | None = None) -> float | None:
        values = self.opponent_elos if window is None else self.opponent_elos[-window:]
        return _mean(values)

    # --- scoring ---
    def season_points_for(self) -> float | None:
        return _mean(self.points_for)

    def season_points_against(self) -> float | None:
        return _mean(self.points_against)

    def season_margin(self) -> float | None:
        return _mean(self.margins)

    def rolling(self, values: list[float], window: int) -> float | None:
        return _mean(values[-window:])

    def ewma(self, values: list[float]) -> float | None:
        """Exponentially weighted mean, most recent game weighted 1.0."""
        if not values:
            return None
        decay = 0.5 ** (1.0 / self.ewma_half_life)
        weight, total, weight_sum = 1.0, 0.0, 0.0
        for value in reversed(values):
            total += weight * value
            weight_sum += weight
            weight *= decay
        return total / weight_sum

    # --- venue splits ---
    def split(self, at_home: bool) -> tuple[float | None, float | None]:
        results = self.home_results if at_home else self.away_results
        if not results:
            return None, None
        return (
            sum(r for r, _ in results) / len(results),
            sum(m for _, m in results) / len(results),
        )

    # --- fatigue ---
    def games_within(self, now: datetime, days: int) -> int:
        cutoff = now - timedelta(days=days)
        return sum(1 for t in self.tipoffs if cutoff <= t < now)

    def record(
        self,
        *,
        opponent_elo: float | None,
        points_for: float,
        points_against: float,
        won: bool,
        at_home: bool,
        tipoff: datetime,
    ) -> None:
        if opponent_elo is not None:
            self.opponent_elos.append(opponent_elo)
        margin = points_for - points_against
        self.points_for.append(points_for)
        self.points_against.append(points_against)
        self.margins.append(margin)
        (self.home_results if at_home else self.away_results).append((int(won), margin))
        self.tipoffs.append(tipoff)


TEAM_STRENGTH_COLUMNS: Final[tuple[str, ...]] = (
    "nba_game_id",
    # strength of schedule
    "home_season_avg_opponent_elo", "away_season_avg_opponent_elo", "season_sos_diff",
    "home_last5_avg_opponent_elo", "away_last5_avg_opponent_elo", "last5_sos_diff",
    "home_last10_avg_opponent_elo", "away_last10_avg_opponent_elo", "last10_sos_diff",
    # season scoring
    "home_season_points_scored", "away_season_points_scored", "season_points_scored_diff",
    "home_season_points_allowed", "away_season_points_allowed", "season_points_allowed_diff",
    "home_season_margin", "away_season_margin", "season_margin_diff",
    # rolling scoring
    "home_last5_points_scored", "away_last5_points_scored", "last5_points_scored_diff",
    "home_last5_points_allowed", "away_last5_points_allowed", "last5_points_allowed_diff",
    "home_last10_points_scored", "away_last10_points_scored", "last10_points_scored_diff",
    "home_last10_points_allowed", "away_last10_points_allowed", "last10_points_allowed_diff",
    # exponentially weighted
    "home_ewma_margin", "away_ewma_margin", "ewma_margin_diff",
    "home_ewma_points_scored", "away_ewma_points_scored", "ewma_points_scored_diff",
    "home_ewma_points_allowed", "away_ewma_points_allowed", "ewma_points_allowed_diff",
    # league-relative
    "home_points_scored_vs_league", "away_points_scored_vs_league",
    "points_scored_vs_league_diff",
    "home_points_allowed_vs_league", "away_points_allowed_vs_league",
    "points_allowed_vs_league_diff",
    "league_avg_points_prior",
    # venue splits
    "home_home_win_pct", "home_home_margin",
    "away_away_win_pct", "away_away_margin",
    "venue_split_win_pct_diff", "venue_split_margin_diff",
    # fatigue
    "home_games_last3d", "away_games_last3d", "games_last3d_diff",
    "home_games_last5d", "away_games_last5d", "games_last5d_diff",
    "home_games_last7d", "away_games_last7d", "games_last7d_diff",
    "home_3_games_in_4_days", "away_3_games_in_4_days",
    "home_4_games_in_6_days", "away_4_games_in_6_days",
)


def build_team_strength_features(
    games: Sequence[Mapping[str, Any]],
    elo_by_game: Mapping[Any, tuple[float, float]],
    *,
    ewma_half_life: float = DEFAULT_EWMA_HALF_LIFE,
) -> list[dict[str, Any]]:
    """Build the Phase 3A2 feature families for chronologically ordered games.

    ``elo_by_game`` maps a game id to its ``(home_pregame_elo, away_pregame_elo)``
    so strength of schedule can store the opponent's rating *as it stood* when the
    game was played.
    """
    state: dict[Any, TeamStrengthState] = {}
    current_season: int | None = None
    league_points_total = 0.0
    league_team_games = 0
    rows: list[dict[str, Any]] = []

    for game in games:
        season = int(game["season"])
        if season != current_season:
            state = {}
            league_points_total = 0.0
            league_team_games = 0
            current_season = season

        home_id, away_id = game["home_franchise_id"], game["away_franchise_id"]
        home = state.setdefault(home_id, TeamStrengthState(ewma_half_life))
        away = state.setdefault(away_id, TeamStrengthState(ewma_half_life))
        tipoff = game["game_datetime_utc"]
        league_average = (
            league_points_total / league_team_games if league_team_games else None
        )

        row: dict[str, Any] = {"nba_game_id": game["nba_game_id"]}

        # --- strength of schedule ---
        for label, window in (("season", None), ("last5", 5), ("last10", 10)):
            h = home.avg_opponent_elo(window)
            a = away.avg_opponent_elo(window)
            row[f"home_{label}_avg_opponent_elo"] = h
            row[f"away_{label}_avg_opponent_elo"] = a
            row[f"{label}_sos_diff"] = _diff(h, a)

        # --- season scoring ---
        for name, getter in (
            ("points_scored", lambda s: s.season_points_for()),
            ("points_allowed", lambda s: s.season_points_against()),
            ("margin", lambda s: s.season_margin()),
        ):
            h, a = getter(home), getter(away)
            row[f"home_season_{name}"] = h
            row[f"away_season_{name}"] = a
            row[f"season_{name}_diff"] = _diff(h, a)

        # --- rolling scoring ---
        for window in ROLLING_WINDOWS:
            for name, values in (
                ("points_scored", "points_for"),
                ("points_allowed", "points_against"),
            ):
                h = home.rolling(getattr(home, values), window)
                a = away.rolling(getattr(away, values), window)
                row[f"home_last{window}_{name}"] = h
                row[f"away_last{window}_{name}"] = a
                row[f"last{window}_{name}_diff"] = _diff(h, a)

        # --- exponentially weighted ---
        for name, attr in (
            ("margin", "margins"),
            ("points_scored", "points_for"),
            ("points_allowed", "points_against"),
        ):
            h = home.ewma(getattr(home, attr))
            a = away.ewma(getattr(away, attr))
            row[f"home_ewma_{name}"] = h
            row[f"away_ewma_{name}"] = a
            row[f"ewma_{name}_diff"] = _diff(h, a)

        # --- league-relative scoring ---
        row["league_avg_points_prior"] = league_average
        for name, getter in (
            ("points_scored", lambda s: s.season_points_for()),
            ("points_allowed", lambda s: s.season_points_against()),
        ):
            h, a = getter(home), getter(away)
            hv = None if h is None or league_average is None else h - league_average
            av = None if a is None or league_average is None else a - league_average
            row[f"home_{name}_vs_league"] = hv
            row[f"away_{name}_vs_league"] = av
            row[f"{name}_vs_league_diff"] = _diff(hv, av)

        # --- venue splits: each team in the role it is about to occupy ---
        home_pct, home_margin = home.split(at_home=True)
        away_pct, away_margin = away.split(at_home=False)
        row["home_home_win_pct"] = home_pct
        row["home_home_margin"] = home_margin
        row["away_away_win_pct"] = away_pct
        row["away_away_margin"] = away_margin
        row["venue_split_win_pct_diff"] = _diff(home_pct, away_pct)
        row["venue_split_margin_diff"] = _diff(home_margin, away_margin)

        # --- fatigue, from actual tipoffs ---
        for days in FATIGUE_WINDOWS_DAYS:
            h = home.games_within(tipoff, days)
            a = away.games_within(tipoff, days)
            row[f"home_games_last{days}d"] = h
            row[f"away_games_last{days}d"] = a
            row[f"games_last{days}d_diff"] = h - a
        # Counts exclude the current game, so "3 in 4 days" means two prior.
        row["home_3_games_in_4_days"] = bool(home.games_within(tipoff, 4) >= 2)
        row["away_3_games_in_4_days"] = bool(away.games_within(tipoff, 4) >= 2)
        row["home_4_games_in_6_days"] = bool(home.games_within(tipoff, 6) >= 3)
        row["away_4_games_in_6_days"] = bool(away.games_within(tipoff, 6) >= 3)

        rows.append({key: row.get(key) for key in TEAM_STRENGTH_COLUMNS})

        # --- state updates only below this line ---
        home_win = game.get("home_win")
        home_score, away_score = game.get("home_score"), game.get("away_score")
        if home_win is None or home_score is None or away_score is None:
            continue
        home_elo, away_elo = elo_by_game.get(game["nba_game_id"], (None, None))
        home.record(
            opponent_elo=away_elo, points_for=float(home_score),
            points_against=float(away_score), won=bool(home_win),
            at_home=True, tipoff=tipoff,
        )
        away.record(
            opponent_elo=home_elo, points_for=float(away_score),
            points_against=float(home_score), won=not bool(home_win),
            at_home=False, tipoff=tipoff,
        )
        league_points_total += float(home_score) + float(away_score)
        league_team_games += 2

    return rows


def ewma_decay(half_life: float) -> float:
    """Per-game decay implied by a half-life, exposed for tests and reporting."""
    if half_life <= 0:
        raise ValueError(f"half_life must be positive, got {half_life}")
    return math.pow(0.5, 1.0 / half_life)
