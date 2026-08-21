"""Normalize paid BALLDONTLIE player-game and advanced-stat feeds.

Both feeds are **player-level, per game**. There is no team-level advanced row:
a game's twelve rows carry twelve distinct paces and ratings, because each is an
on-court estimate for that player. Team-level quantities must therefore be
aggregated deliberately (minutes-weighted), not read off a single row.

**Every field here is post-game.** These describe what happened *in* a game and
are safe as model inputs only when lagged to a team's prior games. Nothing in
this module may be attached to the game it came from.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Final

import pandas as pd

logger = logging.getLogger(__name__)

#: Columns kept from ``/v1/stats``.
PLAYER_GAME_COLUMNS: Final[tuple[str, ...]] = (
    "nba_game_id", "season", "game_date", "player_id", "player_name",
    "team_id", "team_abbreviation", "is_home", "minutes",
    "pts", "reb", "oreb", "dreb", "ast", "stl", "blk", "turnover", "pf",
    "fgm", "fga", "fg_pct", "fg3m", "fg3a", "fg3_pct", "ftm", "fta", "ft_pct",
    "plus_minus",
)

#: Columns kept from ``/v1/stats/advanced`` (available for all 20 seasons).
ADVANCED_COLUMNS: Final[tuple[str, ...]] = (
    "nba_game_id", "season", "game_date", "player_id", "team_id",
    "team_abbreviation", "is_home",
    "pace", "offensive_rating", "defensive_rating", "net_rating", "pie",
    "true_shooting_percentage", "effective_field_goal_percentage",
    "usage_percentage", "assist_percentage", "assist_ratio", "assist_to_turnover",
    "turnover_ratio", "offensive_rebound_percentage",
    "defensive_rebound_percentage", "rebound_percentage",
)

#: Fractions that must lie in [0, 1]; anything outside is rejected as malformed.
BOUNDED_FRACTIONS: Final[tuple[str, ...]] = (
    "fg_pct", "fg3_pct", "ft_pct", "true_shooting_percentage",
    "effective_field_goal_percentage", "usage_percentage", "assist_percentage",
    "offensive_rebound_percentage", "defensive_rebound_percentage",
    "rebound_percentage",
)

_MINUTES = re.compile(r"^(\d+)(?::(\d+))?$")


def parse_minutes(value: Any) -> float | None:
    """Parse ``"34:12"`` or ``"34"`` into decimal minutes.

    Returns ``None`` for blanks or anything unparseable -- a player who did not
    appear has no minutes, which is different from zero.
    """
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        # Verified against 2006-07: rows with a null ``min`` have every other
        # stat null too. They are absent observations, not zero-minute games.
        return None
    if text in {"0:00", "00:00"}:
        return 0.0
    match = _MINUTES.match(text)
    if not match:
        try:
            return float(text)
        except ValueError:
            return None
    minutes = int(match.group(1))
    seconds = int(match.group(2) or 0)
    return minutes + seconds / 60.0


def _number(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return None if number != number else number


def _fraction(value: Any) -> float | None:
    """A bounded fraction, or ``None`` when out of range."""
    number = _number(value)
    if number is None:
        return None
    if not 0.0 <= number <= 1.0:
        logger.warning("Discarding out-of-range fraction %r", value)
        return None
    return number


def _identity(record: dict[str, Any]) -> dict[str, Any]:
    """Shared identity fields, including home/away from the embedded game."""
    game = record.get("game") or {}
    team = record.get("team") or {}
    player = record.get("player") or {}
    team_id = team.get("id")
    home_id = game.get("home_team_id")
    if home_id is None:
        home = game.get("home_team") or {}
        home_id = home.get("id")
    name = " ".join(
        part for part in (player.get("first_name"), player.get("last_name")) if part
    ).strip()
    return {
        "nba_game_id": game.get("id"),
        "season": game.get("season"),
        "game_date": game.get("date"),
        "player_id": player.get("id"),
        "player_name": name or None,
        "team_id": team_id,
        "team_abbreviation": team.get("abbreviation"),
        "is_home": None if (team_id is None or home_id is None) else team_id == home_id,
    }


def normalize_player_game(record: dict[str, Any]) -> dict[str, Any]:
    """Flatten one ``/v1/stats`` record."""
    row = _identity(record)
    row["minutes"] = parse_minutes(record.get("min"))
    for field in ("pts", "reb", "oreb", "dreb", "ast", "stl", "blk", "pf",
                  "fgm", "fga", "fg3m", "fg3a", "ftm", "fta", "plus_minus"):
        row[field] = _number(record.get(field))
    row["turnover"] = _number(record.get("turnover"))
    for field in ("fg_pct", "fg3_pct", "ft_pct"):
        row[field] = _fraction(record.get(field))
    return {key: row.get(key) for key in PLAYER_GAME_COLUMNS}


def normalize_advanced(record: dict[str, Any]) -> dict[str, Any]:
    """Flatten one ``/v1/stats/advanced`` record."""
    row = _identity(record)
    for field in ("pace", "offensive_rating", "defensive_rating", "net_rating",
                  "pie", "assist_ratio", "assist_to_turnover", "turnover_ratio"):
        row[field] = _number(record.get(field))
    for field in ("true_shooting_percentage", "effective_field_goal_percentage",
                  "usage_percentage", "assist_percentage",
                  "offensive_rebound_percentage", "defensive_rebound_percentage",
                  "rebound_percentage"):
        row[field] = _fraction(record.get(field))
    return {key: row.get(key) for key in ADVANCED_COLUMNS}


def build_frame(
    records: list[dict[str, Any]], kind: str
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Normalize and de-duplicate a feed, reporting what was dropped.

    De-duplication is on ``(game, player)``: the same player cannot legitimately
    have two rows for one game, and a repeat would double-count them in any
    aggregate.
    """
    if kind == "player_game":
        normalize, columns = normalize_player_game, PLAYER_GAME_COLUMNS
    elif kind == "advanced":
        normalize, columns = normalize_advanced, ADVANCED_COLUMNS
    else:
        raise ValueError(f"unknown feed kind {kind!r}")

    rows = [normalize(r) for r in records]
    malformed = [
        r for r in rows if r["nba_game_id"] is None or r["player_id"] is None
    ]
    usable = [r for r in rows if r["nba_game_id"] is not None and r["player_id"] is not None]

    frame = pd.DataFrame(usable, columns=list(columns))
    before = len(frame)
    frame = frame.drop_duplicates(subset=["nba_game_id", "player_id"], keep="first")
    duplicates = before - len(frame)

    frame = frame.sort_values(
        ["season", "nba_game_id", "team_id", "player_id"], kind="stable"
    ).reset_index(drop=True)
    return frame, {
        "records_in": len(records),
        "rows_out": len(frame),
        "malformed_dropped": len(malformed),
        "duplicate_game_player_rows": duplicates,
    }
