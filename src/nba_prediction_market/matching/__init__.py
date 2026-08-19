"""Deterministic normalization and NBA-game <-> Kalshi-market matching."""

from nba_prediction_market.matching.team_names import (
    AMBIGUOUS_TEAM_STRINGS,
    NBA_TEAMS,
    TeamResolution,
    is_canonical,
    normalize_team,
    resolve_team,
)

__all__ = [
    "AMBIGUOUS_TEAM_STRINGS",
    "NBA_TEAMS",
    "TeamResolution",
    "is_canonical",
    "normalize_team",
    "resolve_team",
]
