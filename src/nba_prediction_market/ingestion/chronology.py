"""Chronology policy for sequential features.

This module declares *how* time is allowed to be used downstream. It builds no
features; it exists so the rules are stated and tested once, before Elo or rest
features are written against them.

**Ordering invariant.** The only valid sort key is ``game_datetime_utc`` (after
corrections). BALLDONTLIE's ``date`` is the *scheduled* date and was never
updated for postponed games -- 49 games in 2020-21..2022-23 diverge from the
actual tipoff by up to 116 days. Ordering by ``date`` produces five physically
impossible cases of a team playing twice in one day; ordering by tipoff produces
none.

**Rest-day invariant.** Rest between consecutive games for a team must be
computed from the actual played chronology, never from scheduled dates. A game
postponed from January to May gives the team a long rest before the May game and
must not appear as a January back-to-back.

**Date-only precision.** A record verified only to a calendar date has no tipoff
instant. Such a record may still be ordered within a day, but any feature that
reads *other* games must not treat it as ordered against same-day games, since
its position within the day is unknown. None exist today -- every historical
record resolves to an exact timestamp -- but the rule is stated so a future
date-only correction cannot silently create leakage.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from datetime import datetime, timedelta
from typing import Any, Final

from nba_prediction_market.ingestion.source_corrections import (
    PRECISION_DATE_ONLY_VERIFIED,
    PRECISION_EXACT_DATETIME,
)

#: The one column that may order games. See the module docstring.
ORDERING_FIELD: Final = "game_datetime_utc"
#: Never a sort key -- retained only as source provenance.
SCHEDULED_DATE_FIELD: Final = "date"

#: Precisions that may be placed in a sequence at all.
ORDERABLE_PRECISIONS: Final = frozenset(
    {PRECISION_EXACT_DATETIME, PRECISION_DATE_ONLY_VERIFIED}
)


class ChronologyError(RuntimeError):
    """Raised when a sequence cannot be ordered safely."""


def rest_days(previous_tipoff: datetime, next_tipoff: datetime) -> float:
    """Days of rest between two of a team's consecutive games.

    Computed from actual played tipoffs, in UTC. Both inputs must be
    timezone-aware; naive datetimes are refused rather than assumed to be UTC,
    because a silent zone assumption would shift back-to-backs.

    A back-to-back (played on consecutive calendar days, roughly 24h apart)
    yields ~1.0.
    """
    for name, value in (("previous_tipoff", previous_tipoff), ("next_tipoff", next_tipoff)):
        if value.tzinfo is None:
            raise ChronologyError(f"{name} must be timezone-aware; refusing to guess a zone")
    delta = next_tipoff - previous_tipoff
    if delta < timedelta(0):
        raise ChronologyError(
            f"next_tipoff {next_tipoff} precedes previous_tipoff {previous_tipoff}; "
            "the sequence is not ordered by actual played chronology"
        )
    return delta.total_seconds() / 86400.0


def team_game_sequence(
    games: Iterable[Mapping[str, Any]], team: str
) -> list[Mapping[str, Any]]:
    """A team's games ordered by actual tipoff.

    Raises if any game lacks an orderable timestamp -- sequential features must
    never silently skip a game, because a gap changes every rest value after it.
    """
    selected = [
        g for g in games if g.get("home_team") == team or g.get("away_team") == team
    ]
    for game in selected:
        if game.get(ORDERING_FIELD) is None:
            raise ChronologyError(
                f"game {game.get('nba_game_id')} has no {ORDERING_FIELD}; it cannot be "
                "placed in a sequence. Resolve or exclude it explicitly."
            )
    return sorted(selected, key=lambda g: g[ORDERING_FIELD])


def rest_days_for_sequence(
    games: Iterable[Mapping[str, Any]], team: str
) -> list[tuple[Any, float | None]]:
    """``(game_id, rest_days)`` for a team, from actual played chronology.

    The first game of the sequence has no predecessor and yields ``None``.
    """
    ordered = team_game_sequence(games, team)
    out: list[tuple[Any, float | None]] = []
    previous: datetime | None = None
    for game in ordered:
        tipoff = game[ORDERING_FIELD]
        out.append(
            (game.get("nba_game_id"), None if previous is None else rest_days(previous, tipoff))
        )
        previous = tipoff
    return out
